"""
Self-play training for Sequence AI.
Robust multiprocessing architecture for high throughput on multi-core systems.

Architecture:
- N worker processes, each running M games independently
- Central inference server on GPU
- Workers batch locally before sending to server (reduces IPC frequency)
- Per-worker result queues (no dispatcher needed)
"""

import argparse
import json
import importlib.util
import time
import warnings
from pathlib import Path
from collections import deque
from typing import Dict
import random
import queue

import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

# Suppress warnings
warnings.filterwarnings("ignore", message="The given buffer is not writable")
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

from .belief import search_class, unseen_cards
from .inference_transport import InferenceSlot
from .game import Move
from .cards import Card
from .model import create_model, save_model, load_model, STATE_CHANNELS


def require_native_extension() -> None:
    """Fail early with a useful message when the Rust training backend is missing."""
    if importlib.util.find_spec("sequence_native") is None:
        raise RuntimeError(
            "The sequence_native Rust extension is required for multiprocess training. "
            "Build it with: pip install ."
        )


def require_cuda() -> None:
    """Fail fast instead of silently running training on CPU."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "CUDA is required for training. Install a CUDA-enabled PyTorch build "
            "and run with a visible NVIDIA GPU."
        )


def move_optimizer_state_to_device(
    optimizer: optim.Optimizer, device: torch.device
) -> None:
    """Move restored optimizer tensors to the active training device."""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for key, value in optimizer.state.get(parameter, {}).items():
                if not torch.is_tensor(value):
                    continue
                if key == "step":
                    target = (
                        device
                        if group.get("capturable") or group.get("fused")
                        else torch.device("cpu")
                    )
                    optimizer.state[parameter][key] = value.to(target)
                elif value.shape == parameter.shape:
                    optimizer.state[parameter][key] = value.to(device)
                elif (
                    value.ndim == 4
                    and parameter.ndim == 4
                    and value.shape[0] == parameter.shape[0]
                    and value.shape[2:] == parameter.shape[2:]
                    and value.shape[1] < parameter.shape[1]
                ):
                    migrated = torch.zeros_like(parameter)
                    migrated[:, : value.shape[1]] = value.to(device)
                    optimizer.state[parameter][key] = migrated
                else:
                    raise ValueError(
                        "Optimizer checkpoint shape does not match the model"
                    )


def worker_process(
    worker_id: int,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    game_queue: mp.Queue,
    ready_event: mp.Event,
    args_dict: Dict,
):
    """
    Worker process that runs multiple games independently.
    Batches requests locally before sending to inference server.
    """
    try:
        # Imports inside worker to avoid CUDA forking issues
        from .fast_game import FastSequenceGame

        torch.set_num_threads(1)
        random.seed(args_dict.get("seed", 0) + worker_id)
        np.random.seed(args_dict.get("seed", 0) + worker_id)

        slot = args_dict.get("inference_slot")
        shared_views = slot.views() if slot is not None else None
        num_games = args_dict["games_per_worker"]
        simulations = args_dict["simulations"]

        # Initialize games
        games = [FastSequenceGame() for _ in range(num_games)]
        mcts_list = [
            search_class(args_dict.get("search_mode", "ensemble"))(
                args_dict.get("seed", 0) + worker_id * num_games + i
            )
            for i in range(num_games)
        ]
        game_histories = [[] for _ in range(num_games)]
        move_counts = [0] * num_games

        for g in games:
            g.reset()

        def reset_mcts(idx):
            board_bytes = games[idx].c_state.get_tensor()
            hand_ids = [c.to_int() for c in games[idx].hands[games[idx].current_player]]
            g = games[idx]
            unseen = unseen_cards(g, g.current_player)
            mcts_list[idx].reset(
                board_bytes,
                hand_ids,
                g.current_player,
                g.c_state.get_sequences(),
                unseen,
                len(g.hands[3 - g.current_player]),
            )

        for i in range(num_games):
            reset_mcts(i)

        # Signal ready
        ready_event.set()

        # Main game loop
        while True:
            # Each tree has exactly one outstanding leaf. Batch across independent games.
            for simulation in range(simulations):
                pending = []
                tensors = []
                for i, search in enumerate(mcts_list):
                    leaf, data, value, terminal = search.select_leaf()
                    if terminal:
                        search.backpropagate(leaf, b"", value)
                    else:
                        pending.append((i, leaf))
                        tensors.append(
                            np.frombuffer(data, dtype=np.float32).reshape(
                                STATE_CHANNELS, 10, 10
                            )
                        )
                if pending:
                    observations = np.stack(tensors)
                    # Binary masks can use byte IPC, but extra conversion and batching
                    # costs make the throughput benefit host-dependent.
                    if args_dict.get("compact_observations", False):
                        observations = observations.astype(np.uint8)
                    if shared_views is not None:
                        count = len(observations)
                        np.copyto(shared_views[0][:count], observations)
                        request_queue.put((worker_id, count))
                        if result_queue.get() != count:
                            raise RuntimeError(
                                "Shared inference response size mismatch"
                            )
                        policies, values = (
                            shared_views[1][:count],
                            shared_views[2][:count],
                        )
                    else:
                        request_queue.put((worker_id, observations))
                        policies, values = result_queue.get()
                    for k, (i, leaf) in enumerate(pending):
                        if simulation < (
                            4
                            if args_dict.get("search_mode", "ensemble") == "ensemble"
                            else 1
                        ) and args_dict.get("root_noise", True):
                            noise_fraction = args_dict.get("root_noise_fraction", 0.25)
                            legal = np.flatnonzero(
                                tensors[k][4:].any(axis=0).reshape(-1)
                            )
                            if len(legal):
                                priors = policies[k, legal]
                                priors = (
                                    priors / priors.sum()
                                    if priors.sum() > 0
                                    else np.full(len(legal), 1 / len(legal))
                                )
                                policies[k, legal] = (
                                    1 - noise_fraction
                                ) * priors + noise_fraction * np.random.dirichlet(
                                    np.full(len(legal), 0.3)
                                )
                        mcts_list[i].backpropagate(
                            leaf,
                            policies[k].astype(np.float32).tobytes(),
                            float(values[k]),
                        )

            # All simulations done - make moves for all games
            for i in range(num_games):
                temp = 1.0 if move_counts[i] < 10 else 0.1
                res = mcts_list[i].get_action(temp)

                if res is None:
                    if game_histories[i]:
                        game_queue.put(
                            [(state, pi, 0.0) for state, pi, _ in game_histories[i]]
                        )
                    games[i].reset()
                    game_histories[i] = []
                    move_counts[i] = 0
                    reset_mcts(i)
                    continue

                card_int, r, c, is_rem, policy_bytes = res
                policy = np.frombuffer(policy_bytes, dtype=np.float32).copy()

                # Store state (as numpy to avoid SHM leak)
                state_np = games[i].get_state_numpy(games[i].current_player)
                game_histories[i].append((state_np, policy, games[i].current_player))

                # Apply move
                move = Move(
                    card=Card.from_int(card_int), row=r, col=c, is_removal=(is_rem != 0)
                )
                if not games[i].make_move(move):
                    raise RuntimeError(f"Search returned illegal move: {move}")
                move_counts[i] += 1

                # Check game over
                if games[i].game_over or move_counts[i] >= 200:
                    # Package training data and send to main process
                    training_data = []
                    for state, pi, player in game_histories[i]:
                        value = games[i].get_result(player)
                        training_data.append((state, pi, value))

                    if training_data:
                        game_queue.put(training_data)

                    # Reset for next game
                    games[i].reset()
                    game_histories[i] = []
                    move_counts[i] = 0

                reset_mcts(i)

    except Exception as e:
        import traceback

        print(f"[Worker {worker_id}] ERROR: {e}", flush=True)
        traceback.print_exc()
        raise


class InferenceServer:
    """Batches inference requests from multiple workers."""

    def __init__(
        self,
        model,
        device,
        result_queues: Dict[int, mp.Queue],
        batch_size: int = 512,
        timeout: float = 0.002,
        shared_slots=None,
    ):
        self.model = model
        self.device = device
        self.result_queues = result_queues
        self.batch_size = batch_size
        self.timeout = timeout
        self.total_processed = 0
        self.total_batches = 0
        self.shared_views = {
            i: slot.views() for i, slot in (shared_slots or {}).items()
        }

    def _request(self, item):
        worker, payload = item
        if self.shared_views:
            if worker not in self.shared_views or type(payload) is not int:
                raise ValueError("Invalid shared inference request")
            observations = self.shared_views[worker][0]
            if not 0 < payload <= len(observations):
                raise ValueError("Shared inference request exceeds slot capacity")
            return worker, observations[:payload]
        return item

    def step(self, request_queue: mp.Queue) -> int:
        """Process pending requests. Returns count processed."""
        batch_items = []

        # Collect requests with timeout
        try:
            item = self._request(request_queue.get(timeout=self.timeout))
            batch_items.append(item)
        except queue.Empty:
            return 0

        # Collect more until batch is full or timeout
        start_time = time.monotonic()
        samples = len(item[1])
        while samples < self.batch_size:
            remaining = self.timeout - (time.monotonic() - start_time)
            if remaining <= 0:
                break
            try:
                item = self._request(request_queue.get(timeout=min(remaining, 0.005)))
                batch_items.append(item)
                samples += len(item[1])
            except queue.Empty:
                break

        if not batch_items:
            return 0

        # Build mega-batch
        worker_ids = []
        all_arrays = []
        sizes = []

        for w_id, arr in batch_items:
            worker_ids.append(w_id)
            all_arrays.append(arr)
            sizes.append(arr.shape[0])

        mega_batch = (
            torch.from_numpy(np.concatenate(all_arrays, axis=0))
            .to(self.device, memory_format=torch.channels_last)
            .float()
        )

        # Inference
        self.model.eval()
        with (
            torch.inference_mode(),
            autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ),
        ):
            log_policies, values = self.model(mega_batch)
            policies = log_policies.exp()

        policies = policies.float().cpu().numpy()
        values = values.float().cpu().numpy()

        # Distribute results
        curr = 0
        for i in range(len(batch_items)):
            w_id = worker_ids[i]
            sz = sizes[i]

            p_slice = policies[curr : curr + sz]
            v_slice = values[curr : curr + sz].flatten()

            if self.shared_views:
                _, shared_policies, shared_values = self.shared_views[w_id]
                np.copyto(shared_policies[:sz], p_slice)
                np.copyto(shared_values[:sz], v_slice)
                self.result_queues[w_id].put(sz)
            else:
                self.result_queues[w_id].put((p_slice, v_slice))
            curr += sz

        self.total_processed += sum(sizes)
        self.total_batches += 1

        return sum(sizes)


def train_step(
    model, optimizer, states, target_policies, target_values, scaler=None, device=None
):
    """Perform one training step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    device_type = "cuda" if device and device.type == "cuda" else "cpu"

    with autocast(device_type=device_type, enabled=(scaler is not None)):
        policy_logits, values = model(states)
        policy_loss = -torch.mean(torch.sum(target_policies * policy_logits, dim=1))
        value_loss = torch.mean((values.squeeze(-1) - target_values) ** 2)
        total_loss = policy_loss + value_loss

    optimizer_step_ran = True
    if scaler is not None:
        old_scale = scaler.get_scale()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_ran = scaler.get_scale() >= old_scale
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item(), optimizer_step_ran


class ReplayLearner:
    """A bounded set of updates sampled from the replay buffer for one epoch."""

    def __init__(self, model, optimizer, device, replay, batch_size, steps):
        self.model, self.optimizer, self.device = model, optimizer, device
        self.batch_size, self.remaining = batch_size, steps
        self.losses = []
        self.seconds = 0.0
        batch = random.sample(list(replay), min(len(replay), batch_size * 20))
        self.states = torch.from_numpy(np.stack([s for s, _, _ in batch])).to(
            device, memory_format=torch.channels_last
        )
        self.policies = torch.from_numpy(np.stack([p for _, p, _ in batch])).to(device)
        self.values = torch.tensor(
            [v for _, _, v in batch], device=device, dtype=torch.float32
        )

    def step(self):
        if self.remaining == 0:
            return
        start = time.monotonic()
        indices = torch.randperm(len(self.states), device=self.device)[
            : self.batch_size
        ]
        losses = train_step(
            self.model,
            self.optimizer,
            self.states[indices],
            self.policies[indices],
            self.values[indices],
            device=self.device,
        )
        self.losses.append(losses[:3])
        self.remaining -= 1
        self.seconds += time.monotonic() - start


def train(args):
    """Run bounded self-play/learning epochs and clean up every child on failure."""
    import copy

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mp.set_start_method("spawn", force=True)
    require_native_extension()
    require_cuda()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Training requires a GPU with bfloat16 support (Ampere or newer)"
        )
    models_dir = Path(args.output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "latest.pt"
    checkpoint = None
    if args.init_from:
        # Fine-tuning starts a new optimizer/schedule while retaining trained weights.
        model, device, _ = load_model(
            args.init_from, device="cuda", compile_model=not args.no_compile
        )
    elif checkpoint_path.exists() and not args.fresh:
        model, device, checkpoint = load_model(
            checkpoint_path, device="cuda", compile_model=not args.no_compile
        )
    else:
        model, device = create_model(
            "cuda",
            compile_model=not args.no_compile,
            num_filters=args.num_filters,
            num_res_blocks=args.num_res_blocks,
        )
    model.to(memory_format=torch.channels_last)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if checkpoint and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    if checkpoint and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) if checkpoint else 0
    run_config = {
        **vars(args),
        "start_epoch": start_epoch,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": {
            "num_filters": getattr(model, "_orig_mod", model)
            .input_conv[0]
            .out_channels,
            "num_res_blocks": len(getattr(model, "_orig_mod", model).res_blocks),
        },
        "root_noise": {
            "alpha": 0.3,
            "epsilon": args.root_noise_fraction,
            "normalized_over_legal_targets": True,
        },
    }
    (models_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    # Self-play sees a frozen snapshot while the learner updates its own model.
    actor = (
        copy.deepcopy(getattr(model, "_orig_mod", model)).eval()
        if args.overlap_training
        else model
    )
    writer = SummaryWriter(
        log_dir=models_dir.parent / "runs" / f"{models_dir.name}_{int(time.time())}"
    )
    request_queue, game_queue = mp.Queue(), mp.Queue()
    result_queues = {i: mp.Queue() for i in range(args.workers)}
    ready_events = [mp.Event() for _ in range(args.workers)]
    workers = []
    replay = deque(maxlen=args.buffer_size)
    worker_config = {
        "games_per_worker": args.games_per_worker,
        "simulations": args.simulations,
        "seed": args.seed,
        "root_noise_fraction": args.root_noise_fraction,
        "compact_observations": args.compact_observations,
        "search_mode": args.search_mode,
    }
    shared_slots = (
        {
            i: InferenceSlot(args.games_per_worker, STATE_CHANNELS)
            for i in range(args.workers)
        }
        if args.shared_inference
        else None
    )
    server = InferenceServer(
        actor,
        device,
        result_queues,
        batch_size=args.inference_batch_size,
        shared_slots=shared_slots,
    )
    try:
        for i in range(args.workers):
            process = mp.Process(
                target=worker_process,
                args=(
                    i,
                    request_queue,
                    result_queues[i],
                    game_queue,
                    ready_events[i],
                    {
                        **worker_config,
                        "inference_slot": shared_slots[i] if shared_slots else None,
                    },
                ),
                daemon=True,
            )
            process.start()
            workers.append(process)
        for i, event in enumerate(ready_events):
            if not event.wait(args.worker_start_timeout):
                raise RuntimeError(f"Worker {i} did not initialize in time")
        print(
            f"Workers ready: {len(workers)}; device: {device}; overlap: {args.overlap_training}",
            flush=True,
        )
        for epoch in range(start_epoch, start_epoch + args.epochs):
            start = time.monotonic()
            if args.overlap_training:
                actor.load_state_dict(getattr(model, "_orig_mod", model).state_dict())
            actor.eval()
            learner = None
            if args.overlap_training and len(replay) >= args.batch_size:
                learner = ReplayLearner(
                    model, optimizer, device, replay, args.batch_size, args.train_steps
                )
            collected = 0
            decisive = 0
            while collected < args.games_per_epoch:
                dead = [
                    i for i, process in enumerate(workers) if not process.is_alive()
                ]
                if dead:
                    raise RuntimeError(
                        f"Worker process(es) exited unexpectedly: {dead}"
                    )
                server.step(request_queue)
                # CPU actors work on returned predictions during this GPU update.
                if learner is not None and learner.remaining:
                    learner.step()
                try:
                    while collected < args.games_per_epoch:
                        history = game_queue.get_nowait()
                        replay.extend(history)
                        decisive += int(any(value != 0 for _, _, value in history))
                        collected += 1
                except queue.Empty:
                    pass
            collect_seconds = time.monotonic() - start
            if learner is None and len(replay) >= args.batch_size:
                learner = ReplayLearner(
                    model, optimizer, device, replay, args.batch_size, args.train_steps
                )
            if learner is not None:
                while learner.remaining:
                    learner.step()
                if scheduler.last_epoch < scheduler.T_max:
                    scheduler.step()
            metrics = {
                "epoch": epoch + 1,
                "games": collected,
                "decisive_games": decisive,
                "replay_size": len(replay),
                "collect_seconds": collect_seconds,
                "epoch_seconds": time.monotonic() - start,
                "overlap_training": args.overlap_training,
                "inference_transport": "shared" if args.shared_inference else "queue",
                "inference_samples": server.total_processed,
                "inference_batches": server.total_batches,
            }
            if learner is not None:
                total, policy, value = np.mean(learner.losses, axis=0)
                metrics.update(
                    loss=float(total),
                    policy_loss=float(policy),
                    value_loss=float(value),
                    train_seconds=learner.seconds,
                )
                for name, loss in [
                    ("Total", total),
                    ("Policy", policy),
                    ("Value", value),
                ]:
                    writer.add_scalar("Loss/" + name, loss, epoch + 1)
            writer.add_scalar(
                "Perf/GamesPerSec", collected / metrics["epoch_seconds"], epoch + 1
            )
            writer.add_scalar("Replay/Size", len(replay), epoch + 1)
            writer.add_scalar(
                "Train/LearningRate", optimizer.param_groups[0]["lr"], epoch + 1
            )
            with (models_dir / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(metrics) + "\n")
            save_model(
                model, checkpoint_path, optimizer, epoch + 1, scheduler=scheduler
            )
            print(json.dumps(metrics), flush=True)
            del learner
    except KeyboardInterrupt:
        print(
            "Interrupted; the last completed epoch checkpoint is retained.", flush=True
        )
    finally:
        for process in workers:
            process.terminate()
        for process in workers:
            process.join(timeout=10)
        for q in [request_queue, game_queue, *result_queues.values()]:
            q.cancel_join_thread()
            q.close()
        writer.close()


def evaluate(args):
    """Use the same seeded, color-balanced evaluator as the experiment runner."""
    from scripts.evaluate import evaluate as evaluate_games

    torch.set_num_threads(1)
    checkpoint_path = Path(args.output_dir) / "latest.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    model, device, _ = load_model(checkpoint_path, compile_model=False)
    model.eval().to(memory_format=torch.channels_last)
    result = evaluate_games(
        model, device, args.games, args.simulations, args.seed, args.opponent
    )
    result["checkpoint"] = str(checkpoint_path)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--overlap-training", action="store_true")
    train_parser.add_argument(
        "--shared-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share actor inference buffers; --no-shared-inference benchmarks queue IPC",
    )
    train_parser.add_argument(
        "--compact-observations",
        action="store_true",
        help="Experimental byte IPC; benchmark on your host before enabling",
    )
    train_parser.add_argument("--inference-batch-size", type=int, default=4096)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--root-noise-fraction", type=float, default=0.25)
    train_parser.add_argument("--output-dir", default="models")
    train_parser.add_argument(
        "--init-from",
        help="Initialize weights from a checkpoint with a fresh optimizer/schedule",
    )
    train_parser.add_argument(
        "--search-mode", choices=["ensemble", "single"], default="ensemble"
    )
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--games-per-epoch", type=int, default=50)
    train_parser.add_argument("--simulations", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--buffer-size", type=int, default=50000)
    train_parser.add_argument("--train-steps", type=int, default=20)
    train_parser.add_argument("--lr", type=float, default=0.001)
    train_parser.add_argument("--num-filters", type=int, default=128)
    train_parser.add_argument("--num-res-blocks", type=int, default=6)
    train_parser.add_argument("--fresh", action="store_true")
    train_parser.add_argument("--no-compile", action="store_true")
    train_parser.add_argument("--verbose", action="store_true")
    train_parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker processes"
    )
    train_parser.add_argument(
        "--games-per-worker", type=int, default=64, help="Games per worker process"
    )
    train_parser.add_argument(
        "--worker-start-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for each worker to finish startup",
    )

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--output-dir", default="models")
    eval_parser.add_argument("--games", type=int, default=256)
    eval_parser.add_argument("--seed", type=int, default=104729)
    eval_parser.add_argument(
        "--opponent", choices=["random", "greedy"], default="greedy"
    )
    eval_parser.add_argument("--simulations", type=int, default=50)

    args = parser.parse_args()

    if args.command == "train":
        if args.shared_inference and args.compact_observations:
            parser.error(
                "Byte IPC requires --no-shared-inference; shared slots use float32"
            )
        for name in (
            "epochs",
            "games_per_epoch",
            "simulations",
            "batch_size",
            "buffer_size",
            "train_steps",
            "workers",
            "games_per_worker",
            "num_filters",
            "num_res_blocks",
            "inference_batch_size",
        ):
            if getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        if args.lr <= 0 or args.worker_start_timeout <= 0:
            parser.error("learning rate and worker timeout must be positive")
        if not 0 <= args.root_noise_fraction <= 1:
            parser.error("--root-noise-fraction must be between 0 and 1")
        train(args)
    elif args.command == "eval":
        if args.games <= 0 or args.games % 2 or args.simulations < 0:
            parser.error(
                "evaluation needs a positive even game count and non-negative simulations"
            )
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
