"""Matched-budget, resumable PPO and R-NaD for exact-card policies.

Rust owns complete-game environments and public observations; one GPU batches
inference and updates. Rollouts are strictly on-policy and discarded after two
epochs (one complete-episode pass for R-NaD). All contenders share pretrained weights and receive independent Adam
states, retained across alternating training blocks. Monitoring is never an
automatic promotion to the validated production champion.
"""

import argparse
import fcntl
import json
import math
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.arena import summarize
from scripts.overnight import atomic_copy, atomic_json, digest
from src.cards import Deck
from src.exact_policy import ExactPolicy, load_exact_policy, save_exact_policy
from src.history_policy import HistoryPolicy, load_policy

RECIPE = {
    "contenders": ["cnn_ppo", "transformer_ppo", "cnn_rnad"],
    "games_per_iteration": 512,
    "batch_size": 1024,
    "epochs": 2,
    "trunk_lr": 1e-5,
    "new_lr": 1e-4,
    "minimum_lr_fraction": 0.3,
    "schedule_training_seconds": 10800,
    "entropy_initial": 0.003,
    "entropy_final": 0.002,
    "ppo_clip": 0.1,
    "target_kl": 0.01,
    "value_coefficient": 0.5,
    "gradient_clip": 1.0,
    "opponents": "50% current self-play,25% frozen GRU reference,25% historical exact policy; balanced learner seats",
    "historical_pool_size": 3,
    "checkpoint_seconds": 60,
    "seed": 2100000001,
    "monitor_seed": 2100000000001,
    "final_confirmation_seed_reserved": 9100000000001,
    "replay": "none; fresh on-policy trajectories per update",
    "returns": "terminal outcome from each acting player's perspective, undiscounted",
    "rnad_games_per_minibatch": 32,
    "rnad_reference_interval": 4000,
    "rnad_eta": 0.2,
    "rnad_target_avg": 0.001,
    "rnad_gradient_clip": 10000.0,
    "rnad_opponents": "100% current policy both-player self-play; complete episodes",
}
CONTENDERS = {"cnn_ppo": ("cnn", "ppo"), "transformer_ppo": ("transformer", "ppo"), "cnn_rnad": ("cnn", "rnad")}
STOP_REQUESTED = False


def schedule(training_seconds, recipe=RECIPE):
    fraction = min(1.0, max(0.0, training_seconds / recipe["schedule_training_seconds"]))
    scale = recipe["minimum_lr_fraction"] + (1 - recipe["minimum_lr_fraction"]) * (1 + math.cos(math.pi * fraction)) / 2
    entropy = recipe["entropy_initial"] * (1 - fraction) + recipe["entropy_final"] * fraction
    return scale, entropy


def optimizer_for(model, recipe=RECIPE, algorithm="ppo"):
    inherited, new = [], []
    for name, parameter in model.named_parameters():
        (inherited if name.startswith("backbone.") else new).append(parameter)
    groups = [
        {"params": inherited, "lr": recipe["trunk_lr"], "initial_lr": recipe["trunk_lr"], "name": "inherited"},
        {"params": new, "lr": recipe["new_lr"], "initial_lr": recipe["new_lr"], "name": "new"},
    ]
    options = {"betas": (0.0, 0.999), "eps": 1e-7} if algorithm == "rnad" else {}
    return torch.optim.Adam(groups, **options)


def native_environment(count, seed, paired=False):
    import sequence_native

    if not hasattr(sequence_native, "BatchEnv"):
        raise RuntimeError("Build the Rust extension with BatchEnv before exact-policy training")
    if not paired:
        return sequence_native.BatchEnv(count, seed)
    decks = []
    for index in range(count):
        random.seed(seed + index // 2)
        decks.append([card.to_int() for card in Deck().cards])
    return sequence_native.BatchEnv.from_decks(decks)


def public_vector(value):
    return np.frombuffer(value, np.uint8).copy() if isinstance(value, bytes) else np.asarray(value, dtype=np.uint8)


def observations(env, ids, canonical=True):
    raw = env.observe(ids, canonical=canonical)
    count, width = len(ids), int(raw["width"])
    return {
        "board": np.frombuffer(raw["board"], np.uint8).reshape(count, 11, 10, 10).copy(),
        "context": np.frombuffer(raw["context"], np.float32).reshape(count, 157).copy(),
        "history": np.frombuffer(raw["history"], np.uint8).reshape(count, 128, 4).copy(),
        "actions": np.frombuffer(raw["actions"], np.int16).reshape(count, width).copy(),
        "players": public_vector(raw["players"]),
    }


def tensors(data, device, indices=None):
    def get(key, dtype):
        value = data[key] if indices is None else data[key][indices]
        return torch.as_tensor(value).to(device=device, dtype=dtype)

    actions = get("actions", torch.long)
    return (get("board", torch.float32), get("context", torch.float32),
            get("history", torch.long), actions, actions >= 0)


def actor_forward(model, obs, device):
    board, context, history, actions, mask = tensors(obs, device)
    if isinstance(model, ExactPolicy):
        logits, values = model(board, context, history, actions, mask)
    else:
        if isinstance(model, HistoryPolicy):
            spatial, values = model(board, context, history)
        else:
            spatial, values = model(board)
        positions = actions.clamp_min(0) % 100
        logits = spatial.gather(1, positions)
        # Preserve the historical reference's ordinary-card preference when two
        # exact actions share a square; do not alter the learned exact player.
        normal = ((actions.clamp_min(0) // 100) % 13 != 10) & mask
        logits = logits.double() + normal.double() * 1e-12
    return logits.masked_fill(~mask, -torch.inf), values.flatten(), actions, mask


def rollout(model, device, count, seed, opponents=(), full_trajectory=False):
    model.eval()
    for opponent in opponents:
        opponent.eval()
        opponent.requires_grad_(False)
    started = time.monotonic()
    env = native_environment(count, seed)
    active = env.active_ids()
    chunks, record_games, record_players = [], [], []
    old_logs, old_values, sampled_indices = [], [], []
    behavior, record_turns = [], []
    turns = np.zeros(count, dtype=np.int32)
    opponent_moves, inference_seconds, environment_seconds = 0, 0.0, 0.0
    decision_count = 0
    while active:
        info = env.info()
        groups = {}
        for index in active:
            kind = (index // 2) % 4
            opponent_index = kind % len(opponents) if opponents and kind < 2 and info["players"][index] != 1 + index % 2 else -1
            groups.setdefault(opponent_index, []).append(index)
        for opponent_index, ids in groups.items():
            actor = model if opponent_index == -1 else opponents[opponent_index]
            tick = time.monotonic()
            obs = observations(env, ids, isinstance(actor, ExactPolicy))
            environment_seconds += time.monotonic() - tick
            tick = time.monotonic()
            with torch.inference_mode():
                logits, values, actions, _ = actor_forward(actor, obs, device)
                distribution = torch.distributions.Categorical(logits=logits)
                chosen = distribution.sample() if opponent_index == -1 else logits.argmax(1)
                exact = actions.gather(1, chosen[:, None]).flatten().cpu().tolist()
                if opponent_index == -1:
                    old_logs.append(distribution.log_prob(chosen).float().cpu().numpy())
                    old_values.append(values.float().cpu().numpy())
                    sampled_indices.append(chosen.cpu().numpy())
                    if full_trajectory:
                        behavior.append(distribution.probs.float().cpu().numpy())
            inference_seconds += time.monotonic() - tick
            if opponent_index == -1:
                chunks.append(obs)
                record_games.extend(ids)
                record_players.extend(obs["players"].tolist())
                record_turns.extend(turns[ids].tolist())
            else:
                opponent_moves += len(ids)
            tick = time.monotonic()
            env.step(ids, exact)
            turns[ids] += 1
            environment_seconds += time.monotonic() - tick
            decision_count += len(ids)
        active = env.active_ids()
    if not chunks:
        raise RuntimeError("Rollout generated no learner decisions")
    winners = public_vector(env.info()["winners"])
    width = max(chunk["actions"].shape[1] for chunk in chunks)
    data = {key: torch.from_numpy(np.concatenate([chunk[key] for chunk in chunks]))
            for key in ("board", "context", "history")}
    data["actions"] = torch.from_numpy(np.concatenate([
        np.pad(chunk["actions"], ((0, 0), (0, width - chunk["actions"].shape[1])), constant_values=-1)
        for chunk in chunks
    ]))
    data["action"] = torch.from_numpy(np.concatenate(sampled_indices)).long()
    data["old_log_probability"] = torch.from_numpy(np.concatenate(old_logs))
    data["old_value"] = torch.from_numpy(np.concatenate(old_values))
    result = winners[np.asarray(record_games)]
    data["return"] = torch.tensor(np.where(result == 0, 0, np.where(result == np.asarray(record_players), 1, -1)), dtype=torch.float32)
    data["game_index"] = torch.tensor(record_games, dtype=torch.long)
    data["player"] = torch.tensor(record_players, dtype=torch.long) - 1
    data["ply"] = torch.tensor(record_turns, dtype=torch.long)
    if full_trajectory:
        if opponent_moves:
            raise ValueError("RNaD requires complete both-player self-play trajectories")
        data["behavior_policy"] = torch.from_numpy(np.concatenate([
            np.pad(p, ((0, 0), (0, width - p.shape[1]))) for p in behavior
        ]))
        rewards = torch.zeros((len(record_games), 2), dtype=torch.float32)
        for index, (game_id, ply) in enumerate(zip(record_games, record_turns)):
            if ply == turns[game_id] - 1 and winners[game_id]:
                rewards[index, 0] = 1 if winners[game_id] == 1 else -1
                rewards[index, 1] = -rewards[index, 0]
        data["rewards"] = rewards
    seconds = time.monotonic() - started
    return data, {
        "games": count, "positions": len(record_games), "decisions": decision_count,
        "decisive_games": int(np.count_nonzero(winners)), "opponent_moves": opponent_moves,
        "rollout_seconds": seconds, "environment_seconds": environment_seconds,
        "inference_seconds": inference_seconds, "games_per_second": count / seconds,
        "return_mean": float(data["return"].mean()),
        "return_variance": float(data["return"].var(unbiased=False)),
        "old_value_mse": float((data["return"] - data["old_value"]).square().mean()),
        "max_legal_actions": width,
        "data_policy_version": "current frozen learner for entire rollout; no replay",
    }


def _parallel_devices(device, requested):
    if requested is None:
        return None
    devices = list(requested)
    primary = torch.device(device)
    if (primary.type != "cuda" or len(devices) != 2 or len(set(devices)) != 2
            or any(not isinstance(index, int) or index < 0 or index >= torch.cuda.device_count() for index in devices)
            or devices[0] != (primary.index if primary.index is not None else torch.cuda.current_device())):
        raise ValueError("Parallel PPO updates require two distinct visible CUDA devices, primary first")
    return devices


def update(model, optimizer, data, device, recipe=RECIPE, entropy_coefficient=None, parallel_update_devices=None):
    devices = _parallel_devices(device, parallel_update_devices)
    # The raw learner owns Adam and checkpoint state. Parallelize only forward;
    # gather one full-batch loss on the primary GPU and take one optimizer step.
    parallel = torch.nn.DataParallel(model, device_ids=devices, output_device=devices[0]) if devices else None
    parallel_batches = 0
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    if entropy_coefficient is None:
        entropy_coefficient = recipe["entropy_initial"]
    advantage = data["return"] - data["old_value"]
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
    rows, stopped, epochs_completed = [], False, 0
    started = time.monotonic()
    # Keep CPU advantage arithmetic and CPU randperm unchanged for exact resume
    # equivalence. Transfer immutable fresh data once, then only batch indices.
    staged = {
        key: torch.as_tensor(data[key]).to(device=device, dtype=dtype)
        for key, dtype in (
            ("board", torch.float32), ("context", torch.float32),
            ("history", torch.long), ("actions", torch.long),
            ("action", torch.long), ("old_log_probability", torch.float32),
            ("return", torch.float32),
        )
    }
    device_advantage = advantage.to(device)
    metric_names = (
        "loss", "policy_loss", "value_loss", "approximate_kl", "policy_entropy",
        "position_entropy", "card_given_position_entropy", "clip_fraction", "gradient_norm",
    )
    for _ in range(recipe["epochs"]):
        for ids in torch.randperm(len(advantage)).split(recipe["batch_size"]):
            device_ids = ids.to(device)
            board, context, history, actions, mask = tensors(staged, device, device_ids)
            if parallel is not None and len(ids) >= 1024:
                logits, values = parallel(board, context, history, actions, mask)
                parallel_batches += 1
            else:
                logits, values = model(board, context, history, actions, mask)
            distribution = torch.distributions.Categorical(logits=logits.masked_fill(~mask, -torch.inf))
            log_prob = distribution.log_prob(staged["action"][device_ids])
            delta = log_prob - staged["old_log_probability"][device_ids]
            ratio = delta.exp()
            kl = (ratio - 1 - delta).mean()
            if float(kl.detach()) > recipe["target_kl"]:
                stopped = True
                break
            adv = device_advantage[device_ids]
            policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - recipe["ppo_clip"], 1 + recipe["ppo_clip"]) * adv).mean()
            value_loss = (values.flatten() - staged["return"][device_ids]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + recipe["value_coefficient"] * value_loss - entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite exact policy PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                probabilities = distribution.probs
                positions = actions.clamp_min(0) % 100
                positional = torch.zeros((len(ids), 100), device=device).scatter_add_(1, positions, probabilities * mask)
                position_entropy = -(positional * positional.clamp_min(1e-30).log()).sum(1).mean()
            # Detach immediately: diagnostics must not retain computation graphs.
            # Defer the host transfer until all updates finish; safety checks and
            # the per-minibatch KL stopping decision above remain synchronous.
            rows.append(torch.stack((
                loss.detach(), policy_loss.detach(), value_loss.detach(), kl.detach(),
                entropy.detach(), position_entropy.detach(),
                (entropy.detach() - position_entropy).detach(),
                ((ratio - 1).abs() > recipe["ppo_clip"]).float().mean().detach(),
                norm.detach(),
            )))
        epochs_completed += 1
        if stopped:
            break
    model.eval()
    if not rows:
        raise RuntimeError("PPO policy differed from frozen rollout before its first update")
    # The old Python-float/NumPy reduction used float64. Preserve that reduction
    # precision while replacing nine host synchronizations per update with one.
    host_metrics = torch.stack(rows).detach().cpu().numpy().astype(np.float64)
    metrics = {key: float(host_metrics[:, i].mean()) for i, key in enumerate(metric_names)}
    variance = float(data["return"].var(unbiased=False))
    metrics.update(updates=len(rows), epochs_started=epochs_completed, kl_early_stop=stopped,
                   update_seconds=time.monotonic() - started,
                   samples_per_fresh_position=sum(min(recipe["batch_size"], len(advantage) - (i % math.ceil(len(advantage) / recipe["batch_size"])) * recipe["batch_size"]) for i in range(len(rows))) / len(advantage),
                   rollout_value_explained_variance=1 - float((data["return"] - data["old_value"]).var(unbiased=False)) / variance if variance else None)
    if devices:
        metrics.update(parallel_update_gpu_count=len(devices), parallel_forward_batches=parallel_batches)
    return metrics


def checkpoint(model, optimizer, path, state, recipe, rnad_state=None):
    numpy_rng = np.random.get_state()
    save_exact_policy(model, path, {"recipe": recipe, **state}, training_state={
        "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_rng[0], torch.from_numpy(numpy_rng[1].astype(np.int64)), *numpy_rng[2:]),
        "rnad": rnad_state.state_dict() if rnad_state is not None else None,
    })


def restore(path, device, recipe=None, parallel_update_devices=None):
    model, metadata = load_exact_policy(path, device)
    if recipe is not None and metadata["recipe"] != recipe:
        raise ValueError("Refusing to change a continued training recipe silently")
    payload = torch.load(path, map_location="cpu", weights_only=False)["training_state"]
    optimizer = optimizer_for(model, metadata["recipe"], metadata.get("algorithm", "ppo"))
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] and torch.cuda.is_available():
        states = payload["cuda_rng"]
        if parallel_update_devices is not None:
            count = torch.cuda.device_count()
            if len(states) == 1 and count == 2:
                # Explicit single-device -> shared two-device transition. Do not
                # leave GPU1's generator at an uncontrolled process default.
                states = [states[0].clone() for _ in range(count)]
            elif len(states) != count:
                raise ValueError("Saved CUDA RNG count is incompatible with parallel execution")
        torch.cuda.set_rng_state_all(states)
    random.setstate(payload["python_rng"])
    numpy_rng = payload["numpy_rng"]
    np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
    return model, optimizer, {key: value for key, value in metadata.items() if key != "recipe"}


def restore_rnad(model, path, recipe):
    from src.rnad import RNaDConfig, RNaDState

    # Constructors deep-copy weights only, but preserve RNG defensively to make
    # uninterrupted and resumed updates identical even if implementation changes.
    rng = torch.get_rng_state()
    state = RNaDState(model, RNaDConfig(eta=recipe["rnad_eta"], target_avg=recipe["rnad_target_avg"],
                                      reference_interval=recipe["rnad_reference_interval"]))
    payload = torch.load(path, map_location="cpu", weights_only=True)["training_state"].get("rnad")
    if payload is not None:
        state.load_state_dict(payload)
    torch.set_rng_state(rng)
    return state


def update_rnad(model, optimizer, data, state, device, recipe=RECIPE):
    from src.rnad import legal_policy, rnad_loss

    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    games = torch.unique(data["game_index"])
    all_metrics = []
    started = time.monotonic()
    for group in games[torch.randperm(len(games))].split(recipe["rnad_games_per_minibatch"]):
        rows = [torch.where(data["game_index"] == game)[0] for game in group]
        rows = [ids[data["ply"][ids].argsort()] for ids in rows]
        if any(not torch.equal(data["ply"][ids], torch.arange(len(ids))) for ids in rows):
            raise ValueError("RNaD minibatch must contain every move of each complete game")
        length, batch = max(map(len, rows)), len(rows)
        lookup = torch.full((length, batch), -1, dtype=torch.long)
        for column, ids in enumerate(rows):
            lookup[:len(ids), column] = ids
        valid_cpu = lookup >= 0
        ids = lookup[valid_cpu]
        valid = valid_cpu.to(device)
        slots = torch.where(valid.flatten())[0]

        def padded(tensor, length=length, batch=batch, slots=slots):
            return torch.zeros((length * batch, *tensor.shape[1:]), dtype=tensor.dtype, device=device).index_copy(0, slots, tensor).reshape(length, batch, *tensor.shape[1:])

        inputs = tensors(data, device, ids)
        logits, values = model(*inputs)
        legal = padded(inputs[-1])
        logits, values = padded(logits), padded(values.flatten())
        with torch.no_grad():
            _, target_values = state.target(*inputs)
            reference_logits, _ = state.reference(*inputs)
            previous_logits, _ = state.previous_reference(*inputs)
            _, reference_log = legal_policy(reference_logits, inputs[-1])
            _, previous_log = legal_policy(previous_logits, inputs[-1])
        loss, metrics = rnad_loss(
            logits, values, padded(target_values.flatten()),
            padded(data["behavior_policy"][ids].to(device)),
            padded(reference_log), padded(previous_log),
            padded(data["action"][ids].to(device)), legal, valid,
            padded(data["player"][ids].to(device)), padded(data["rewards"][ids].to(device)),
            state.alpha, state.config,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite RNaD loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["rnad_gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        rotated = state.after_update(model)
        all_metrics.append({**{key: float(value) for key, value in metrics.items()}, "loss": float(loss.detach()), "gradient_norm": float(norm),
                            "reference_rotations": int(rotated)})
    model.eval()
    result = {key: float(np.mean([row[key] for row in all_metrics])) for key in all_metrics[0]}
    result.update(updates=len(all_metrics), update_seconds=time.monotonic() - started,
                  reference_steps=state.learner_steps, reference_alpha=state.alpha,
                  reference_rotations=sum(row["reference_rotations"] for row in all_metrics),
                  samples_per_fresh_position=1.0)
    return result


def evaluate(model, reference, device, games, seed):
    if games < 2 or games % 2:
        raise ValueError("Evaluation needs complete paired deals")
    model.eval()
    reference.eval()
    before = [{k: v.detach().cpu().clone() for k, v in actor.state_dict().items()} for actor in (model, reference)]
    started = time.monotonic()
    env = native_environment(games, seed, paired=True)
    active = env.active_ids()
    seconds, moves = [0.0, 0.0], [0, 0]
    while active:
        players = env.info()["players"]
        for candidate in (True, False):
            ids = [i for i in active if (players[i] == 1 + i % 2) == candidate]
            if not ids:
                continue
            actor = model if candidate else reference
            tick = time.monotonic()
            obs = observations(env, ids, isinstance(actor, ExactPolicy))
            with torch.inference_mode():
                logits, _, actions, _ = actor_forward(actor, obs, device)
                chosen = logits.argmax(1)
                exact = actions.gather(1, chosen[:, None]).flatten().cpu().tolist()
            seconds[0 if candidate else 1] += time.monotonic() - tick
            moves[0 if candidate else 1] += len(ids)
            env.step(ids, exact)
        active = env.active_ids()
    winners = env.info()["winners"]
    for actor, snapshot in zip((model, reference), before):
        if actor.training or any(not torch.equal(value.detach().cpu(), snapshot[key]) for key, value in actor.state_dict().items()):
            raise RuntimeError("Evaluation mutated model parameters or buffers")
    outcomes = [0.5 if winner == 0 else float(winner == 1 + index % 2) for index, winner in enumerate(winners)]
    return summarize(outcomes) | {
        "evaluation_seconds": time.monotonic() - started, "parameters_and_buffers_unchanged": True,
        "batched_inference_seconds": seconds, "decision_counts": moves,
        "inference_cost_scope": "batched observation and inference; not single-move browser latency",
    }


def initialize(initial, contender, directory, recipe, device):
    directory.mkdir(exist_ok=True)
    if (directory / "latest.pt").exists():
        return
    torch.manual_seed(recipe["seed"])
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"] % (2**32))
    variant, algorithm = CONTENDERS[contender]
    base = initial.backbone if isinstance(initial, ExactPolicy) else initial
    options = {"value_mode": "unbounded"} if algorithm == "rnad" else {}
    model = ExactPolicy.from_history_policy(base, variant=variant, **options).to(device)
    if isinstance(initial, ExactPolicy):
        missing = model.load_state_dict(initial.state_dict(), strict=False)
        if missing.unexpected_keys:
            raise ValueError(f"Shared warm start has incompatible parameters: {missing.unexpected_keys}")
    optimizer = optimizer_for(model, recipe, algorithm)
    state = {"variant": variant, "algorithm": algorithm, "contender": contender, "iterations": 0, "games": 0, "positions": 0,
             "updates": 0, "training_seconds": 0.0, "blocks": 0}
    checkpoint(model, optimizer, directory / "latest.pt", state, recipe)
    save_exact_policy(model, directory / "historical_0.pt", {"initial": True})


def train_block(directory, reference, device, recipe, duration, deadline, parallel_update_devices=None):
    # Load other models before restoring learner RNG state.
    historical = [load_exact_policy(path, device)[0] for path in sorted(directory.glob("historical_*.pt"))]
    devices = _parallel_devices(device, parallel_update_devices)
    model, optimizer, state = restore(directory / "latest.pt", device, recipe, parallel_update_devices=devices)
    algorithm = state.get("algorithm", "ppo")
    if devices and (algorithm != "ppo" or model.variant != "cnn"):
        raise ValueError("Shared update execution is validated only for CNN PPO")
    rnad_state = restore_rnad(model, directory / "latest.pt", recipe) if algorithm == "rnad" else None
    start = time.monotonic()
    previous, last_saved = state["training_seconds"], start
    execution_segment = None
    if devices or "execution_accounting" in state:
        execution_segment = {
            "mode": "update_data_parallel" if devices else "single_gpu",
            "visible_device_indices": devices or [torch.device(device).index or 0],
            "allocated_gpu_count": len(devices) if devices else 1,
            "training_seconds_start": previous, "wall_seconds": 0.0, "allocated_gpu_seconds": 0.0,
            "scope": "Allocated GPU time for this whole block, including primary-only rollouts and checkpoints; not measured GPU busy time",
            "parallel_minimum_batch": 1024 if devices else None,
            "global_batch_size": recipe["batch_size"],
            "rng_transition": "A saved single CUDA state is copied to both visible devices; same-mode states restored independently" if devices else "unchanged",
        }
        state.setdefault("execution_accounting", {"segments": []})["segments"].append(execution_segment)
    def account_execution():
        if execution_segment is not None:
            elapsed = time.monotonic() - start
            execution_segment["wall_seconds"] = elapsed
            execution_segment["allocated_gpu_seconds"] = elapsed * execution_segment["allocated_gpu_count"]
    while not STOP_REQUESTED and time.monotonic() - start < duration and time.time() < deadline - 15:
        tick = time.monotonic()
        scale, entropy = schedule(state["training_seconds"], recipe)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * scale
        seed = recipe["seed"] + state["iterations"] * max(10000, recipe["games_per_iteration"])
        if seed + recipe["games_per_iteration"] >= recipe["monitor_seed"]:
            raise ValueError("Training seeds reached held-out monitoring reservation")
        opponents = (reference, random.choice(historical)) if algorithm == "ppo" else ()
        data, metrics = rollout(model, device, recipe["games_per_iteration"], seed,
                                opponents=opponents, full_trajectory=algorithm == "rnad")
        metrics.update(update_rnad(model, optimizer, data, rnad_state, device, recipe)
                       if algorithm == "rnad" else update(model, optimizer, data, device, recipe, entropy, parallel_update_devices=devices))
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        state["iterations"] += 1
        state["games"] += metrics["games"]
        state["positions"] += metrics["positions"]
        state["updates"] += metrics["updates"]
        state["training_seconds"] = previous + time.monotonic() - start
        account_execution()
        metrics.update(state, iteration_seconds=time.monotonic() - tick, selfplay_seed=seed,
                       trunk_learning_rate=recipe["trunk_lr"] * scale,
                       new_learning_rate=recipe["new_lr"] * scale,
                       entropy_coefficient=entropy if algorithm == "ppo" else 0.0,
                       regularization_eta=recipe["rnad_eta"] if algorithm == "rnad" else 0.0,
                       opponent_distribution=recipe["rnad_opponents"] if algorithm == "rnad" else recipe["opponents"])
        with (directory / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        if time.monotonic() - last_saved >= recipe["checkpoint_seconds"]:
            checkpoint(model, optimizer, directory / "latest.pt", state, recipe, rnad_state)
            atomic_json(directory / "progress.json", state)
            last_saved = time.monotonic()
        del data
    state["blocks"] += 1
    state["training_seconds"] = previous + time.monotonic() - start
    account_execution()
    checkpoint(model, optimizer, directory / "latest.pt", state, recipe, rnad_state)
    atomic_json(directory / "progress.json", state)
    save_exact_policy(model, directory / f"historical_{state['blocks'] % recipe['historical_pool_size']}.pt",
                      {"training_seconds": state["training_seconds"], "iterations": state["iterations"]})
    return model.eval(), state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-seconds", type=float, default=900)
    parser.add_argument("--games-per-iteration", type=int, default=512)
    parser.add_argument("--evaluation-games", type=int, default=256)
    parser.add_argument("--contenders", nargs="+", choices=tuple(CONTENDERS), default=list(CONTENDERS))
    args = parser.parse_args()
    if args.games_per_iteration < 4 or args.evaluation_games < 2 or args.evaluation_games % 2 or args.block_seconds <= 0:
        parser.error("Invalid game count or block duration")
    torch.set_num_threads(1)
    deadline = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00")).timestamp()
    if time.time() >= deadline:
        raise SystemExit("Training deadline has passed; no new training started")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def request_stop(*_):
        global STOP_REQUESTED
        STOP_REQUESTED = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    recipe = dict(RECIPE, contenders=args.contenders, games_per_iteration=args.games_per_iteration,
                  common_pretraining_checkpoint_sha256=digest(args.init_from),
                  reference_sha256=digest(args.reference))
    existing = output / "recipe.json"
    if existing.exists() and json.loads(existing.read_text()) != recipe:
        raise ValueError("Existing run recipe differs; explicit migration is required")
    atomic_json(existing, recipe)
    initial_payload = torch.load(args.init_from, map_location="cpu", weights_only=True)
    initial, _ = (load_exact_policy(args.init_from, args.device) if initial_payload.get("kind") == "exact_sequence_policy"
                  else load_policy(args.init_from, args.device))
    reference, _ = load_policy(args.reference, args.device)
    reference.requires_grad_(False)
    for contender in args.contenders:
        initialize(initial, contender, output / contender, recipe, args.device)
    del initial
    while not STOP_REQUESTED and time.time() < deadline - 15:
        states = {name: torch.load(output / name / "latest.pt", map_location="cpu", weights_only=True)["metadata"] for name in args.contenders}
        allowed = args.contenders
        selection = output / "selection.json"
        if selection.exists():
            selected = json.loads(selection.read_text())["contenders"]
            if not selected or not set(selected).issubset(args.contenders):
                raise ValueError("Invalid contender selection")
            if len(selected) < len(args.contenders) and any(state["training_seconds"] < 3600 for state in states.values()):
                raise ValueError("All serious contenders require one matched training hour before narrowing")
            allowed = selected
        variant = min(allowed, key=lambda name: states[name]["training_seconds"])
        directory = output / variant
        atomic_json(output / "state.json", {"status": "training", "variant": variant, "stop_at": args.stop_at,
                                            "updated_utc": datetime.now(timezone.utc).isoformat()})
        model, state = train_block(directory, reference, args.device, recipe, args.block_seconds, deadline)
        if not STOP_REQUESTED and time.time() < deadline:
            seed = recipe["monitor_seed"] + state["blocks"] * 1000000
            monitor = evaluate(model, reference, args.device, args.evaluation_games, seed)
            report = dict(state, result=monitor, seed=seed, monitoring_only=True,
                          production_promoted=False, checkpoint_sha256=digest(directory / "latest.pt"))
            atomic_json(directory / f"evaluation_{state['blocks']:03d}.json", report)
            best_file = directory / "best_monitor.json"
            best = json.loads(best_file.read_text()) if best_file.exists() else None
            if best is None or monitor["score"] > best["score"]:
                atomic_copy(directory / "latest.pt", directory / "best_monitor.pt")
                atomic_json(best_file, {"score": monitor["score"], "block": state["blocks"], "monitoring_only": True})
            print(json.dumps({"variant": variant, "training_seconds": state["training_seconds"],
                              "monitor_score": monitor["score"], "monitor_ci": monitor["paired_score_95ci"]}), flush=True)
        del model
    atomic_json(output / "state.json", {"status": "stopped", "updated_utc": datetime.now(timezone.utc).isoformat(),
                                        "reason": "signal" if STOP_REQUESTED else "training_deadline"})


if __name__ == "__main__":
    main()
