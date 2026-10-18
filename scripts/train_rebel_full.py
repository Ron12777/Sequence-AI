"""Sustained full Sequence public-belief value learning with complete games.

Workers generate CFR root targets and Monte Carlo targets for chosen pre-draw
leaves. The latter anchor the learned continuation to real card draws and full
game outcomes. Replay pools and losses are separated so bootstrapping cannot
hide failure on actual outcomes. This approximate recipe is not exact ReBeL.
"""

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.architecture_experiments import direct_moves
from scripts.overnight import atomic_json
from src.fast_game import FastSequenceGame
from src.history_policy import apply_public_move, load_policy
from src.rebel_full_agent import ReBeLAgent
from src.rebel_full_value import load_value, save_value

_worker_agent = None
_worker_reference = None


def generate_game(config, index):
    global _worker_agent, _worker_reference
    torch.set_num_threads(1)
    seed = config["seed"] + index
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if _worker_agent is None:
        _worker_agent = ReBeLAgent(
            config["weights"],
            particles=config["particles"],
            iterations=config["iterations"],
            depth=config["depth"],
        )
        _worker_reference, _ = load_policy(config["reference"], "cpu")
    payload = torch.load(config["weights"], map_location="cpu", weights_only=False)
    _worker_agent.model.residual.load_state_dict(payload["residual"])
    _worker_agent.model.residual.eval()
    version = payload["metadata"].get("games", 0)
    _worker_agent.reset(seed=seed)
    game, events = FastSequenceGame(), []
    search_rows, monte_carlo = [], []
    start = time.monotonic()
    # Half self-play, half frozen trained opponent, balanced learner seat.
    reference_seat = 0 if index % 4 < 2 else 1 + index % 2
    decisions, fallback, unsupported, search_seconds = 0, 0, 0, 0.0
    while not game.game_over and len(events) < 120 and game.get_legal_moves():
        player = game.current_player
        if player == reference_seat:
            move = direct_moves(_worker_reference, "cpu", [game], [events], [0])[0]
        else:
            move = _worker_agent.act(game, player, tuple(events))
            result = _worker_agent.last_search
            adapter = result["adapter"]
            x, base = _worker_agent.model.examples(
                adapter, adapter.root, result["root_belief"]
            )
            target = np.concatenate(result["mean_root_conditional_values"]).astype(
                np.float32
            )
            valid = np.concatenate(result["root_belief"].compatible_mass) > 0
            search_rows.extend(
                (f, b, y) for f, b, y in zip(x[valid], base[valid], target[valid])
            )
            action, belief, actual_hand = result["last_query"]
            xq, bq = _worker_agent.model.query_example(
                adapter, action, belief, actual_hand
            )
            monte_carlo.append((xq, bq, player))
            metrics = result["metrics"]
            decisions += 1
            fallback += metrics["public_legality_fallbacks"]
            unsupported += metrics["unsupported_action_fallbacks"]
            search_seconds += metrics["wall_seconds"]
        apply_public_move(game, events, move)
    mc_rows = [
        (x, b, np.float32(game.get_result(player))) for x, b, player in monte_carlo
    ]
    return (
        search_rows,
        mc_rows,
        {
            "index": index,
            "seed": seed,
            "version": version,
            "moves": sum(e[3] != 2 for e in events),
            "decisions": decisions,
            "winner": game.winner,
            "reference_seat": reference_seat,
            "seconds": time.monotonic() - start,
            "search_seconds": search_seconds,
            "public_legality_fallbacks": fallback,
            "unsupported_action_fallbacks": unsupported,
            "search_examples": len(search_rows),
            "monte_carlo_examples": len(mc_rows),
            "termination": "win" if game.game_over else "no_legal_move_or_exhaustion",
        },
    )


def update(value, optimizer, root_replay, mc_replay, rng, steps=8, batch_size=128):
    losses = []
    value.residual.train()
    for _ in range(steps):
        loss_terms = []
        for replay in (root_replay, mc_replay):
            ids = rng.integers(0, len(replay), batch_size)
            rows = [replay[int(i)] for i in ids]
            x = torch.tensor(np.stack([r[0] for r in rows]), device=value.device)
            base = torch.tensor(
                np.array([r[1] for r in rows], np.float32), device=value.device
            )
            target = torch.tensor(
                np.array([r[2] for r in rows], np.float32), device=value.device
            )
            pred = value.residual(x, base)
            loss_terms.append((pred - target).square().mean())
        loss = 0.5 * (loss_terms[0] + loss_terms[1])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite full-game belief value loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(value.residual.parameters(), 1.0)
        optimizer.step()
        losses.append([float(t.detach()) for t in loss_terms] + [float(norm)])
    value.residual.eval()
    mean = np.mean(losses, axis=0)
    return {
        "root_cfr_mse": float(mean[0]),
        "pre_draw_return_mse": float(mean[1]),
        "gradient_norm": float(mean[2]),
        "updates": steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="artifacts/checkpoints/board-four-hour.pt")
    p.add_argument("--reference", default="artifacts/checkpoints/gru-four-hour.pt")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seconds", type=float, default=3600)
    p.add_argument("--games", type=int, default=100000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--particles", type=int, default=8)
    p.add_argument("--iterations", type=int, default=16)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--seed", type=int, default=1717000001)
    p.add_argument("--device", default="cpu")
    p.add_argument("--updates-per-game", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "training.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    weights, latest = args.output / "weights.pt", args.output / "latest.pt"
    teacher = str(Path(args.teacher).resolve())
    recipe = {
        "teacher": teacher,
        "reference": str(Path(args.reference).resolve()),
        "particles": args.particles,
        "iterations": args.iterations,
        "depth": args.depth,
        "seed": args.seed,
        "learning_rate": 1e-4,
        "updates_per_game": args.updates_per_game,
        "root_replay": 20000,
        "mc_replay": 10000,
        "batch_per_pool": 128,
        "objective": "equal mean CFR root MSE and chosen pre-draw full-game return MSE",
        "opponents": "50% self-play, 50% frozen GRU, balanced seats",
    }
    value = load_value(latest if latest.exists() else teacher, args.device)
    optimizer = torch.optim.Adam(value.residual.parameters(), lr=1e-4)
    rng = np.random.default_rng(args.seed)
    root_replay, mc_replay, games, next_index, elapsed = [], [], 0, 0, 0.0
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        state = saved["training_state"]
        if state["recipe"] != recipe:
            raise ValueError("Refusing to silently change a resumed training recipe")
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        root_replay, mc_replay = state["root_replay"], state["mc_replay"]
        games, next_index, elapsed = (
            state["games"],
            state["next_index"],
            state["elapsed"],
        )
    start = time.monotonic()

    def save(continuation=False):
        metadata = {
            "games": games,
            "elapsed_seconds": elapsed + time.monotonic() - start,
            "recipe": recipe,
            "scope": "full standard Sequence, approximate particle public-belief search",
        }
        save_value(value, weights, teacher, metadata=metadata)
        if continuation:
            state = {
                "recipe": recipe,
                "optimizer": optimizer.state_dict(),
                "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "root_replay": root_replay,
                "mc_replay": mc_replay,
                "games": games,
                "next_index": next_index,
                "elapsed": metadata["elapsed_seconds"],
            }
            save_value(value, latest, teacher, training_state=state, metadata=metadata)
        atomic_json(
            args.output / "progress.json",
            metadata | {"pid": os.getpid(), "status": "training"},
        )

    save(continuation=True)
    config = recipe | {"weights": str(weights.resolve())}
    pending = set()
    atomic_json(args.output / "recipe.json", recipe)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        while time.monotonic() - start < args.seconds and games < args.games or pending:
            while (
                len(pending) < args.workers
                and time.monotonic() - start < args.seconds
                and games + len(pending) < args.games
            ):
                pending.add(pool.submit(generate_game, config, next_index))
                next_index += 1
            if not pending:
                break
            done, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for future in done:
                roots, mc, metrics = future.result()
                root_replay.extend(roots)
                mc_replay.extend(mc)
                del root_replay[: -recipe["root_replay"]]
                del mc_replay[: -recipe["mc_replay"]]
                games += 1
                fit = (
                    update(
                        value,
                        optimizer,
                        root_replay,
                        mc_replay,
                        rng,
                        steps=args.updates_per_game,
                    )
                    if roots and mc
                    else {}
                )
                metrics.update(fit)
                metrics.update(
                    {
                        "games_completed": games,
                        "policy_lag_games": games - metrics["version"],
                        "total_wall_seconds": elapsed + time.monotonic() - start,
                    }
                )
                with (args.output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                save(continuation=games % 8 == 0)
                print(json.dumps(metrics), flush=True)
    save(continuation=True)
    atomic_json(
        args.output / "completion.json",
        {"games": games, "elapsed_seconds": elapsed + time.monotonic() - start},
    )


if __name__ == "__main__":
    main()
