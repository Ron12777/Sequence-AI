"""Paired complete standard Sequence games for a public-belief agent.

The candidate must implement reset(seed, player) and act(game, player, events),
returning an exact legal Move. No tactical override or truncated-game scoring is
used. The game object contains hidden state for the rules engine; respecting
information boundaries is an independently tested responsibility of the agent.
"""

import argparse
import fcntl
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.architecture_experiments import direct_moves
from scripts.arena import choose_moves, summarize
from scripts.overnight import atomic_json
from src.fast_game import FastSequenceGame
from src.game import Move
from src.history_policy import apply_public_move, load_policy
from src.model import load_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def move_key(move):
    return (move.card.to_int(), move.row, move.col, bool(move.is_removal))


def game_fingerprint(game):
    """Catch mutation of the real game by a search implementation."""
    return (
        game.c_state.get_tensor(),
        tuple(game.c_state.get_info()),
        tuple(tuple(card.to_int() for card in game.hands[p]) for p in (1, 2)),
        tuple(tuple(card.to_int() for card in game.discard_piles[p]) for p in (1, 2)),
        tuple(card.to_int() for card in game.deck.cards),
        repr(game.c_state.get_sequences()),
    )


def model_snapshot(model):
    model.eval()
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def assert_unchanged(models, snapshots):
    for model, snapshot in zip(models, snapshots):
        if model.training:
            raise RuntimeError("Evaluation changed a frozen model to training mode")
        state = model.state_dict()
        if state.keys() != snapshot.keys() or not all(
            torch.equal(value.detach().cpu(), snapshot[key])
            for key, value in state.items()
        ):
            raise RuntimeError("Evaluation mutated model parameters or buffers")


def latency_summary(seconds):
    values = np.asarray(seconds, dtype=float)
    return {
        "moves": len(seconds),
        "total_seconds": float(values.sum()),
        "mean_seconds": float(values.mean()) if len(values) else None,
        "median_seconds": float(np.median(values)) if len(values) else None,
        "p95_seconds": float(np.percentile(values, 95)) if len(values) else None,
    }


def evaluate_full(
    agent,
    opponent_model,
    *,
    games,
    seed,
    opponent="history",
    simulations=0,
    device="cpu",
    frozen_models=(),
    completed_games=(),
    on_pair=None,
):
    """Evaluate complete games, resuming only at complete paired-deal boundaries.

    Every record includes the deal seed, candidate seat, all exact moves and
    unbatched decision latencies. References and explicitly supplied candidate
    models are frozen and checked after every pair, including all buffers.
    """
    if games <= 0 or games % 2 or len(completed_games) % 2:
        raise ValueError("Positive even game count and complete paired resume required")
    if len(completed_games) > games:
        raise ValueError("Resume contains too many games")
    if opponent not in {"history", "champion"}:
        raise ValueError("Opponent must be history or champion")
    if simulations < 0 or (opponent == "history" and simulations):
        raise ValueError("History reference is direct; simulations must be nonnegative")
    device = torch.device(device)
    models = [opponent_model, *frozen_models]
    snapshots = [model_snapshot(model) for model in models]
    records = list(completed_games)
    for i, record in enumerate(records):
        if (
            record["index"] != i
            or record["deal_seed"] != seed + i // 2
            or record["candidate_player"] != 1 + i % 2
        ):
            raise ValueError(
                "Resume records do not match requested paired seed schedule"
            )
    for index in range(len(records), games):
        deal_seed = seed + index // 2
        random.seed(deal_seed)
        game = FastSequenceGame()
        player = 1 + index % 2
        agent_seed = seed + 100_000_000 + index
        agent.reset(seed=agent_seed, player=player)
        initial_digest = hashlib.sha256(
            repr(game_fingerprint(game)).encode()
        ).hexdigest()
        events = []
        moves = []
        started = time.monotonic()
        termination = "terminal"
        while not game.game_over:
            legal = game.get_legal_moves()
            if not legal:
                termination = "no_legal_moves"
                break
            if len(moves) >= 200:
                raise RuntimeError(
                    "Standard finite-deck game exceeded 200 moves; no truncated score emitted"
                )
            actor = game.current_player
            before = game_fingerprint(game)
            tick = time.perf_counter()
            with torch.inference_mode():
                if actor == player:
                    move = agent.act(game, actor, tuple(events))
                elif opponent == "history":
                    move = direct_moves(opponent_model, device, [game], [events], [0])[
                        0
                    ]
                else:
                    move = choose_moves(
                        opponent_model,
                        device,
                        [game],
                        [0],
                        simulations,
                        seed + index * 1000,
                        [len(moves)],
                        "ensemble",
                    )[0]
            elapsed = time.perf_counter() - tick
            if game_fingerprint(game) != before:
                raise RuntimeError(
                    "Agent mutated the live game during action selection"
                )
            if not isinstance(move, Move) or move_key(move) not in {
                move_key(m) for m in legal
            }:
                raise RuntimeError(
                    f"Agent returned an illegal or incomplete action: {move!r}"
                )
            event_offset = len(events)
            apply_public_move(game, events, move)
            moves.append(
                {
                    "ply": len(moves),
                    "player": actor,
                    "candidate": actor == player,
                    "card": move.card.to_int(),
                    "row": move.row,
                    "col": move.col,
                    "is_removal": bool(move.is_removal),
                    "seconds": elapsed,
                    "public_events": events[event_offset:],
                }
            )
        score = 0.5 if game.winner is None else float(game.winner == player)
        records.append(
            {
                "index": index,
                "deal_seed": deal_seed,
                "candidate_player": player,
                "agent_seed": agent_seed,
                "initial_state_sha256": initial_digest,
                "winner": game.winner,
                "score": score,
                "termination": termination,
                "game_over": bool(game.game_over),
                "plies": len(moves),
                "elapsed_seconds": time.monotonic() - started,
                "moves": moves,
            }
        )
        if len(records) % 2 == 0:
            if (
                records[-1]["initial_state_sha256"]
                != records[-2]["initial_state_sha256"]
            ):
                raise RuntimeError("Paired games did not use identical initial deals")
            assert_unchanged(models, snapshots)
            if on_pair is not None:
                on_pair(records)
    assert_unchanged(models, snapshots)
    return {
        **summarize([row["score"] for row in records]),
        "game_records": records,
        "candidate_latency": latency_summary(
            [m["seconds"] for row in records for m in row["moves"] if m["candidate"]]
        ),
        "opponent_latency": latency_summary(
            [
                m["seconds"]
                for row in records
                for m in row["moves"]
                if not m["candidate"]
            ]
        ),
        "parameters_and_buffers_unchanged": True,
        "frozen_model_count": len(models),
        "latency_scope": "unbatched action selection; includes candidate search and reference inference/search",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent-checkpoint", required=True)
    p.add_argument("--opponent", choices=("history", "champion"), default="history")
    p.add_argument("--simulations", type=int, default=0)
    p.add_argument("--particles", type=int, default=8)
    p.add_argument("--iterations", type=int, default=16)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(a.threads)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = a.output.with_suffix(".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(a).items()
        if key != "output"
    }
    plan["checkpoint_sha256"] = digest(a.checkpoint)
    plan["opponent_checkpoint_sha256"] = digest(a.opponent_checkpoint)
    report = {"status": "running", "plan": plan, "game_records": []}
    if a.output.exists():
        report = json.loads(a.output.read_text())
        if report["plan"] != plan:
            raise ValueError(
                "Cannot resume with different settings or checkpoint contents"
            )
        if report["status"] == "complete":
            return
    from src.rebel_full_agent import ReBeLAgent

    agent = ReBeLAgent(
        a.checkpoint,
        device=a.device,
        particles=a.particles,
        iterations=a.iterations,
        depth=a.depth,
        seed=a.seed,
    )
    candidate_models = getattr(agent, "evaluation_models", None)
    if candidate_models is None:
        raise ValueError(
            "Agent must expose evaluation_models for checkpoint immutability checks"
        )
    if callable(candidate_models):
        candidate_models = candidate_models()
    candidate_models = tuple(candidate_models)
    if not candidate_models:
        raise ValueError(
            "Checkpoint-backed agent must expose at least one evaluation model"
        )
    if a.opponent == "history":
        reference, _ = load_policy(a.opponent_checkpoint, a.device)
    else:
        reference, _, _ = load_model(
            a.opponent_checkpoint, device=a.device, compile_model=False
        )
    before_seconds = report.get("elapsed_seconds", 0.0)
    started = time.monotonic()

    def save_pair(records):
        report.update(
            game_records=records,
            summary=summarize([row["score"] for row in records]),
            elapsed_seconds=before_seconds + time.monotonic() - started,
        )
        atomic_json(a.output, report)
        print(
            json.dumps({"games": len(records), "score": report["summary"]["score"]}),
            flush=True,
        )

    result = evaluate_full(
        agent,
        reference,
        games=a.games,
        seed=a.seed,
        opponent=a.opponent,
        simulations=a.simulations,
        device=a.device,
        frozen_models=candidate_models,
        completed_games=report["game_records"],
        on_pair=save_pair,
    )
    if (
        digest(a.checkpoint) != plan["checkpoint_sha256"]
        or digest(a.opponent_checkpoint) != plan["opponent_checkpoint_sha256"]
    ):
        raise RuntimeError("Checkpoint file changed during evaluation")
    report.update(
        result,
        status="complete",
        elapsed_seconds=before_seconds + time.monotonic() - started,
    )
    atomic_json(a.output, report)


if __name__ == "__main__":
    main()
