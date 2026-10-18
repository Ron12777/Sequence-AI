"""Independent paired full-game confirmation for exact-card policies.

Candidates use canonical hand channels; legacy references retain original hand
order and exact historical tie-breaking. Rust owns decks, full rules and public
history. Search receives only public state, own hand and unseen-card counts.
Greedy play remains the default. Optional candidate-only categorical play uses
the learned T=1 distribution with stateless per-game, per-decision draws.
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
import sequence_native
import torch

from scripts.arena import summarize
from scripts.evaluate import predict
from scripts.overnight import atomic_json
from src.cards import Deck
from src.exact_policy import load_exact_policy
from src.history_policy import load_policy
from src.model import load_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(model):
    model.eval()
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def verify_frozen(model, before):
    current = model.state_dict()
    if (
        model.training
        or current.keys() != before.keys()
        or not all(
            torch.equal(value.detach().cpu(), before[key])
            for key, value in current.items()
        )
    ):
        raise RuntimeError("Evaluation mutated model parameters, buffers or mode")


def paired_env(count, seed, threads=1):
    decks = []
    for index in range(count):
        random.seed(seed + index // 2)
        decks.append([card.to_int() for card in Deck().cards])
    return sequence_native.BatchEnv.from_decks(decks, threads=threads)


def observation_tensors(env, ids, device, canonical):
    data = env.observe(ids, canonical=canonical)
    n, width = len(ids), data["width"]

    def convert(key, dtype, shape, target):
        return torch.from_numpy(
            np.frombuffer(data[key], dtype).reshape(shape).copy()
        ).to(device=device, dtype=target)

    return (
        convert("board", np.uint8, (n, 11, 10, 10), torch.float32),
        convert("context", "<f4", (n, 157), torch.float32),
        convert("history", np.uint8, (n, 128, 4), torch.long),
        convert("actions", "<i2", (n, width), torch.long),
    )


def validate_action_mode(mode, action_seed):
    if mode not in {"greedy", "sample"}:
        raise ValueError("candidate_action_mode must be greedy or sample")
    if mode == "sample" and (not isinstance(action_seed, int) or isinstance(action_seed, bool)):
        raise ValueError("Sample mode requires an integer action_seed")


def action_uniform(action_seed, global_game_index, candidate_decision_count):
    """A stable [0,1) draw independent of global RNG, batching and resumption."""
    data = f"{action_seed}:{global_game_index}:{candidate_decision_count}".encode("ascii")
    bits = int.from_bytes(hashlib.blake2b(data, digest_size=8, person=b"seq-exact-eval").digest(), "little")
    return (bits >> 11) * (2.0 ** -53)


def categorical_actions(logits, actions, uniforms):
    """Sample T=1 probabilities over legal exact IDs only, using float64 CDFs."""
    if logits.shape != actions.shape or len(uniforms) != len(actions):
        raise ValueError("Inconsistent categorical action shapes")
    selected = []
    for row, ids, draw in zip(logits, actions, uniforms):
        legal = ids >= 0
        legal_logits, legal_ids = np.asarray(row[legal], dtype=np.float64), ids[legal]
        if not len(legal_ids) or not np.isfinite(legal_logits).all() or not 0 <= draw < 1:
            raise ValueError("Categorical sampling requires finite legal logits and a [0,1) draw")
        weights = np.exp(legal_logits - legal_logits.max())
        cumulative = np.cumsum(weights)
        index = int(np.searchsorted(cumulative, draw * cumulative[-1], side="right"))
        selected.append(int(legal_ids[min(index, len(legal_ids) - 1)]))
    return selected


def exact_moves(
    model, env, ids, device, *, action_mode="greedy", action_seed=None,
    game_offset=0, decision_counts=None,
):
    validate_action_mode(action_mode, action_seed)
    board, context, history, actions = observation_tensors(env, ids, device, True)
    mask = actions >= 0
    with torch.inference_mode():
        logits, values = model(board, context, history, actions, mask)
    if not torch.isfinite(logits[mask]).all() or not torch.isfinite(values).all():
        raise RuntimeError("Exact model produced nonfinite outputs")
    if action_mode == "sample":
        if decision_counts is None:
            raise ValueError("Sample mode requires per-game candidate decision counts")
        uniforms = [action_uniform(action_seed, game_offset + index, decision_counts[index]) for index in ids]
        return categorical_actions(logits.detach().cpu().numpy(), actions.cpu().numpy(), uniforms)
    chosen = logits.masked_fill(~mask, -torch.inf).argmax(1)
    return actions.gather(1, chosen[:, None]).flatten().cpu().tolist()


def legacy_moves(model, env, ids, device, kind):
    board, context, history, _ = observation_tensors(env, ids, device, False)
    with torch.inference_mode():
        policy, values = (
            model(board, context, history) if kind == "history" else model(board)
        )
    if not torch.isfinite(policy).all() or not torch.isfinite(values).all():
        raise RuntimeError("Legacy model produced nonfinite outputs")
    policies = policy.float().cpu().numpy()
    states = env.search_states(ids)
    # max keeps original legal ordering when both score and normal-card status
    # tie, precisely matching direct_moves/arena.choose_moves.
    return [
        max(
            state["legal"],
            key=lambda action: (float(pi[action % 100]), (action // 100) % 13 != 10),
        )
        for state, pi in zip(states, policies)
    ]


def search_moves(model, env, ids, device, simulations, seed, counts, game_offset=0):
    searches = []
    for index, state in zip(ids, env.search_states(ids)):
        search = sequence_native.EnsembleMCTS(
            seed + (game_offset + index) * 1000 + counts[index]
        )
        search.reset(
            state["board"],
            state["hand"],
            state["player"],
            state["sequences"],
            state["unseen"],
            state["opponent_hand_size"],
        )
        searches.append(search)
    for _ in range(simulations):
        pending, tensors = [], []
        for search in searches:
            token, data, value, terminal = search.select_leaf()
            if terminal:
                search.backpropagate(token, b"", value)
            else:
                pending.append((search, token))
                tensors.append(np.frombuffer(data, np.float32).reshape(11, 10, 10))
        if pending:
            policies, values = predict(model, device, tensors)
            for (search, token), policy, value in zip(pending, policies, values):
                search.backpropagate(token, policy.tobytes(), float(value))
    actions = []
    for search in searches:
        action = search.get_action(0.0)
        if action is None:
            raise RuntimeError("Search returned no move for a legal live position")
        card, row, col, *_ = action
        actions.append(card * 100 + row * 10 + col)
    return actions


def evaluate_exact(
    candidate,
    opponent,
    *,
    games,
    seed,
    kind="history",
    simulations=0,
    device="cpu",
    env_threads=1,
    game_offset=0,
    search_seed=None,
    candidate_action_mode="greedy",
    action_seed=None,
):
    validate_action_mode(candidate_action_mode, action_seed)
    if games <= 0 or games % 2 or game_offset % 2:
        raise ValueError(
            "Evaluation requires positive paired games and even game offset"
        )
    if (
        kind not in {"exact", "history", "champion"}
        or simulations < 0
        or (kind != "champion" and simulations)
    ):
        raise ValueError("Invalid reference kind or simulation budget")
    device = torch.device(device)
    snapshots = [freeze(model) for model in (candidate, opponent)]
    start = time.monotonic()
    env = paired_env(games, seed, env_threads)
    counts = [0] * games
    candidate_counts = [0] * games
    traces = [[] for _ in range(games)]
    seconds = [0.0, 0.0]
    move_counts = [0, 0]
    search_seed = seed if search_seed is None else search_seed
    for _ in range(200):
        if not env.active_ids():
            break
        for side in (0, 1):
            players = env.info()["players"]
            ids = [
                i for i in env.active_ids() if (players[i] == 1 + i % 2) == (side == 0)
            ]
            if not ids:
                continue
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            if side == 0:
                actions = exact_moves(
                    candidate, env, ids, device, action_mode=candidate_action_mode,
                    action_seed=action_seed, game_offset=game_offset,
                    decision_counts=candidate_counts,
                )
            elif kind == "exact":
                actions = exact_moves(opponent, env, ids, device)
            elif kind == "champion" and simulations:
                actions = search_moves(
                    opponent,
                    env,
                    ids,
                    device,
                    simulations,
                    search_seed,
                    counts,
                    game_offset,
                )
            else:
                actions = legacy_moves(opponent, env, ids, device, kind)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds[side] += time.perf_counter() - tick
            move_counts[side] += len(ids)
            env.step(ids, actions)
            for index, action in zip(ids, actions):
                traces[index].append(int(action))
                counts[index] += 1
                if side == 0:
                    candidate_counts[index] += 1
    if env.active_ids():
        raise RuntimeError(
            "Finite-deck games exceeded 200 rounds; no truncated scores emitted"
        )
    for model, snapshot in zip((candidate, opponent), snapshots):
        verify_frozen(model, snapshot)
    winners = env.info()["winners"]
    outcomes = [
        0.5 if winner == 0 else float(winner == 1 + i % 2)
        for i, winner in enumerate(winners)
    ]
    result = {
        **summarize(outcomes),
        "seed": seed,
        "reference_kind": kind,
        "reference_simulations": simulations,
        "reference_search": "ensemble_four_deal" if simulations else "none",
        "candidate_search": "none",
        "seconds": time.monotonic() - start,
        "candidate_seconds": seconds[0],
        "opponent_seconds": seconds[1],
        "candidate_moves": move_counts[0],
        "opponent_moves": move_counts[1],
        "timing_scope": "batched observation+device transfer+inference/search; synchronized CUDA; not single-move latency",
        "parameters_and_buffers_unchanged": True,
        "game_records": [
            {
                "index": game_offset + i,
                "deal_seed": seed + i // 2,
                "candidate_player": 1 + i % 2,
                "winner": winners[i],
                "plies": counts[i],
                "exact_actions": traces[i],
            }
            for i in range(games)
        ],
    }
    if candidate_action_mode == "sample":
        result.update(
            candidate_action_mode="sample", action_seed=action_seed, candidate_temperature=1.0,
            candidate_sampler="blake2b-64-person-seq-exact-eval/top53-float64-cdf-v1",
            opponent_action_mode="search" if simulations else "greedy",
        )
        for index, record in enumerate(result["game_records"]):
            record.update(action_seed=action_seed, candidate_decisions=candidate_counts[index])
    return result


def confirm(plan, output):
    output = Path(output)
    candidate_action_mode = plan.get("candidate_action_mode", "greedy")
    action_seed = plan.get("action_seed")
    validate_action_mode(candidate_action_mode, action_seed)
    batch_size = plan.get("batch_size", 256)
    if batch_size <= 0 or batch_size % 2 or plan.get("threads", 1) < 1:
        raise ValueError("Positive even batch size and positive threads required")
    labels = [row["label"] for row in plan["opponents"]]
    if len(labels) != len(set(labels)):
        raise ValueError("Opponent labels must be unique")
    for row in plan["opponents"]:
        if (
            row["games"] <= 0
            or row["games"] % 2
            or row["kind"] not in {"exact", "history", "champion"}
        ):
            raise ValueError("Invalid opponent specification")
        if row.get("simulations", 0) < 0 or (
            row["kind"] != "champion" and row.get("simulations", 0)
        ):
            raise ValueError("Only champion references may use search")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for entry in [plan["candidate"], *plan["opponents"]]:
        if digest(entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("Checkpoint checksum mismatch: " + entry["checkpoint"])
    report = {"status": "running", "plan": plan, "evaluations": []}
    if output.exists():
        report = json.loads(output.read_text())
        if report["plan"] != plan:
            raise ValueError("Cannot resume another plan")
        if report["status"] == "complete":
            return report
    torch.set_num_threads(plan.get("threads", 1))
    device = plan.get("device", "cpu")
    candidate, metadata = load_exact_policy(plan["candidate"]["checkpoint"], device)
    report["candidate_metadata"] = metadata
    elapsed_before = report.get("elapsed_seconds", 0)
    started = time.monotonic()
    for entry in plan["opponents"]:
        label = entry["label"]
        if any(row["label"] == label for row in report["evaluations"]):
            continue
        if entry["kind"] == "exact":
            opponent, _ = load_exact_policy(entry["checkpoint"], device)
        elif entry["kind"] == "history":
            opponent, _ = load_policy(entry["checkpoint"], device)
        else:
            opponent, _, _ = load_model(
                entry["checkpoint"], device=device, compile_model=False
            )
        active = report.get("active", {})
        outcomes = active.get("outcomes", []) if active.get("label") == label else []
        chunks = active.get("chunks", []) if active.get("label") == label else []
        if (
            len(outcomes) % 2
            or len(outcomes) > entry["games"]
            or any(value not in (0.0, 0.5, 1.0) for value in outcomes)
            or sum(chunk["games"] for chunk in chunks) != len(outcomes)
        ):
            raise ValueError("Invalid partial paired results")
        for offset in range(len(outcomes), entry["games"], batch_size):
            count = min(batch_size, entry["games"] - offset)
            result = evaluate_exact(
                candidate,
                opponent,
                games=count,
                seed=entry["seed"] + offset // 2,
                kind=entry["kind"],
                simulations=entry.get("simulations", 0),
                device=device,
                env_threads=plan.get("env_threads", 1),
                game_offset=offset,
                search_seed=entry["seed"],
                candidate_action_mode=candidate_action_mode,
                action_seed=action_seed,
            )
            outcomes.extend(result.pop("outcomes"))
            chunks.append(result)
            report["active"] = {"label": label, "outcomes": outcomes, "chunks": chunks}
            report["elapsed_seconds"] = elapsed_before + time.monotonic() - started
            atomic_json(output, report)
            print(
                json.dumps(
                    {
                        "label": label,
                        "games": len(outcomes),
                        "score": summarize(outcomes)["score"],
                    }
                ),
                flush=True,
            )
        report["evaluations"].append(
            {"label": label, **summarize(outcomes), "chunks": chunks}
        )
        report.pop("active", None)
        atomic_json(output, report)
    for entry in [plan["candidate"], *plan["opponents"]]:
        if digest(entry["checkpoint"]) != entry["sha256"]:
            raise RuntimeError("Checkpoint changed during evaluation")
    report.update(
        status="complete",
        parameters_and_buffers_unchanged=True,
        elapsed_seconds=elapsed_before + time.monotonic() - started,
    )
    atomic_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    confirm(json.loads(Path(args.plan).read_text()), args.output)
