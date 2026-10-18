"""Controlled direct-policy architecture experiment on public-information histories.

Distill one fixed four-deal teacher into identically initialized CNN controls,
public-summary, GRU, and attention models. Split by complete game; evaluate on
fresh paired deals against the frozen teacher and a held-out greedy opponent.
This is supervised search distillation, not ReBeL or equilibrium learning.
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

from scripts.arena import choose_moves, summarize
from scripts.evaluate import predict
from scripts.overnight import atomic_copy, atomic_json
from src.belief import unseen_cards
from src.cards import Card
from src.fast_game import FastSequenceGame
from src.game import Move
from src.history_policy import (
    HistoryPolicy,
    apply_public_move,
    batch_inputs,
    load_policy,
    observe,
    save_policy,
)
from src.mcts import GreedyPlayer
from src.model import load_model


def teacher_moves(model, device, games, ids, counts, seed, simulations):
    searches = []
    for i in ids:
        g = games[i]
        p = g.current_player
        search = sequence_native.EnsembleMCTS(seed + i * 1000 + counts[i])
        search.reset(
            g.c_state.get_tensor(),
            [c.to_int() for c in g.hands[p]],
            p,
            g.c_state.get_sequences(),
            unseen_cards(g, p),
            len(g.hands[3 - p]),
        )
        searches.append(search)
    for _ in range(simulations):
        pending = []
        states = []
        for search in searches:
            token, data, value, terminal = search.select_leaf()
            if terminal:
                search.backpropagate(token, b"", value)
            else:
                pending.append((search, token))
                states.append(np.frombuffer(data, np.float32).reshape(11, 10, 10))
        if states:
            policies, values = predict(model, device, states)
            for (search, token), pi, value in zip(pending, policies, values):
                search.backpropagate(token, pi.tobytes(), float(value))
    result = []
    for i, search in zip(ids, searches):
        card, r, c, rem, policy = search.get_action(1.0 if counts[i] < 12 else 0.1)
        result.append(
            (
                Move(Card.from_int(card), r, c, rem),
                np.frombuffer(policy, np.float32).copy(),
            )
        )
    return result


def collect(teacher, device, count, seed, simulations, output):
    rows = []
    targets = []
    values = []
    game_ids = []
    start = time.monotonic()
    for offset in range(0, count, 128):
        games = []
        for i in range(offset, min(offset + 128, count)):
            random.seed(seed + i)
            games.append(FastSequenceGame())
        events = [[] for _ in games]
        records = [[] for _ in games]
        counts = [0] * len(games)
        active = list(range(len(games)))
        while active:
            moves = teacher_moves(
                teacher,
                device,
                games,
                active,
                counts,
                seed + offset * 1000,
                simulations,
            )
            for i, (move, pi) in zip(active, moves):
                g = games[i]
                records[i].append((observe(g, events[i]), pi, g.current_player))
                apply_public_move(g, events[i], move)
                counts[i] += 1
            active = [
                i
                for i in active
                if not games[i].game_over
                and counts[i] < 200
                and games[i].get_legal_moves()
            ]
        for i, g in enumerate(games):
            for observation, pi, player in records[i]:
                rows.append(observation)
                targets.append(pi)
                values.append(g.get_result(player))
                game_ids.append(offset + i)
        print(
            json.dumps(
                {
                    "stage": "collect",
                    "games": offset + len(games),
                    "positions": len(rows),
                    "seconds": time.monotonic() - start,
                }
            ),
            flush=True,
        )
    dataset = {
        "board": torch.from_numpy(np.stack([r[0] for r in rows])),
        "context": torch.from_numpy(np.stack([r[1] for r in rows])),
        "history": torch.from_numpy(np.stack([r[2] for r in rows]).astype(np.uint8)),
        "policy": torch.from_numpy(np.stack(targets)),
        "value": torch.tensor(values, dtype=torch.float32),
        "game": torch.tensor(game_ids),
    }
    temporary = output.with_suffix(".tmp")
    torch.save(dataset, temporary)
    temporary.replace(output)
    return dataset


def inputs_from_data(data, indices, device):
    return (
        data["board"][indices].to(device).float(),
        data["context"][indices].to(device),
        data["history"][indices].to(device).long(),
    )


def validation_losses(model, data, indices, device):
    model.eval()
    policy_total = value_total = 0.0
    with torch.inference_mode():
        for ids in indices.split(256):
            logpi, value = model(*inputs_from_data(data, ids, device))
            policy_total += float(-(data["policy"][ids].to(device) * logpi).sum())
            value_total += float((value - data["value"][ids].to(device)).square().sum())
    return {
        "heldout_policy_cross_entropy": policy_total / len(indices),
        "heldout_value_mse": value_total / len(indices),
        "heldout_game_loss": (policy_total + value_total) / len(indices),
    }


def fit(
    base,
    architecture,
    data,
    device,
    epochs,
    seed,
    output,
    learning_rate=0.0001,
    freeze_backbone=False,
):
    torch.manual_seed(seed)
    model = HistoryPolicy(base, architecture).to(device)
    if freeze_backbone:
        if architecture == "board":
            raise ValueError("A frozen board-only control has no trainable parameters")
        model.base.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=0.0001,
    )
    train_ids = torch.where(data["game"] % 8 != 0)[0]
    validation_ids = torch.where(data["game"] % 8 == 0)[0]
    if not len(train_ids) or not len(validation_ids):
        raise ValueError("Need separate training and validation games")
    start = time.monotonic()
    initial = validation_losses(model, data, validation_ids, device)
    best = initial["heldout_game_loss"]
    metrics = [{"architecture": architecture, "epoch": 0, **initial, "seconds": 0.0}]
    metadata = {
        "architecture": architecture,
        "seed": seed,
        "training": "four-deal search distillation",
        "learning_rate": learning_rate,
        "freeze_backbone": freeze_backbone,
    }
    # An unsuccessful experiment must be allowed to retain its starting weights.
    save_policy(model, output, {**metadata, "epoch": 0, **initial})
    for epoch in range(epochs):
        model.train()
        if freeze_backbone:
            model.base.eval()
        permutation = train_ids[torch.randperm(len(train_ids))]
        losses = []
        for ids in permutation.split(256):
            logpi, value = model(*inputs_from_data(data, ids, device))
            loss = -(data["policy"][ids].to(device) * logpi).sum(
                1
            ).mean() + torch.nn.functional.mse_loss(
                value, data["value"][ids].to(device)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite architecture loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = validation_losses(model, data, validation_ids, device)
        row = {
            "architecture": architecture,
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            **validation,
            "seconds": time.monotonic() - start,
        }
        metrics.append(row)
        print(json.dumps(row), flush=True)
        if validation["heldout_game_loss"] < best:
            best = validation["heldout_game_loss"]
            save_policy(model, output, {**metadata, "epoch": epoch + 1, **validation})
        atomic_json(output.with_suffix(".metrics.json"), metrics)
    return load_policy(output, device)[0]


def direct_moves(model, device, games, events, ids):
    if not ids:
        return []
    with torch.inference_mode():
        logpi, _ = model(
            *batch_inputs([observe(games[i], events[i]) for i in ids], device)
        )
    policies = logpi.float().cpu().numpy()
    return [
        max(
            games[i].get_legal_moves(),
            key=lambda m: (float(pi[m.row * 10 + m.col]), not m.card.is_jack()),
        )
        for i, pi in zip(ids, policies)
    ]


def evaluate_policy(
    candidate, teacher, device, count, seed, opponent="champion", simulations=0
):
    if count <= 0 or count % 2:
        raise ValueError("Evaluation requires positive paired games")
    if opponent not in {"champion", "greedy", "history"}:
        raise ValueError("Unknown evaluation opponent")
    if opponent == "history" and simulations:
        raise ValueError("History opponents use direct policy only")
    # Checkpoint loaders return train-mode modules. Both sides must freeze
    # BatchNorm here, including callers outside this script's main function.
    candidate.eval()
    teacher.eval()
    games = []
    for i in range(count):
        random.seed(seed + i // 2)
        games.append(FastSequenceGame())
    events = [[] for _ in games]
    counts = [0] * count
    active = set(range(count))
    greedy = GreedyPlayer()
    start = time.monotonic()
    candidate_seconds = 0
    candidate_moves = 0
    while active:
        for side in (0, 1):
            ids = []
            for i in sorted(active):
                g = games[i]
                if g.game_over or counts[i] >= 200 or not g.get_legal_moves():
                    active.remove(i)
                elif (g.current_player == 1 + i % 2) == (side == 0):
                    ids.append(i)
            if not ids:
                continue
            if side == 0:
                tick = time.monotonic()
                moves = direct_moves(candidate, device, games, events, ids)
                candidate_seconds += time.monotonic() - tick
                candidate_moves += len(ids)
            elif opponent == "greedy":
                moves = [
                    greedy.get_move(games[i], games[i].current_player) for i in ids
                ]
            elif opponent == "history":
                moves = direct_moves(teacher, device, games, events, ids)
            else:
                moves = choose_moves(
                    teacher, device, games, ids, simulations, seed, counts, "ensemble"
                )
            for i, move in zip(ids, moves):
                if move is None:
                    active.discard(i)
                else:
                    apply_public_move(games[i], events[i], move)
                    counts[i] += 1
    outcomes = [
        1.0 if g.winner == 1 + i % 2 else 0.0 if g.winner == 2 - i % 2 else 0.5
        for i, g in enumerate(games)
    ]
    return {
        **summarize(outcomes),
        "seed": seed,
        "opponent": opponent,
        "opponent_simulations": simulations,
        "candidate_simulations": 0,
        "seconds": time.monotonic() - start,
        "candidate_seconds": candidate_seconds,
        "candidate_moves": candidate_moves,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--champion", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--teacher-simulations", type=int, default=32)
    p.add_argument("--evaluation-games", type=int, default=256)
    p.add_argument("--architectures", default="board,summary,gru,attention")
    p.add_argument("--seed", type=int, default=801711)
    a = p.parse_args()
    if a.learning_rate <= 0:
        p.error("Learning rate must be positive")
    if a.freeze_backbone and "board" in a.architectures.split(","):
        p.error("Frozen board-only models have no trainable parameters")
    if a.evaluation_games < 4 or a.evaluation_games % 4:
        p.error("Evaluation games must be a positive multiple of four")
    if a.games < 8 or a.epochs < 1 or a.teacher_simulations < 4:
        p.error("Need >=8 games, positive epochs, >=4 teacher simulations")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture campaign requires CUDA")
    snapshot = output / "teacher.pt"
    if not snapshot.exists():
        atomic_copy(a.champion, snapshot)
    teacher, device, _ = load_model(snapshot, device="cuda", compile_model=False)
    teacher.eval()
    data_path = output / "training_data.pt"
    data = (
        torch.load(data_path, weights_only=True)
        if data_path.exists()
        else collect(teacher, device, a.games, a.seed, a.teacher_simulations, data_path)
    )
    manifest = {
        "config": vars(a),
        "teacher_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "positions": len(data["game"]),
        "train_games": [i for i in range(a.games) if i % 8 != 0],
        "validation_games": [i for i in range(a.games) if i % 8 == 0],
        "training": "Supervised four-deal search distillation; not ReBeL or independent reinforcement learning",
        "production_changed": False,
    }
    atomic_json(output / "manifest.json", manifest)
    for architecture in a.architectures.split(","):
        checkpoint = output / (architecture + ".pt")
        result_path = output / (architecture + ".results.json")
        results = json.loads(result_path.read_text()) if result_path.exists() else None
        if results and len(results["evaluations"]) == 3:
            continue
        if results:
            candidate = load_policy(checkpoint, device)[0]
        else:
            candidate = fit(
                teacher,
                architecture,
                data,
                device,
                a.epochs,
                a.seed + 1,
                checkpoint,
                learning_rate=a.learning_rate,
                freeze_backbone=a.freeze_backbone,
            )
        results = results or {
            "architecture": architecture,
            "parameters": sum(p.numel() for p in candidate.parameters()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "evaluations": [],
        }
        specifications = [
            ("champion", 0, a.evaluation_games, a.seed + 100000),
            ("greedy", 0, a.evaluation_games, a.seed + 200000),
            ("champion", 128, a.evaluation_games // 2, a.seed + 300000),
        ]
        for opponent, sims, count, seed in specifications[
            len(results["evaluations"]) :
        ]:
            result = evaluate_policy(
                candidate, teacher, device, count, seed, opponent, sims
            )
            results["evaluations"].append(result)
            atomic_json(result_path, results)
            print(
                json.dumps(
                    {
                        "architecture": architecture,
                        **{k: v for k, v in result.items() if k != "outcomes"},
                    }
                ),
                flush=True,
            )
        del candidate
    print("Architecture comparison completed", flush=True)


if __name__ == "__main__":
    main()
