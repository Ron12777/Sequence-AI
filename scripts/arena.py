"""Color-balanced, batched matches between two checkpoint agents."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.belief import search_class, unseen_cards
from scripts.evaluate import predict
from src.cards import Card
from src.fast_game import FastSequenceGame
from src.game import Move
from src.model import load_model


def choose_moves(
    model, device, games, ids, simulations, seed, move_counts, search_mode="single"
):
    if not ids:
        return []
    if simulations == 0:
        policies, _ = predict(
            model,
            device,
            [games[i].get_state_numpy(games[i].current_player) for i in ids],
        )
        return [
            max(
                games[i].get_legal_moves(),
                key=lambda m: (float(pi[m.row * 10 + m.col]), not m.card.is_jack()),
            )
            for i, pi in zip(ids, policies)
        ]
    searches = []
    for i in ids:
        g = games[i]
        p = g.current_player
        search = search_class(search_mode)(seed + i * 1000 + move_counts[i])
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
        pending, states = [], []
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
    for search in searches:
        action = search.get_action(0.0)
        result.append(
            None
            if action is None
            else Move(Card.from_int(action[0]), action[1], action[2], action[3])
        )
    return result


def summarize(outcomes):
    """Use paired deals as the uncertainty unit because the colors share a deal."""
    values = np.asarray(outcomes, dtype=float)
    if len(values) == 0 or len(values) % 2:
        raise ValueError("An even, positive number of outcomes is required")
    pairs = values.reshape(-1, 2).mean(axis=1)
    score = float(values.mean())
    half = (
        1.96 * float(pairs.std(ddof=1)) / math.sqrt(len(pairs))
        if len(pairs) > 1
        else 1.0
    )
    return {
        "games": len(values),
        "wins": int((values == 1).sum()),
        "losses": int((values == 0).sum()),
        "draws": int((values == 0.5).sum()),
        "score": score,
        "paired_score_95ci": [max(0.0, score - half), min(1.0, score + half)],
        "outcomes": outcomes,
    }


def arena(
    candidate,
    champion,
    games_count=512,
    simulations=32,
    seed=271828,
    candidate_search="single",
    champion_search="single",
    champion_simulations=None,
):
    if games_count <= 0 or games_count % 2 or simulations < 0:
        raise ValueError("Use a positive even game count and non-negative simulations")
    models = [candidate, champion]
    modes = [candidate_search, champion_search]
    budgets = [
        simulations,
        simulations if champion_simulations is None else champion_simulations,
    ]
    games = []
    for i in range(games_count):
        random.seed(seed + i // 2)
        games.append(FastSequenceGame())
    counts = [0] * games_count
    active = set(range(games_count))
    start = time.monotonic()
    while active:
        for side in (0, 1):
            ids = []
            for i in sorted(active):
                g = games[i]
                if g.game_over or counts[i] >= 200 or not g.get_legal_moves():
                    active.remove(i)
                elif (g.current_player == 1 + i % 2) == (side == 0):
                    ids.append(i)
            model, device = models[side]
            for i, move in zip(
                ids,
                choose_moves(
                    model, device, games, ids, budgets[side], seed, counts, modes[side]
                ),
            ):
                if move is None:
                    active.remove(i)
                elif not games[i].make_move(move):
                    raise RuntimeError("Arena agent returned an illegal move")
                else:
                    counts[i] += 1
    outcomes = [
        1.0 if g.winner == 1 + i % 2 else 0.0 if g.winner == 2 - i % 2 else 0.5
        for i, g in enumerate(games)
    ]
    return {
        **summarize(outcomes),
        "seed": seed,
        "simulations": simulations,
        "champion_simulations": budgets[1],
        "candidate_search": candidate_search,
        "champion_search": champion_search,
        "seconds": time.monotonic() - start,
        "mean_moves": float(np.mean(counts)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--champion", required=True)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--simulations", type=int, default=32)
    p.add_argument("--seed", type=int, default=271828)
    p.add_argument("--candidate-search", choices=["ensemble", "single"], default="single")
    p.add_argument("--champion-search", choices=["ensemble", "single"], default="single")
    p.add_argument("--champion-simulations", type=int)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    models = []
    for path in (a.candidate, a.champion):
        model, device, _ = load_model(path, compile_model=False)
        model.eval().to(memory_format=torch.channels_last)
        models.append((model, device))
    result = arena(
        *models,
        a.games,
        a.simulations,
        a.seed,
        a.candidate_search,
        a.champion_search,
        a.champion_simulations,
    )
    result.update(candidate=a.candidate, champion=a.champion)
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "outcomes"}), flush=True)


if __name__ == "__main__":
    main()
