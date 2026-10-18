"""Seeded, color-balanced evaluation of policy-only and search agents."""

import argparse
import json
import math
import random
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.belief import search_class, unseen_cards
from src.cards import Card
from src.game import Move
from src.fast_game import FastSequenceGame
from src.mcts import GreedyPlayer
from src.model import create_model, load_model


def predict(model, device, states):
    if model.training:
        raise ValueError(
            "Evaluation prediction requires model.eval(); BatchNorm must remain frozen"
        )
    x = torch.from_numpy(np.stack(states)).to(device, memory_format=torch.channels_last)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ),
    ):
        logpi, value = model(x)
    return logpi.exp().float().cpu().numpy(), value.float().cpu().numpy().reshape(-1)


def evaluate(
    model, device, games_count, simulations, seed, opponent, search_mode="single"
):
    if games_count <= 0 or games_count % 2 or simulations < 0:
        raise ValueError("Use a positive even game count and non-negative simulations")
    if opponent not in {"random", "greedy"}:
        raise ValueError("Opponent must be random or greedy")
    random.seed(seed)
    np.random.seed(seed)
    games = []
    for i in range(games_count):
        random.seed(seed + i // 2)
        games.append(FastSequenceGame())
    rng = random.Random(seed)
    greedy = GreedyPlayer()
    active = set(range(games_count))
    moves = [0] * games_count
    nodes = 0
    start = time.monotonic()
    while active:
        searching = []
        for i in sorted(active):
            game = games[i]
            legal = game.get_legal_moves()
            if game.game_over or not legal or moves[i] >= 200:
                active.remove(i)
                continue
            ai = 1 + i % 2
            if game.current_player != ai:
                move = (
                    rng.choice(legal)
                    if opponent == "random"
                    else greedy.get_move(game, game.current_player)
                )
                if not game.make_move(move):
                    raise RuntimeError("illegal baseline move")
                moves[i] += 1
            else:
                searching.append(i)
        if not searching:
            continue
        if simulations == 0:
            policies, _ = predict(
                model,
                device,
                [games[i].get_state_tensor(games[i].current_player) for i in searching],
            )
            nodes += len(searching)
            for i, policy in zip(searching, policies):
                game = games[i]
                # Stable tie-breaking preserves a wild card when an ordinary card targets the same square.
                move = max(
                    game.get_legal_moves(),
                    key=lambda m: (
                        float(policy[m.row * 10 + m.col]),
                        not m.card.is_jack(),
                    ),
                )
                if not game.make_move(move):
                    raise RuntimeError("illegal policy move")
                moves[i] += 1
        else:
            searches = []
            for i in searching:
                game = games[i]
                p = game.current_player
                search = search_class(search_mode)(seed + i * 1000 + moves[i])
                search.reset(
                    game.c_state.get_tensor(),
                    [c.to_int() for c in game.hands[p]],
                    p,
                    game.c_state.get_sequences(),
                    unseen_cards(game, p),
                    len(game.hands[3 - p]),
                )
                searches.append(search)
            for _ in range(simulations):
                pending = []
                states = []
                for search in searches:
                    token, tensor, value, terminal = search.select_leaf()
                    if terminal:
                        search.backpropagate(token, b"", value)
                    else:
                        pending.append((search, token))
                        states.append(
                            np.frombuffer(tensor, np.float32).reshape(11, 10, 10)
                        )
                if states:
                    policies, values = predict(model, device, states)
                    nodes += len(states)
                    for (search, token), policy, value in zip(
                        pending, policies, values
                    ):
                        search.backpropagate(token, policy.tobytes(), float(value))
            for i, search in zip(searching, searches):
                result = search.get_action(0.0)
                if result is None:
                    active.remove(i)
                    continue
                card, r, c, removal, _ = result
                if not games[i].make_move(Move(Card.from_int(card), r, c, removal)):
                    raise RuntimeError("illegal search move")
                moves[i] += 1
    wins = sum(g.winner == 1 + i % 2 for i, g in enumerate(games))
    losses = sum(g.winner == 2 - i % 2 for i, g in enumerate(games))
    draws = games_count - wins - losses
    # Wilson interval for decisive wins; draws are reported separately.
    n = wins + losses
    if n:
        z = 1.96
        p = wins / n
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        interval = [center - half, center + half]
    else:
        interval = None
    return {
        "simulations": simulations,
        "search_mode": search_mode,
        "opponent": opponent,
        "games": games_count,
        "seed": seed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": (wins + 0.5 * draws) / games_count,
        "decisive_win_rate_95ci": interval,
        "seconds": time.monotonic() - start,
        "neural_evaluations": nodes,
        "mean_moves": sum(moves) / games_count,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--search-mode", choices=["ensemble", "single"], default="single")
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--simulations", default="0,8,32")
    p.add_argument("--seed", type=int, default=104729)
    p.add_argument("--opponent", choices=["random", "greedy"], default="greedy")
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--output", default="artifacts/evaluation.jsonl")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if args.checkpoint:
        model, device, _ = load_model(args.checkpoint, compile_model=False)
    else:
        model, device = create_model(
            compile_model=False, num_filters=args.filters, num_res_blocks=args.blocks
        )
    model.eval().to(memory_format=torch.channels_last)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for sims in map(int, args.simulations.split(",")):
        result = evaluate(
            model, device, args.games, sims, args.seed, args.opponent, args.search_mode
        )
        result["checkpoint"] = args.checkpoint or "untrained"
        print(json.dumps(result), flush=True)
        with output.open("a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
