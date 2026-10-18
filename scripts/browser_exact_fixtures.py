"""Generate full-game Python observations/predictions for browser parity checks."""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.exact_policy import (
    ExactPolicy,
    action_id,
    batch_inputs,
    legal_actions,
    load_exact_policy,
    observe_exact,
)
from src.fast_game import FastSequenceGame
from src.history_policy import HistoryPolicy, apply_public_move
from src.model import SequenceNet


def generate(model, games=8, seed=7100):
    model.eval()
    fixtures = []
    for index in range(games):
        random.seed(seed + index)
        game, events = FastSequenceGame(), []
        fixture = {"hands": {p: list(map(str, hand)) for p, hand in game.hands.items()},
                   "deck": list(map(str, game.deck.cards)), "steps": []}
        while not game.game_over:
            moves, actions = legal_actions(game)
            if not moves:
                break
            observation = observe_exact(game, events)
            with torch.inference_mode():
                logits, value = model(*batch_inputs([observation], "cpu"), torch.from_numpy(actions)[None])
                probabilities = logits.softmax(1)[0].numpy()
            policy = np.zeros(100, np.float32)
            np.add.at(policy, actions % 100, probabilities)
            move = random.choice(game.get_legal_moves())
            fixture["steps"].append({
                **{name: data.flatten().tolist() for name, data in zip(("board", "context", "history"), observation)},
                "actions": actions.tolist(), "logits": logits[0].tolist(), "value": value.item(),
                "chosen": action_id(moves[int(logits.argmax(1))]), "policy": policy.tolist(),
                "move": {"card": str(move.card), "row": move.row, "col": move.col, "is_removal": move.is_removal},
            })
            apply_public_move(game, events, move)
        fixture["events"] = events
        fixtures.append(fixture)
    return fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--games", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.checkpoint:
        model, _ = load_exact_policy(args.checkpoint, "cpu")
    else:
        torch.manual_seed(918)
        model = ExactPolicy.from_history_policy(HistoryPolicy(SequenceNet(num_filters=8, num_res_blocks=1), "board", width=8))
        # Exercise learned card preferences as well as transferred square scores.
        with torch.no_grad():
            model.spatial_projection.weight.normal_(std=0.1)
            model.card_bias.normal_(std=0.1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(generate(model, args.games)))


if __name__ == "__main__":
    main()
