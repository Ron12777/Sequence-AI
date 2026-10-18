"""Generate real-state browser fixtures for an original-hand-order board policy."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.fast_game import FastSequenceGame
from src.history_policy import apply_public_move, batch_inputs, load_policy, observe


def generate(checkpoint, count=32, seed=731813):
    checkpoint = Path(checkpoint)
    model, _ = load_policy(checkpoint, "cpu")
    if model.architecture != "board":
        raise ValueError("This fixture contract is for original-hand-order board policies")
    random.seed(seed)
    rows, seen = [], 0
    while len(rows) < count:
        game, events = FastSequenceGame(), []
        while not game.game_over and len(rows) < count:
            moves = game.get_legal_moves()
            if not moves:
                break
            if seen % 4 == 0:
                inputs = batch_inputs([observe(game, events)], "cpu")
                with torch.inference_mode():
                    logits, value = model(*inputs)
                logpi = logits[0].numpy()
                chosen = max(moves, key=lambda move: (logpi[move.row * 10 + move.col], not move.card.is_jack()))
                positions = sorted({move.row * 10 + move.col for move in moves})
                weights = np.exp(logpi[positions] - np.max(logpi[positions]))
                policy = np.zeros(100, np.float32)
                policy[positions] = weights / weights.sum()
                rows.append({
                    "state": {"board": game.board.tolist(), "currentPlayer": game.current_player,
                              "hands": {p: list(map(str, hand)) for p, hand in game.hands.items()},
                              "completedSequences": {p: [[f"{r},{c}" for r, c in sorted(seq)] for seq in seqs]
                                                     for p, seqs in game.completed_sequences.items()}},
                    "input": inputs[0].flatten().tolist(), "logits": logpi.tolist(),
                    "value": value.item(), "policy": policy.tolist(), "public_events": len(events),
                    "move": {"card": str(chosen.card), "row": chosen.row, "col": chosen.col,
                             "is_removal": chosen.is_removal},
                })
            apply_public_move(game, events, random.choice(moves))
            seen += 1
    return {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "seed": seed, "observation": "original own-hand order; no canonicalization", "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(generate(args.checkpoint, args.states)))
