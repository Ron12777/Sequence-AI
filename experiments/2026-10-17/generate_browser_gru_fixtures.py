import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch

from scripts.architecture_experiments import direct_moves
from src.fast_game import FastSequenceGame
from src.history_policy import apply_public_move, batch_inputs, load_policy, observe

parser = argparse.ArgumentParser(
    description="Generate exact-checkpoint browser parity fixtures"
)
parser.add_argument("--checkpoint", default="artifacts/checkpoints/gru-two-hour.pt")
parser.add_argument(
    "--output", default="experiments/2026-10-17/browser-gru-fixtures.json"
)
args = parser.parse_args()
model, _ = load_policy(args.checkpoint, "cpu")
model.eval()
torch.set_num_threads(2)
fixtures = []
for seed in range(3):
    random.seed(98917 + seed)
    g = FastSequenceGame()
    events = []
    f = {
        "hands": {p: list(map(str, h)) for p, h in g.hands.items()},
        "deck": list(map(str, g.deck.cards)),
        "steps": [],
    }
    while not g.game_over:
        moves = g.get_legal_moves()
        if not moves:
            break
        with torch.inference_mode():
            pi, v = model(*batch_inputs([observe(g, events)], "cpu"))
        best = direct_moves(model, "cpu", [g], [events], [0])[0]
        move = random.choice(moves)

        def pack(m):
            return {
                "card": str(m.card),
                "row": m.row,
                "col": m.col,
                "is_removal": m.is_removal,
            }

        f["steps"].append(
            {
                "logits": pi[0].tolist(),
                "value": v.item(),
                "best": pack(best),
                "move": pack(move),
            }
        )
        apply_public_move(g, events, move)
    fixtures.append(f)
Path(args.output).write_text(json.dumps(fixtures))
print(sum(len(f["steps"]) for f in fixtures), "states")
