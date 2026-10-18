"""Incremental exact CFR reference for the fixed restricted Sequence benchmark."""

import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.public_belief_cfr import PublicBeliefCFR
from src.rebel_sequence_game import SequenceBeliefGame


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    output = Path(__file__).resolve().parent
    target = output / "exact-oracle.json"
    started = time.monotonic()
    result = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "algorithm": "Simultaneous vanilla CFR; uniform iteration weighting; own-reach averaged behavior policies",
        "scope": "Fixed synthetic restricted Sequence endgame variant0; exact full-tree utilities and best responses; not full-game Sequence",
        "threads": 1,
        "device": "cpu",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
            ROOT / "src/public_belief_cfr.py", ROOT / "src/rebel_sequence_game.py", Path(__file__).resolve(),
        )},
        "checkpoints": [],
    }
    save_json(target, result)
    game = SequenceBeliefGame(0)
    solver = PublicBeliefCFR(game.root, game.prior)
    result["game"] = game.summary()
    result["build_seconds"] = time.monotonic() - started
    sums = {key: np.zeros(mask.shape) for key, mask in solver.masks.items()}
    root_sum = [np.zeros(n) for n in solver.shape]
    cfr_seconds = 0.0
    before = time.monotonic()
    for step in solver.iterations(1024):
        cfr_seconds += time.monotonic() - before
        for key, policy in step.strategies.items():
            actor = solver.nodes[key].actor
            sums[key] += step.evaluation.beliefs[key].reaches[actor][:, None] * policy
        for player in (0, 1):
            root_sum[player] += step.evaluation.counterfactual_values[player]
        if step.iteration in (32, 128, 512, 1024):
            policies = {}
            for key, total in sums.items():
                mass = total.sum(axis=1, keepdims=True)
                legal = solver.masks[key]
                fallback = np.divide(legal, legal.sum(axis=1, keepdims=True),
                                     out=np.zeros(legal.shape), where=legal.sum(axis=1, keepdims=True) > 0)
                policies[key] = np.divide(total, mass, out=fallback, where=mass > 0)
            measured = time.monotonic()
            evaluation = solver.evaluate(policies)
            br = [solver.best_response(policies, p)[0] for p in (0, 1)]
            filename = output / f"exact-oracle-{step.iteration:04d}.pkl"
            temporary = filename.with_suffix(".tmp")
            temporary.write_bytes(pickle.dumps({
                "variant": 0, "iterations": step.iteration, "strategies": policies,
                "joint_prior": game.prior,
                "root_mean_counterfactual_values": tuple(v / step.iteration for v in root_sum),
            }, protocol=pickle.HIGHEST_PROTOCOL))
            temporary.replace(filename)
            check = {
                "iterations": step.iteration,
                "value": evaluation.value,
                "best_response_player0": br[0], "best_response_player1": br[1],
                "exploitability": max(0.0, sum(br) / 2),
                "root_average_policy_conditional_values": [v.tolist() for v in evaluation.conditional_values],
                "root_mean_iteration_counterfactual_values": [(v / step.iteration).tolist() for v in root_sum],
                "cfr_seconds": cfr_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "evaluation_and_save_seconds": time.monotonic() - measured,
                "strategy_file": str(filename.relative_to(ROOT)),
                "strategy_sha256": digest(filename),
            }
            result["checkpoints"].append(check)
            save_json(target, result)
            print(json.dumps({k: check[k] for k in ("iterations", "value", "exploitability", "elapsed_seconds")}), flush=True)
        before = time.monotonic()
    result.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - started)
    save_json(target, result)


if __name__ == "__main__":
    main()
