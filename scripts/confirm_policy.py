"""Resumable CPU confirmation from an explicit checkpoint/opponent/seed plan."""

import argparse
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from scripts.architecture_experiments import evaluate_policy
from scripts.arena import summarize
from scripts.overnight import atomic_json
from src.history_policy import load_policy
from src.model import load_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confirm(plan, output):
    if plan["threads"] < 1:
        raise ValueError("Positive thread count required")
    if len({x["label"] for x in plan["opponents"]}) != len(plan["opponents"]):
        raise ValueError("Opponent labels must be unique")
    for opponent in plan["opponents"]:
        if opponent["games"] <= 0 or opponent["games"] % 256:
            raise ValueError("Game counts must be positive multiples of 256")
        if opponent["kind"] not in {"history", "champion"}:
            raise ValueError("Unsupported opponent contract")
        if opponent["simulations"] < 0 or (
            opponent["kind"] == "history" and opponent["simulations"]
        ):
            raise ValueError("Invalid opponent simulation count")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(plan["threads"])
    for entry in [plan["candidate"], *plan["opponents"]]:
        if digest(entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("Checkpoint checksum mismatch: " + entry["checkpoint"])
    report = {"status": "running", "device": "cpu", "plan": plan, "evaluations": []}
    if output.exists():
        report = json.loads(output.read_text())
        if report["plan"] != plan:
            raise ValueError("Cannot resume a different confirmation plan")
        if report["status"] == "complete":
            return report
    candidate, metadata = load_policy(plan["candidate"]["checkpoint"], "cpu")
    candidate.eval()
    candidate_before = {k: v.clone() for k, v in candidate.state_dict().items()}
    report["candidate_metadata"] = metadata
    elapsed_before = report.get("elapsed_seconds", 0)
    start = time.monotonic()
    for opponent in plan["opponents"]:
        label = opponent["label"]
        if any(x["label"] == label for x in report["evaluations"]):
            continue
        if opponent["kind"] == "history":
            teacher, _ = load_policy(opponent["checkpoint"], "cpu")
        else:
            teacher, _, _ = load_model(
                opponent["checkpoint"], device="cpu", compile_model=False
            )
        teacher.eval()
        before = {k: v.clone() for k, v in teacher.state_dict().items()}
        active = report.get("active", {})
        outcomes, chunks = [], []
        if active.get("label") == label:
            outcomes, chunks = active["summary"]["outcomes"], active["chunks"]
            if len(outcomes) % 256 or len(outcomes) > opponent["games"]:
                raise ValueError("Invalid partial results")
        for offset in range(len(outcomes), opponent["games"], 256):
            result = evaluate_policy(
                candidate,
                teacher,
                torch.device("cpu"),
                256,
                opponent["seed"] + offset // 2,
                opponent=opponent["kind"],
                simulations=opponent["simulations"],
            )
            outcomes.extend(result.pop("outcomes"))
            chunks.append(result)
            for model, expected in [(candidate, candidate_before), (teacher, before)]:
                if not all(
                    torch.equal(expected[k], v) for k, v in model.state_dict().items()
                ):
                    raise RuntimeError("Evaluation mutated model parameters or buffers")
            report["active"] = {
                "label": label,
                "completed_games": len(outcomes),
                "summary": summarize(outcomes),
                "chunks": chunks,
            }
            report["elapsed_seconds"] = elapsed_before + time.monotonic() - start
            atomic_json(output, report)
            print(
                label, len(outcomes), report["active"]["summary"]["score"], flush=True
            )
        report["evaluations"].append(
            {"label": label, **summarize(outcomes), "chunks": chunks}
        )
        report.pop("active", None)
        atomic_json(output, report)
    report.update(
        status="complete",
        parameters_and_buffers_unchanged=True,
        elapsed_seconds=elapsed_before + time.monotonic() - start,
    )
    atomic_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    confirm(json.loads(Path(args.plan).read_text()), args.output)
