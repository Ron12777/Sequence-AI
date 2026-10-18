"""Independent CPU confirmation; leaves the rental's training uninterrupted."""

import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from scripts.architecture_experiments import evaluate_policy
from scripts.arena import summarize
from scripts.overnight import atomic_json
from src.history_policy import load_policy
from src.model import load_model

output = Path(__file__).with_name("gru-two-hour-confirmation.json")
lock = output.with_suffix(".lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
torch.set_num_threads(2)
candidate_path = ROOT / "artifacts/checkpoints/gru-two-hour.pt"
sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
assert sha == "63c0efcf79237f1ba019f22d2ce55cee50156104c78b072dc2f4008bddef4cb2"
candidate, metadata = load_policy(candidate_path, "cpu")
candidate.eval()
report = {
    "status": "running",
    "device": "cpu",
    "threads": 2,
    "candidate_sha256": sha,
    "metadata": metadata,
    "candidate_simulations": 0,
    "seed": 800000000001,
    "purpose": "Fresh intermediate confirmation, separate from final reserved seeds",
    "evaluations": [],
}
if output.exists():
    previous = json.loads(output.read_text())
    assert previous["candidate_sha256"] == sha
    assert previous["seed"] == report["seed"]
    assert previous["device"] == "cpu"
    if previous["status"] == "complete":
        sys.exit(0)
    report = previous
elapsed_before = report.get("elapsed_seconds", 0)
start = time.monotonic()
for label, checkpoint, simulations, count, seed in [
    ("champion_search", "overnight-round0.pt", 128, 2048, 800000000001),
    ("heldout_baseline", "small-standard.pt", 32, 512, 800001000001),
]:
    if any(e["label"] == label for e in report["evaluations"]):
        continue
    teacher, device, _ = load_model(
        ROOT / "artifacts/checkpoints" / checkpoint, device="cpu", compile_model=False
    )
    teacher.eval()
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    outcomes, chunks = [], []
    active = report.get("active", {})
    if active.get("label") == label:
        outcomes = active["summary"]["outcomes"]
        chunks = active["chunks"]
        assert len(outcomes) == active["completed_games"]
        assert len(outcomes) % 256 == 0
    for offset in range(len(outcomes), count, 256):
        result = evaluate_policy(
            candidate, teacher, device, 256, seed + offset // 2, simulations=simulations
        )
        outcomes.extend(result.pop("outcomes"))
        chunks.append(result)
        assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())
        report["active"] = {
            "label": label,
            "completed_games": len(outcomes),
            "summary": summarize(outcomes),
            "chunks": chunks,
        }
        report["elapsed_seconds"] = elapsed_before + time.monotonic() - start
        atomic_json(output, report)
        print(label, len(outcomes), report["elapsed_seconds"], flush=True)
    report["evaluations"].append(
        {
            "label": label,
            "opponent_checkpoint": checkpoint,
            "opponent_simulations": simulations,
            **summarize(outcomes),
            "chunks": chunks,
        }
    )
    report.pop("active", None)
    atomic_json(output, report)
report.pop("active", None)
report["status"] = "complete"
report["reference_parameters_and_buffers_unchanged"] = True
report["elapsed_seconds"] = elapsed_before + time.monotonic() - start
atomic_json(output, report)
