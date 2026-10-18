"""Matched-input CPU latency and paired four-hour architecture comparison."""

import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts.architecture_experiments import direct_moves
from src.fast_game import FastSequenceGame
from src.history_policy import apply_public_move, batch_inputs, load_policy, observe


def main():
    torch.set_num_threads(2)
    models, metadata = {}, {}
    for name in ["board", "gru"]:
        path = ROOT / f"artifacts/checkpoints/{name}-four-hour.pt"
        models[name], metadata[name] = load_policy(path, "cpu")
        models[name].eval()
    games, histories = [], []
    for i in range(128):
        random.seed(781009 + i)
        g, events = FastSequenceGame(), []
        for _ in range(i % 48):
            legal = g.get_legal_moves()
            if g.game_over or not legal:
                break
            apply_public_move(g, events, random.choice(legal))
        if g.game_over or not g.get_legal_moves():
            g, events = FastSequenceGame(), []
        games.append(g)
        histories.append(events)
    inputs = {
        n: batch_inputs(
            [observe(g, e) for g, e in zip(games[:n], histories[:n])], "cpu"
        )
        for n in [1, 32, 128]
    }
    timings = []
    with torch.inference_mode():
        for n, x in inputs.items():
            for model in models.values():
                for _ in range(5):
                    model(*x)
            samples = {name: [] for name in models}
            # Alternate model order to reduce temperature/order bias.
            for rep in range(40):
                for name in ["board", "gru"] if rep % 2 else ["gru", "board"]:
                    start = time.perf_counter()
                    models[name](*x)
                    samples[name].append((time.perf_counter() - start) * 1000)
            for name, values in samples.items():
                timings.append(
                    {
                        "architecture": name,
                        "batch": n,
                        "scope": "forward_only",
                        "median_batch_ms": statistics.median(values),
                        "median_ms_per_position": statistics.median(values) / n,
                        "p95_batch_ms": float(np.percentile(values, 95)),
                        "samples": len(values),
                    }
                )
            for name, model in models.items():
                values = []
                for _ in range(15):
                    start = time.perf_counter()
                    direct_moves(
                        model, torch.device("cpu"), games, histories, list(range(n))
                    )
                    values.append((time.perf_counter() - start) * 1000)
                timings.append(
                    {
                        "architecture": name,
                        "batch": n,
                        "scope": "observe_forward_legal_argmax",
                        "median_batch_ms": statistics.median(values),
                        "median_ms_per_position": statistics.median(values) / n,
                        "p95_batch_ms": float(np.percentile(values, 95)),
                        "samples": len(values),
                    }
                )
    records = {
        name: json.loads(
            (
                ROOT / f"experiments/2026-10-17/{name}-four-hour-confirmation.json"
            ).read_text()
        )
        for name in models
    }
    assert all(
        r["status"] == "complete" and r["parameters_and_buffers_unchanged"]
        for r in records.values()
    )
    differences = []
    for label in ["original_champion_search", "heldout_baseline"]:
        rows = {
            name: next(e for e in r["evaluations"] if e["label"] == label)
            for name, r in records.items()
        }
        specs = {
            name: next(e for e in r["plan"]["opponents"] if e["label"] == label)
            for name, r in records.items()
        }
        assert specs["board"] == specs["gru"]
        diff = (
            (np.array(rows["board"]["outcomes"]) - np.array(rows["gru"]["outcomes"]))
            .reshape(-1, 2)
            .mean(1)
        )
        half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
        differences.append(
            {
                "opponent": label,
                "games": len(diff) * 2,
                "cnn_minus_gru_score": float(diff.mean()),
                "paired_95ci": [float(diff.mean() - half), float(diff.mean() + half)],
            }
        )
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    result = {
        "device": "cpu",
        "cpu": cpu,
        "torch": torch.__version__,
        "threads": 2,
        "timings": timings,
        "paired_reference_differences": differences,
        "architectures": {
            name: {
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_seconds": metadata[name]["training_seconds"],
                "allocated_training_gpu_hours": metadata[name]["training_seconds"]
                / 3600,
                "allocated_training_cost_usd": metadata[name]["training_seconds"]
                / 3600
                * 0.4681481481481482,
                "checkpoint_sha256": hashlib.sha256(
                    (ROOT / f"artifacts/checkpoints/{name}-four-hour.pt").read_bytes()
                ).hexdigest(),
            }
            for name, model in models.items()
        },
        "caveats": [
            "CPU PyTorch timings; not browser ONNX or GPU timings.",
            "Allocated training hours include self-play and CPU rollout while owning the GPU; not active kernel hours.",
            "Incremental training cost excludes common pretraining, tuning, evaluations and other rental overhead.",
            "One training seed per architecture and a shared pretrained CNN; does not establish asymptotic architecture ceilings.",
        ],
    }
    path = ROOT / "experiments/2026-10-17/four-hour-cost-comparison.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
