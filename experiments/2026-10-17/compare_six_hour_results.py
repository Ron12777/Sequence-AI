"""Compare completed six-hour runs on identical reference deals."""

import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    reports = {
        name: json.loads((root / f"{name}-six-hour-confirmation.json").read_text())
        for name in ["board", "gru"]
    }
    assert all(
        r["status"] == "complete" and r["parameters_and_buffers_unchanged"]
        for r in reports.values()
    )
    differences = []
    for label in ["original_champion_search", "heldout_baseline"]:
        specs = [
            next(o for o in r["plan"]["opponents"] if o["label"] == label)
            for r in reports.values()
        ]
        assert specs[0] == specs[1]
        outcomes = {
            name: np.array(
                next(e for e in r["evaluations"] if e["label"] == label)["outcomes"]
            )
            for name, r in reports.items()
        }
        delta = (outcomes["board"] - outcomes["gru"]).reshape(-1, 2).mean(1)
        half = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
        differences.append(
            {
                "opponent": label,
                "games": len(delta) * 2,
                "cnn_minus_gru_score": float(delta.mean()),
                "paired_95ci": [float(delta.mean() - half), float(delta.mean() + half)],
            }
        )
    result = {
        "architectures": {
            name: {
                "checkpoint": r["plan"]["candidate"],
                "training_seconds": r["candidate_metadata"]["training_seconds"],
                "allocated_gpu_hours": r["candidate_metadata"]["training_seconds"]
                / 3600,
                "allocated_training_cost_usd": r["candidate_metadata"][
                    "training_seconds"
                ]
                / 3600
                * 0.4681481481481482,
                "evaluations": [
                    {k: v for k, v in e.items() if k not in ["outcomes", "chunks"]}
                    for e in r["evaluations"]
                ],
            }
            for name, r in reports.items()
        },
        "paired_reference_differences": differences,
        "caveats": [
            "Exclusive allocated RTX4090 time includes CPU-bound self-play; not active GPU kernel hours.",
            "Incremental training cost excludes shared pretraining, tuning, evaluation and rental overhead.",
            "One training seed per architecture; no inference of asymptotic ceilings.",
            "Same architectures as four-hour timing comparison; latency was not remeasured here.",
        ],
    }
    (root / "six-hour-cost-comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
