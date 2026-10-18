"""Bounded A/B of isolated scripted scan, preserving the production source.

Run CPU by default. A CUDA run must be coordinated with the sole training job.
The fixed complete-game rollout is reused for all A/B updates; no weights save.
"""

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from scripts.rnad_vtrace_candidate import v_trace as scripted_v_trace
from scripts.train_exact_policy import RECIPE, optimizer_for, rollout, update_rnad
from src import rnad
from src.exact_policy import ExactPolicy, load_exact_policy
from src.history_policy import HistoryPolicy
from src.model import SequenceNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initial")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 1 or args.repeats < 1:
        parser.error("Positive game and repeat counts required")
    torch.set_num_threads(1)
    torch.manual_seed(841)
    device = torch.device(args.device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if args.initial:
        initial, _ = load_exact_policy(args.initial, "cpu")
        model = ExactPolicy.from_history_policy(
            initial.backbone, variant="cnn", value_mode="unbounded"
        )
        model.load_state_dict(initial.state_dict(), strict=False)
    else:
        model = ExactPolicy(
            HistoryPolicy(
                SequenceNet(num_filters=8, num_res_blocks=1), "board", width=8
            ),
            value_mode="unbounded",
        )
    model = model.to(device).eval()
    data, rollout_metrics = rollout(
        model, device, args.games, 3900000117, full_trajectory=True
    )
    original = rnad.v_trace
    # Warm the scripted interpreter on representative sequence dimensions.
    length, batch, actions = 64, 32, 202
    values = torch.randn(length, batch, device=device)
    valid = (
        torch.arange(length, device=device)[:, None]
        < torch.randint(1, length + 1, (batch,), device=device)[None]
    )
    players = torch.randint(0, 2, (length, batch), device=device)
    policy = torch.randn(length, batch, actions, device=device).softmax(-1)
    behavior = torch.randn_like(policy).softmax(-1)
    ratios = torch.randn_like(policy)
    chosen = torch.randint(actions, (length, batch), device=device)
    scan_args = (values, valid, players, behavior, policy, ratios, chosen, values)
    for _ in range(3):
        for player in (0, 1):
            scripted_v_trace(*scan_args, player)
    sync()
    scan = {}
    for label, implementation in (
        ("baseline", original),
        ("scripted", scripted_v_trace),
    ):
        timings = []
        for _ in range(10):
            sync()
            tick = time.perf_counter()
            for player in (0, 1):
                implementation(*scan_args, player)
            sync()
            timings.append(time.perf_counter() - tick)
        scan[label] = {
            "median_seconds_two_players": statistics.median(timings),
            "seconds": timings,
        }
    rows = []
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    try:
        for repeat in range(args.repeats):
            weights = {}
            metrics = {}
            for label, implementation in (
                ("baseline", original),
                ("scripted", scripted_v_trace),
            ):
                actor = copy.deepcopy(model)
                optimizer = optimizer_for(actor, RECIPE, "rnad")
                state = rnad.RNaDState(
                    actor,
                    rnad.RNaDConfig(
                        reference_interval=RECIPE["rnad_reference_interval"]
                    ),
                )
                torch.set_rng_state(cpu_rng)
                if cuda_rng:
                    torch.cuda.set_rng_state_all(cuda_rng)
                rnad.v_trace = implementation
                sync()
                tick = time.perf_counter()
                result = update_rnad(actor, optimizer, data, state, device)
                sync()
                seconds = time.perf_counter() - tick
                metrics[label] = result
                weights[label] = {
                    key: value.detach().cpu().clone()
                    for key, value in actor.state_dict().items()
                }
                rows.append(
                    {
                        "repeat": repeat,
                        "variant": label,
                        "seconds": seconds,
                        "metrics": result,
                    }
                )
                del actor, optimizer, state
            rows[-1]["max_weight_difference_vs_baseline"] = max(
                float((weights["baseline"][key].double() - value.double()).abs().max())
                for key, value in weights["scripted"].items()
            )
            rows[-1]["max_metric_difference_vs_baseline"] = max(
                abs(float(value) - float(metrics["baseline"][key]))
                for key, value in metrics["scripted"].items()
                if key != "update_seconds"
            )
    finally:
        rnad.v_trace = original
    medians = {
        label: statistics.median(
            row["seconds"] for row in rows if row["variant"] == label
        )
        for label in ("baseline", "scripted")
    }
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "games": args.games,
        "rollout": rollout_metrics,
        "scan": scan,
        "updates": rows,
        "update_median_seconds": medians,
        "update_speedup": medians["baseline"] / medians["scripted"],
        "production_source_modified": False,
        "scope": "fixed-data execution throughput, not model strength; CUDA nondeterminism is not disabled",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "scan": {k: v["median_seconds_two_players"] for k, v in scan.items()},
                "update_median_seconds": medians,
                "speedup": report["update_speedup"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
