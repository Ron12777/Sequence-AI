"""Isolated same-model two-GPU feasibility benchmark; never saves checkpoints.

The raw ExactPolicy owns every parameter, optimizer slot and checkpoint key.
Only its forward is parallelized. One gathered global loss produces one update.
Run only within an explicitly coordinated exclusive lease on BOTH GPUs.
"""

import argparse
import copy
import hashlib
import json
import signal
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from scripts import train_exact_policy as trainer
from src.exact_policy import load_exact_policy
from src.history_policy import load_policy


class ForwardAdapter:
    def __init__(self, raw, parallel, threshold=1024):
        self.raw, self.parallel, self.threshold = raw, parallel, threshold
        self.parallel_calls = 0
        self.single_calls = 0

    def __call__(self, *inputs):
        if inputs[0].shape[0] >= self.threshold:
            self.parallel_calls += 1
            return self.parallel(*inputs)
        self.single_calls += 1
        return self.raw(*inputs)


class ForwardProxy:
    """The update uses delegated raw parameters/mode with a separate forward."""

    def __init__(self, raw, forward):
        self.raw, self.forward = raw, forward

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __call__(self, *inputs):
        return self.forward(*inputs)


class CpuSplitForward:
    """Numerical integration check only; this is not a CPU performance claim."""

    def __init__(self, raw):
        self.raw = raw

    def __call__(self, *inputs):
        cut = (inputs[0].shape[0] + 1) // 2
        groups = [self.raw(*(item[:cut] for item in inputs))]
        if cut < inputs[0].shape[0]:
            groups.append(self.raw(*(item[cut:] for item in inputs)))
        return tuple(torch.cat([group[i] for group in groups]) for i in range(2))


def adapted_rollout(raw, forward):
    """Copy the function globals, without monkeypatching the production module."""

    original = trainer.actor_forward

    def actor_forward(model, obs, device):
        if model is not raw:
            return original(model, obs, device)
        board, context, history, actions, mask = trainer.tensors(obs, device)
        logits, values = forward(board, context, history, actions, mask)
        return logits.masked_fill(~mask, -torch.inf), values.flatten(), actions, mask

    return types.FunctionType(
        trainer.rollout.__code__,
        {**trainer.rollout.__globals__, "actor_forward": actor_forward},
        trainer.rollout.__name__,
        trainer.rollout.__defaults__,
        trainer.rollout.__closure__,
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=1024)
    parser.add_argument("--max-seconds", type=int, default=80)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("single", "update_dp", "all_dp"),
        default=["single", "update_dp"],
    )
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        parser.error("An exclusive lease on two visible CUDA GPUs is required")
    if args.games < 4 or args.repeats < 1 or not 10 <= args.max_seconds <= 90:
        parser.error("Invalid bounded benchmark arguments")
    # Hard process limit guarantees that no candidate can occupy the lease past
    # its agreed deadline. Partial completed comparisons are saved after each run.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(args.max_seconds)
    started = time.monotonic()
    torch.set_num_threads(1)
    source_hash = digest(args.checkpoint)
    reference_hash = digest(args.reference)
    initial, metadata = load_exact_policy(args.checkpoint, "cpu")
    if initial.variant != "cnn" or metadata.get("algorithm", "ppo") != "ppo":
        raise ValueError("This benchmark is scoped to the same CNN PPO learner")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    recipe = metadata["recipe"]
    optimizer_state = payload.get("training_state", {}).get("optimizer")
    reference, _ = load_policy(args.reference, "cuda:0")
    historical = copy.deepcopy(initial).to("cuda:0").eval()
    rows = []
    report = {
        "checkpoint_sha256": source_hash,
        "reference_sha256": reference_hash,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
        "recipe": recipe,
        "games_per_cycle": args.games,
        "threshold": args.threshold,
        "global_batch_unchanged": recipe["batch_size"],
        "optimizer_restored": optimizer_state is not None,
        "parameter_count": sum(p.numel() for p in initial.parameters()),
        "precision": torch.get_float32_matmul_precision(),
        "rows": rows,
        "checkpoints_saved": False,
        "production_source_modified": False,
        "historical_reference_scope": "Saved learner clone used as fixed historical exact opponent in every variant; frozen GRU reference unchanged",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        report["checkpoint_unchanged"] = digest(args.checkpoint) == source_hash
        report["reference_unchanged"] = digest(args.reference) == reference_hash
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def sync():
        for device in (0, 1):
            torch.cuda.synchronize(device)

    for repeat in range(args.repeats):
        baseline_weights = None
        baseline_metrics = None
        for variant in args.variants:
            if rows and time.monotonic() - started > args.max_seconds - 15:
                report["stopped_at_time_budget"] = True
                save()
                signal.alarm(0)
                return
            raw = copy.deepcopy(initial).to("cuda:0").eval()
            optimizer = trainer.optimizer_for(raw, recipe)
            if optimizer_state is not None:
                optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            parallel = torch.nn.DataParallel(raw, device_ids=[0, 1], output_device=0)
            adapter = ForwardAdapter(raw, parallel, args.threshold)
            proxy = ForwardProxy(raw, adapter)
            # Warm both full and split batch shapes without updating parameters.
            env = trainer.native_environment(1024, 3900040011)
            warm = trainer.tensors(
                trainer.observations(env, env.active_ids()), "cuda:0"
            )
            with torch.no_grad():
                baseline_output = raw(*warm)
                split_output = parallel(*warm)
            errors = [
                float((a - b).abs().max())
                for a, b in zip(baseline_output, split_output)
            ]
            if not all(
                torch.allclose(a, b, rtol=2e-5, atol=2e-5)
                for a, b in zip(baseline_output, split_output)
            ):
                raise AssertionError(f"Parallel forward mismatch: {errors}")
            del env, warm, baseline_output, split_output
            torch.manual_seed(2100001931)
            torch.cuda.manual_seed_all(2100001931)
            sync()
            tick = time.perf_counter()
            run_rollout = (
                adapted_rollout(raw, adapter)
                if variant == "all_dp"
                else trainer.rollout
            )
            data, rollout_metrics = run_rollout(
                raw, "cuda:0", args.games, 3900030001, opponents=(reference, historical)
            )
            sync()
            rollout_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            update_metrics = trainer.update(
                proxy if variant != "single" else raw, optimizer, data, "cuda:0", recipe
            )
            sync()
            update_seconds = time.perf_counter() - tick
            row = {
                "repeat": repeat,
                "variant": variant,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "cycle_seconds": rollout_seconds + update_seconds,
                "rollout": rollout_metrics,
                "update": update_metrics,
                "parallel_calls": adapter.parallel_calls,
                "single_calls": adapter.single_calls,
                "fixed_initial_forward_max_abs_error": errors,
                "finite_parameters": all(
                    bool(torch.isfinite(p).all()) for p in raw.parameters()
                ),
            }
            current_weights = {
                name: value.detach().cpu().clone()
                for name, value in raw.state_dict().items()
            }
            if variant == "single":
                baseline_weights, baseline_metrics = current_weights, update_metrics
            elif baseline_weights is not None:
                row["max_parameter_error_vs_single"] = max(
                    float(
                        (value.double() - baseline_weights[name].double()).abs().max()
                    )
                    for name, value in current_weights.items()
                )
                row["same_update_count_as_single"] = (
                    update_metrics["updates"] == baseline_metrics["updates"]
                )
                row["same_kl_stop_as_single"] = (
                    update_metrics["kl_early_stop"] == baseline_metrics["kl_early_stop"]
                )
                row["metric_absolute_errors_vs_single"] = {
                    name: abs(float(value) - float(baseline_metrics[name]))
                    for name, value in update_metrics.items()
                    if name != "update_seconds"
                    and value is not None
                    and baseline_metrics[name] is not None
                }
            rows.append(row)
            save()
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "repeat",
                            "variant",
                            "rollout_seconds",
                            "update_seconds",
                            "cycle_seconds",
                            "parallel_calls",
                        )
                    }
                ),
                flush=True,
            )
            del raw, optimizer, parallel, adapter, proxy, data
    report["completed"] = True
    save()
    signal.alarm(0)


if __name__ == "__main__":
    main()
