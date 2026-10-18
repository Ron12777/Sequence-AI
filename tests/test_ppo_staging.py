"""Fixed-data PPO equivalence, controlling CUDA nondeterministic reductions.

CUDA checks run in a fresh subprocess so CUBLAS_WORKSPACE_CONFIG is effective
before the first CUDA context/BLAS handle. A baseline-versus-baseline control
must pass before interpreting the optimized comparison; tolerances remain zero.
"""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.train_exact_policy import RECIPE, optimizer_for, rollout, update
from src.exact_policy import ExactPolicy
from src.history_policy import HistoryPolicy
from src.model import SequenceNet


def original_update():
    path = Path(__file__).parent / "fixtures" / "ppo_update_before_staging.py"
    spec = importlib.util.spec_from_file_location("ppo_before_staging", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.update


def assert_identical(left, right, path="root"):
    if isinstance(left, torch.Tensor):
        delta = (left.detach().double() - right.detach().double()).abs()
        assert torch.equal(left, right), (
            f"{path}: max_absolute_error={float(delta.max()) if delta.numel() else 0}"
        )
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_identical(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            assert_identical(a, b, f"{path}[{i}]")
    else:
        assert left == right, path


def max_weight_error(left, right):
    return max(
        float((left[key].double() - right[key].double()).abs().max()) for key in left
    )


def run_comparison(device, kl_limit):
    previous_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.set_num_threads(1)
    if device == "cuda":
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.manual_seed(891)
        net = (
            ExactPolicy(
                HistoryPolicy(
                    SequenceNet(num_filters=8, num_res_blocks=1), "board", width=8
                ),
                card_width=8,
            )
            .to(device)
            .eval()
        )
        data, _ = rollout(net, device, 4, 1267)
        data_snapshot = {key: value.clone() for key, value in data.items()}
        baseline, control, staged = (copy.deepcopy(net) for _ in range(3))
        recipe = {**RECIPE, "epochs": 2, "batch_size": 32, "target_kl": kl_limit}
        optimizers = [
            optimizer_for(actor, recipe) for actor in (baseline, control, staged)
        ]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else []
        results = []
        for actor, optimizer, perform in zip(
            (baseline, control, staged),
            optimizers,
            (original_update(), original_update(), update),
        ):
            torch.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
            metrics = perform(actor, optimizer, data, device, recipe)
            results.append(
                (
                    metrics,
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state_all() if device == "cuda" else [],
                )
            )
        print(
            json.dumps(
                {
                    "device": device,
                    "kl_limit": kl_limit,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "baseline_repeat_max_weight_error": max_weight_error(
                        baseline.state_dict(), control.state_dict()
                    ),
                    "staging_max_weight_error": max_weight_error(
                        baseline.state_dict(), staged.state_dict()
                    ),
                }
            ),
            flush=True,
        )
        for label, actor, optimizer, result in zip(
            ("baseline_repeat", "staged"),
            (control, staged),
            optimizers[1:],
            results[1:],
        ):
            assert_identical(
                baseline.state_dict(), actor.state_dict(), label + ".weights"
            )
            assert_identical(
                optimizers[0].state_dict(), optimizer.state_dict(), label + ".optimizer"
            )
            assert_identical(results[0][1:], result[1:], label + ".rng")
            for key in results[0][0]:
                if key != "update_seconds":
                    assert result[0][key] == results[0][0][key], f"{label}.{key}"
        assert_identical(data, data_snapshot, "unmodified_data")
        actual = results[-1][0]
        assert all(value is None or np.isfinite(value) for value in actual.values())
        assert actual["kl_early_stop"] == (kl_limit == 1e-7)
        assert actual["updates"] > 0
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("kl_limit", [0.01, 1e-7])
def test_staging_preserves_losses_weights_optimizer_and_rng(device, kl_limit):
    if device == "cpu":
        run_comparison(device, kl_limit)
        return
    path = str(Path(__file__).resolve())
    code = f"import runpy; runpy.run_path({path!r})['run_comparison']('cuda', {kl_limit!r})"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
