"""Frozen diagnostic checks; no learning updates are performed."""

import numpy as np
import torch

from scripts.diagnose_rnad_policy import centered_logits, diagnose, force_metrics
from src.exact_policy import ExactPolicy, save_exact_policy
from src.history_policy import HistoryPolicy
from src.model import SequenceNet
from src.rnad import RNaDState


def test_sharp_warm_start_crosses_beta_two_and_force_gates_are_asymmetric():
    probabilities = torch.tensor([[0.8] + [0.2 / 19] * 19])
    logits = probabilities.log()
    legal = torch.ones_like(logits, dtype=torch.bool)
    centered = centered_logits(logits, legal)
    assert 4.1 < centered[0, 0] < 4.2
    logits = torch.tensor([[3.0, -3.0]])
    legal = torch.ones_like(logits, dtype=torch.bool)
    policy = logits.softmax(-1)
    positive = force_metrics(logits, policy, torch.tensor([[1.0, -1.0]]), legal, 10000)
    assert positive["thresholds"]["2.0"]["positive_force_abs_blocked_fraction"] == 1
    assert positive["thresholds"]["2.0"]["negative_force_abs_blocked_fraction"] == 1
    assert positive["thresholds"]["2.0"]["mean_log_top_probability_sgd_proxy"] == 0
    assert positive["thresholds"]["4.0"]["mean_log_top_probability_sgd_proxy"] > 0
    negative = force_metrics(logits, policy, torch.tensor([[-1.0, 1.0]]), legal, 10000)
    assert negative["thresholds"]["2.0"]["positive_force_abs_blocked_fraction"] == 0
    assert negative["thresholds"]["2.0"]["negative_force_abs_blocked_fraction"] == 0
    assert negative["thresholds"]["2.0"]["mean_log_top_probability_sgd_proxy"] < 0


def test_complete_cpu_diagnostic_uses_frozen_reference_state_without_optimizer(tmp_path):
    torch.manual_seed(171)
    base = HistoryPolicy(SequenceNet(num_filters=8, num_res_blocks=1), "board", width=8)
    common = ExactPolicy(base, card_width=8).eval()
    current = ExactPolicy(base, card_width=8, value_mode="unbounded").eval()
    current.load_state_dict(common.state_dict(), strict=False)
    state = RNaDState(current)
    common_path, current_path = tmp_path / "common.pt", tmp_path / "rnad.pt"
    save_exact_policy(common, common_path, {})
    save_exact_policy(current, current_path, {}, training_state={"rnad": state.state_dict()})
    before = (common_path.read_bytes(), current_path.read_bytes())
    result = diagnose(common_path, current_path, tmp_path / "diagnostic.json", games=4, seed=1768)
    assert result["device"] == "cpu" and result["optimizer_steps"] == 0
    assert result["optimizer_constructed"] is False
    assert result["parameters_and_buffers_unchanged"]
    assert result["neurd"]["status"] == "complete"
    np.testing.assert_allclose(result["neurd"]["selected_current_over_behavior"]["mean"], 1, atol=1e-6)
    assert result["warm_start_drift"]["mean_kl_common_to_rnad"] == 0
    assert result["neurd"]["force_gating"]["thresholds"].keys() == {"2.0", "4.0", "8.0"}
    assert np.isfinite(result["neurd"]["critic_target_mse_per_state"])
    assert before == (common_path.read_bytes(), current_path.read_bytes())
