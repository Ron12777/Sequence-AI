"""The isolated scripted scan must retain production R-NaD and official targets."""

import importlib.util
from pathlib import Path

import pytest
import torch

import src.rnad as production
from scripts.rnad_vtrace_candidate import v_trace


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference_tests():
    path = Path(__file__).parent / "test_rnad.py"
    spec = importlib.util.spec_from_file_location("rnad_scan_reference_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    [
        "test_vtrace_matches_independent_scalar_reference_with_offpolicy_and_padding",
        "test_on_policy_complete_episode_eta_zero_has_exact_monte_carlo_value_targets",
        "test_opponent_regularization_reward_has_opposite_sign",
        "test_full_loss_finite_with_empty_padding_and_detached_targets",
        "test_sampled_action_with_zero_behavior_probability_rejected",
        "test_numerical_equivalence_to_unmodified_official_jax_functions",
    ],
)
def test_scripted_candidate_passes_existing_independent_references(monkeypatch, name):
    references = reference_tests()
    monkeypatch.setattr(references, "v_trace", v_trace)
    monkeypatch.setattr(production, "v_trace", v_trace)
    getattr(references, name)()


@pytest.mark.parametrize(
    "length,batch,actions", [(1, 1, 1), (17, 3, 11), (64, 32, 202)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scan_is_bit_identical_for_both_players_and_detaches(
    length, batch, actions, dtype
):
    torch.manual_seed(591)
    values = torch.randn(length, batch, dtype=dtype, requires_grad=True)
    valid = torch.arange(length)[:, None] <= torch.randint(0, length, (batch,))[None]
    players = torch.randint(0, 2, (length, batch))
    behavior = torch.randn(length, batch, actions, dtype=dtype).softmax(-1)
    policy = torch.randn_like(behavior).softmax(-1)
    ratios = torch.randn_like(behavior)
    chosen = torch.randint(0, actions, (length, batch))
    rewards = torch.randn_like(values)
    config = production.RNaDConfig(c=0.8, rho=3, lambda_=0.7)
    for player in (0, 1):
        arguments = (
            values,
            valid,
            players,
            behavior,
            policy,
            ratios,
            chosen,
            rewards,
            player,
            config,
        )
        expected = production.v_trace(*arguments)
        actual = v_trace(*arguments)
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)
            assert not a.requires_grad
