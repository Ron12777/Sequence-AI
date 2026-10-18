"""R-NaD recurrences, legal-logit forces, regularization and continuation."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.rnad import RNaDConfig, RNaDState, legal_policy, nerd_loss, rnad_loss, v_trace


def numpy_reference(v, valid, players, behavior, policy, reg, actions, reward, player, cfg):
    """Independent scalar translation of OpenSpiel 0af322f's backward scan."""
    t_count, batch = valid.shape
    values = np.zeros_like(v)
    qs = np.zeros_like(policy)
    for b in range(batch):
        carry = {"reward": 0.0, "uncorrected": 0.0, "value": 0.0, "target": 0.0, "importance": 1.0}
        for t in reversed(range(t_count)):
            if not valid[t, b]:
                carry = {"reward": 0.0, "uncorrected": 0.0, "value": 0.0, "target": 0.0, "importance": 1.0}
                continue
            a = actions[t, b]
            mu = behavior[t, b, a]
            ratio = policy[t, b, a] / mu
            sign = 1 if players[t, b] == player else -1
            entropy = -cfg.eta * np.sum(policy[t, b] * reg[t, b]) * sign
            uncorrected = reward[t, b] + carry["uncorrected"] + entropy
            discounted = reward[t, b] + carry["reward"]
            if players[t, b] == player:
                importance = ratio * carry["importance"]
                target = (
                    v[t, b]
                    + min(cfg.rho, importance) * (uncorrected + carry["value"] - v[t, b])
                    + cfg.lambda_ * min(cfg.c, importance) * (carry["target"] - carry["value"])
                )
                q = v[t, b] - cfg.eta * reg[t, b] * sign
                q = q.copy()
                q[a] += (discounted + carry["importance"] * carry["target"] - v[t, b]) / mu
                values[t, b] = target
                qs[t, b] = q
                carry = {"reward": 0.0, "uncorrected": 0.0, "value": v[t, b], "target": target, "importance": 1.0}
            else:
                carry["reward"] = entropy + ratio * discounted
                carry["uncorrected"] = uncorrected
                carry["importance"] *= ratio
    return values, qs


def test_vtrace_matches_independent_scalar_reference_with_offpolicy_and_padding():
    rng = np.random.default_rng(14)
    t_count, batch, a_count = 7, 3, 4
    valid = np.arange(t_count)[:, None] < np.array([7, 4, 6])[None]
    players = rng.integers(0, 2, (t_count, batch))
    logits = torch.tensor(rng.normal(size=(t_count, batch, a_count)), dtype=torch.float64)
    policy = logits.softmax(-1).numpy()
    behavior = torch.tensor(rng.normal(size=policy.shape)).softmax(-1).numpy()
    reg = np.log(policy) - np.log(torch.tensor(rng.normal(size=policy.shape)).softmax(-1).numpy())
    actions = rng.integers(0, a_count, (t_count, batch))
    values = rng.normal(size=valid.shape)
    rewards = rng.normal(size=valid.shape)
    cfg = RNaDConfig(eta=0.15, c=0.8, rho=3.0, lambda_=0.7)
    for player in (0, 1):
        expected_v, expected_q = numpy_reference(
            values, valid, players, behavior, policy, reg, actions, rewards, player, cfg
        )
        actual_v, mask, actual_q = v_trace(
            *map(torch.as_tensor, (values, valid, players, behavior, policy, reg, actions, rewards)),
            player, cfg,
        )
        np.testing.assert_allclose(actual_v.numpy(), expected_v, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_q.numpy(), expected_q, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(mask, valid & (players == player))


def test_on_policy_complete_episode_eta_zero_has_exact_monte_carlo_value_targets():
    values = torch.tensor([[0.3], [-0.2], [0.6], [-0.5]])
    valid = torch.ones(4, 1, dtype=torch.bool)
    players = torch.tensor([[0], [1], [1], [0]])
    pi = torch.full((4, 1, 2), 0.5)
    actions = torch.zeros(4, 1, dtype=torch.long)
    rewards = torch.tensor([[0.0], [0.0], [0.0], [1.0]])
    for player in (0, 1):
        target, mask, _ = v_trace(
            values, valid, players, pi, pi, pi * 0, actions,
            rewards * (1 if player == 0 else -1), player, RNaDConfig(eta=0),
        )
        torch.testing.assert_close(target[mask], torch.full_like(target[mask], 1 if player == 0 else -1))


def test_opponent_regularization_reward_has_opposite_sign():
    values = torch.zeros(2, 1)
    valid = torch.ones(2, 1, dtype=torch.bool)
    players = torch.tensor([[0], [1]])
    pi = torch.full((2, 1, 2), 0.5)
    # Constant log ratio makes each actor's KL=1. Own cost -eta,
    # opponent's cost is our +eta: cancellation at player zero's root.
    reg = torch.ones_like(pi)
    vt0, _, _ = v_trace(values, valid, players, pi, pi, reg, torch.zeros_like(players), values, 0)
    vt1, _, _ = v_trace(values, valid, players, pi, pi, reg, torch.zeros_like(players), values, 1)
    assert vt0[0, 0].item() == pytest.approx(0)
    assert vt1[1, 0].item() == pytest.approx(-0.2)


def test_nerd_correct_legal_mean_ignores_arbitrary_padding_logits_and_constant_shift():
    legal = torch.tensor([[[True, True, False, False]]])
    policy = torch.tensor([[[0.5, 0.5, 0.0, 0.0]]])
    q = torch.tensor([[[1.0, -1.0, 1000.0, -9999.0]]])
    grads = []
    for data in ([1.0, -1.0, 1e9, -1e9], [501.0, 499.0, -77.0, 999.0]):
        logits = torch.tensor([[data]], requires_grad=True)
        loss = nerd_loss(logits, policy, q, legal, torch.ones(1, 1))
        loss.backward()
        grads.append(logits.grad)
    torch.testing.assert_close(grads[0], grads[1])
    torch.testing.assert_close(grads[0], torch.tensor([[[-1.0, 1.0, 0.0, 0.0]]]))


def test_nerd_threshold_prevents_outward_force_but_allows_recovery():
    logits = torch.tensor([[[3.0, -3.0]]], requires_grad=True)
    legal, mask = torch.ones_like(logits, dtype=torch.bool), torch.ones(1, 1)
    policy = torch.full_like(logits, 0.5)
    loss = nerd_loss(logits, policy, torch.tensor([[[1.0, -1.0]]]), legal, mask)
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    logits.grad = None
    nerd_loss(logits, policy, torch.tensor([[[-1.0, 1.0]]]), legal, mask).backward()
    assert logits.grad[0, 0, 0] > 0
    assert logits.grad[0, 0, 1] < 0


def test_full_loss_finite_with_empty_padding_and_detached_targets():
    torch.manual_seed(7)
    logits = torch.randn(4, 2, 3, requires_grad=True)
    values = torch.randn(4, 2, requires_grad=True)
    target = torch.randn(4, 2, requires_grad=True)
    valid = torch.tensor([[1, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.bool)
    legal = valid[..., None].expand_as(logits)
    policy, logpi = legal_policy(logits.detach(), legal)
    player = torch.tensor([[0, 0], [1, 1], [0, 0], [0, 0]])
    reward = torch.zeros(4, 2, 2)
    reward[2, 0] = torch.tensor([1.0, -1.0])
    reward[1, 1] = torch.tensor([-1.0, 1.0])
    loss, metrics = rnad_loss(
        logits, values, target, policy, logpi, logpi, torch.zeros_like(player),
        legal, valid, player, reward, 0.5,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert torch.isfinite(logits.grad).all() and torch.isfinite(values.grad).all()
    assert target.grad is None
    assert torch.equal(logits.grad[~valid], torch.zeros_like(logits.grad[~valid]))


def test_sampled_action_with_zero_behavior_probability_rejected():
    with pytest.raises(ValueError, match="positive behavior"):
        v_trace(
            torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long),
            torch.tensor([[[0.0, 1.0]]]), torch.tensor([[[0.5, 0.5]]]), torch.zeros(1, 1, 2),
            torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1), 0,
        )


def test_reference_rotation_ema_and_resume_are_exact():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    state = RNaDState(model, RNaDConfig(target_avg=0.5, reference_interval=4))
    assert state.alpha == 0
    alphas, rotations = [], []
    for i in range(3):
        alphas.append(state.alpha)
        with torch.no_grad():
            model.weight.fill_(i + 1)
        rotations.append(state.after_update(model))
    assert alphas == [0.0, 0.5, 1.0] and rotations == [False] * 3
    resumed = RNaDState(model)
    resumed.load_state_dict(copy.deepcopy(state.state_dict()))
    with torch.no_grad():
        model.weight.fill_(4)
    assert state.after_update(model) and resumed.after_update(model)
    assert state.alpha == 0 and state.learner_steps == 4
    assert state.previous_reference.weight.item() == 0
    assert state.reference.weight.item() == pytest.approx(3.0625)
    for name in ("target", "reference", "previous_reference"):
        torch.testing.assert_close(getattr(state, name).weight, getattr(resumed, name).weight)
        assert not getattr(state, name).weight.requires_grad


def test_numerical_equivalence_to_unmodified_official_jax_functions():
    """Golden arrays come from executing OpenSpiel source, not this port.

    Regenerate with experiments/2026-10-17/rnad_reference_validation.py.
    The fixture includes variable legal masks, finite IS clipping, repeated
    turns, terminal padding and NeuRD forces crossing both thresholds.
    """
    with np.load(Path(__file__).parent / "fixtures/rnad_open_spiel_0af322f.npz") as data:
        tensors = {key: torch.from_numpy(value.copy()) for key, value in data.items()}
    eta, c, rho, lambda_, alpha = tensors["config"].tolist()
    cfg = RNaDConfig(eta=eta, c=c, rho=rho, lambda_=lambda_, nerd_clip=1.7)
    for player in (0, 1):
        value, mask, q = v_trace(
            tensors["target_values"], tensors["valid"], tensors["player_ids"],
            tensors["behavior"], tensors["policy"], tensors["regularizer"],
            tensors["actions"], tensors["rewards"][..., player], player, cfg,
        )
        torch.testing.assert_close(value, tensors["target_v"][player].squeeze(-1), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(mask, tensors["masks"][player].bool())
        torch.testing.assert_close(q, tensors["target_q"][player], atol=1e-10, rtol=1e-10)
    logits = tensors["logits"].requires_grad_()
    values = tensors["values"].requires_grad_()
    loss, metrics = rnad_loss(
        logits, values, tensors["target_values"], tensors["behavior"],
        tensors["reference"], tensors["previous_reference"], tensors["actions"],
        tensors["legal"], tensors["valid"], tensors["player_ids"], tensors["rewards"], alpha, cfg,
    )
    loss.backward()
    torch.testing.assert_close(metrics["nerd_loss"], tensors["nerd_value"], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(metrics["critic_loss"], tensors["critic_value"], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(logits.grad, tensors["nerd_grad"], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(values.grad, tensors["critic_grad"].squeeze(-1), atol=1e-10, rtol=1e-10)
