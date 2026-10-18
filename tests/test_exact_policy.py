"""Exact action identity, warm start, information boundaries, and set invariance."""

import random

import numpy as np
import pytest
import torch

from src.cards import Card
from src.exact_policy import (
    ACTION_COUNT,
    ExactPolicy,
    action_id,
    batch_inputs,
    legal_actions,
    load_exact_policy,
    observe_exact,
    save_exact_policy,
)
from src.game import SequenceGame
from src.history_policy import HistoryPolicy
from src.model import SequenceNet


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def game():
    random.seed(217)
    g = SequenceGame()
    g.reset()
    g.hands[1] = [
        Card.from_string(s) for s in ("AS", "JC", "AS", "5D", "9H", "JH", "2S")
    ]
    g.board[4, 4] = 2
    return g


def teacher(architecture="board"):
    torch.manual_seed(11)
    return HistoryPolicy(
        SequenceNet(num_filters=8, num_res_blocks=1), architecture, width=8
    ).eval()


def inputs(g):
    observation = observe_exact(g, [])
    _, ids = legal_actions(g)
    return (*batch_inputs([observation], "cpu"), torch.tensor(ids)[None])


@pytest.mark.parametrize("variant", ["cnn", "transformer"])
@pytest.mark.parametrize("architecture", ["board", "gru"])
def test_initial_exact_policy_preserves_teacher_position_marginals_and_value(
    variant, architecture
):
    g, old = game(), teacher(architecture)
    model = ExactPolicy.from_history_policy(
        old, card_width=8, variant=variant, transformer_width=16, transformer_layers=1
    ).eval()
    board, context, history, actions = inputs(g)
    with torch.no_grad():
        old_logpi, old_value = old(board, context, history)
        logits, value = model(board, context, history, actions)
    positions = actions % 100
    legal_position = torch.zeros(1, 100, dtype=torch.bool).scatter_(1, positions, True)
    expected = old_logpi.masked_fill(~legal_position, -1e9).softmax(1)
    actual = torch.zeros_like(expected).scatter_add_(1, positions, logits.softmax(1))
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(value, old_value, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("variant", ["cnn", "transformer"])
def test_permuted_actual_hand_has_identical_observations_and_predictions_after_learning(
    variant,
):
    g = game()
    changed = g.clone()
    changed.hands[1] = [g.hands[1][i] for i in [3, 6, 0, 2, 5, 1, 4]]
    a, b = observe_exact(g, []), observe_exact(changed, [])
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    model = ExactPolicy(
        teacher(),
        card_width=8,
        variant=variant,
        transformer_width=16,
        transformer_layers=1,
    ).eval()
    with torch.no_grad():
        model.spatial_projection.weight.normal_()
        model.set_value.weight.normal_()
        if variant == "transformer":
            model.transformer_position.weight.normal_()
            model.transformer_spatial.weight.normal_()
            model.transformer_value.weight.normal_()
        pa, va = model(*inputs(g))
        pb, vb = model(*inputs(changed))
    torch.testing.assert_close(pa, pb, atol=0, rtol=0)
    torch.testing.assert_close(va, vb, atol=0, rtol=0)


def test_observations_do_not_read_opponent_cards_or_deck_identity():
    class LengthOnly(list):
        def __iter__(self):
            raise AssertionError("hidden identity read")

        def __getitem__(self, key):
            raise AssertionError("hidden identity read")

    g = game()
    expected = observe_exact(g, [])
    g.hands[2] = LengthOnly(g.hands[2])
    g.deck.cards = LengthOnly(g.deck.cards)
    actual = observe_exact(g, [])
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)


def test_card_and_square_actions_are_distinct_duplicate_physical_cards_are_deduplicated():
    g = game()
    moves, ids = legal_actions(g)
    assert len(ids) == len(set(ids))
    assert ids.tolist() == sorted(ids.tolist())
    square = next((m.row, m.col) for m in moves if str(m.card) == "AS")
    choices = [m for m in moves if (m.row, m.col) == square]
    assert {str(m.card) for m in choices} == {"AS", "JC"}
    assert len({action_id(m) for m in choices}) == 2
    removal = [m for m in moves if m.is_removal]
    assert removal and all(str(m.card) == "JH" for m in removal)
    assert all(0 <= action_id(m) < ACTION_COUNT for m in moves)


def test_padded_legal_only_logits_equal_dense_masked_logits():
    g = game()
    model = ExactPolicy(teacher(), card_width=8).eval()
    board, context, history, actions = inputs(g)
    dense_mask = torch.zeros(1, ACTION_COUNT, dtype=torch.bool).scatter_(
        1, actions, True
    )
    padded = torch.cat((actions, torch.full((1, 3), -1)), dim=1)
    mask = padded >= 0
    with torch.no_grad():
        dense, _ = model(board, context, history, action_mask=dense_mask)
        listed, _ = model(board, context, history, padded, mask)
    torch.testing.assert_close(
        listed[:, :-3], dense.gather(1, actions), atol=1e-7, rtol=1e-6
    )
    assert (listed[:, -3:] == -1e9).all()
    assert torch.all(listed.softmax(1)[:, -3:] == 0)


@pytest.mark.parametrize("variant", ["cnn", "transformer"])
def test_exact_card_head_can_learn_to_distinguish_same_square_cards(variant):
    g = game()
    model = ExactPolicy(
        teacher(),
        card_width=8,
        variant=variant,
        transformer_width=16,
        transformer_layers=1,
    ).eval()
    board, context, history, actions = inputs(g)
    moves, _ = legal_actions(g)
    normal = next(i for i, m in enumerate(moves) if str(m.card) == "AS")
    square = actions[0, normal] % 100
    wild = next(
        i
        for i, m in enumerate(moves)
        if str(m.card) == "JC" and action_id(m) % 100 == square
    )
    optimizer = torch.optim.SGD([model.card_bias], lr=0.5)
    before, _ = model(board, context, history, actions)
    assert before[0, normal].item() == pytest.approx(before[0, wild].item())
    loss = -before.log_softmax(1)[0, normal]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after, _ = model(board, context, history, actions)
    assert after[0, normal] > after[0, wild]
    assert model.card_bias.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", ["cnn", "transformer"])
def test_checkpoint_roundtrip_restores_exact_model_and_training_state(
    tmp_path, variant
):
    model = ExactPolicy(
        teacher(),
        card_width=8,
        variant=variant,
        transformer_width=16,
        transformer_layers=1,
    ).eval()
    target = tmp_path / "nested" / "latest.pt"
    save_exact_policy(
        model, target, {"games": 12}, training_state={"optimizer": {"step": 3}}
    )
    loaded, metadata = load_exact_policy(target)
    assert loaded.variant == variant and metadata == {"games": 12}
    with torch.no_grad():
        a = model(*inputs(game()))
        b = loaded(*inputs(game()))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    payload = torch.load(target, weights_only=True)
    assert payload["training_state"]["optimizer"]["step"] == 3
    assert not list(target.parent.glob("*.tmp"))


def test_unbounded_rnad_critic_preserves_start_but_can_fit_regularized_returns(tmp_path):
    old = teacher()
    common = ExactPolicy(old, card_width=8).eval()
    with torch.no_grad():
        common.set_value.weight.normal_()
        common.set_value.bias.fill_(0.3)
    model = ExactPolicy(old, card_width=8, value_mode="unbounded").eval()
    incompatible = model.load_state_dict(common.state_dict(), strict=False)
    assert incompatible.missing_keys == ["rnad_value.weight", "rnad_value.bias"]
    assert incompatible.unexpected_keys == []
    board, context, history, actions = inputs(game())
    with torch.no_grad():
        expected = common(board, context, history, actions)[1]
        actual = model(board, context, history, actions)[1]
    torch.testing.assert_close(actual, expected)
    with torch.no_grad():
        model.rnad_value.bias.fill_(3)
        actual = model(board, context, history, actions)[1]
    assert actual.min() > 1
    path = tmp_path / "rnad.pt"
    save_exact_policy(model, path, {})
    loaded, _ = load_exact_policy(path)
    assert loaded.value_mode == "unbounded"
    torch.testing.assert_close(loaded(board, context, history, actions)[1], actual)
