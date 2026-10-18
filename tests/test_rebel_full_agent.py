"""Full-game agent isolation: private queries cannot alter public search."""

import random

import numpy as np
import pytest

from src import rebel_full_agent
from src.game import SequenceGame
from src.public_belief_cfr import LeafValues
from src.rebel_full_agent import ReBeLAgent


class ValueStub:
    def evaluate_leaf(self, adapter, node, belief):
        return LeafValues(
            np.zeros(len(adapter.hands[0])), np.zeros(len(adapter.hands[1]))
        )

    def evaluate_query(self, adapter, action, belief, actual_hand):
        # Exercise the actual rules and card compatibility through the adapter.
        state, counts = adapter.query_after_action(action, actual_hand)
        assert not state.hands[1] and not state.hands[2] and not state.deck.cards
        assert counts.sum() == len(actual_hand) - 1
        return action[0] / 52 + action[1] / 1000 + action[2] / 10000


def game(seed=71):
    random.seed(seed)
    result = SequenceGame()
    result.reset()
    return result


def make_agent(monkeypatch, model=None, **kwargs):
    model = ValueStub() if model is None else model
    monkeypatch.setattr(
        rebel_full_agent, "load_value", lambda checkpoint, device: model
    )
    return ReBeLAgent(
        "test-only-value-stub", particles=4, iterations=3, seed=25, **kwargs
    )


def test_full_game_act_is_legal_and_retains_coupled_search_training_data(monkeypatch):
    state = game()
    agent = make_agent(monkeypatch)
    agent.reset(seed=123, player=1)
    move = agent.act(state, 1, [])
    assert move in state.get_legal_moves()
    search = agent.last_search
    assert (
        search["selected_iteration"].iteration
        == agent.last_metrics["selected_iteration"]
    )
    assert search["root_player"] == 1
    action, belief, hand = search["last_query"]
    assert action == rebel_full_agent._action_key(move)
    assert hand == tuple(state.hands[1])
    assert belief.mass > 0
    assert agent.last_metrics["cfr_iterations"] == 3
    for mean in search["mean_root_conditional_values"]:
        assert np.isfinite(mean).all()
    assert not agent.last_metrics["equilibrium_guarantee"]


def test_actual_own_hand_changes_do_not_change_public_bank_or_searched_policy(
    monkeypatch,
):
    original = game()
    changed = original.clone()
    changed.hands[1][0], changed.deck.cards[-1] = (
        changed.deck.cards[-1],
        changed.hands[1][0],
    )
    first, second = make_agent(monkeypatch), make_agent(monkeypatch)
    first.act(original, 1, [])
    second.act(changed, 1, [])
    assert (
        first.last_metrics["public_search_fingerprint"]
        == second.last_metrics["public_search_fingerprint"]
    )
    a, b = first.last_search["adapter"], second.last_search["adapter"]
    np.testing.assert_array_equal(a.prior, b.prior)
    for x, y in zip(a.hands, b.hands):
        np.testing.assert_array_equal(x, y)


def test_opponent_hand_and_future_deck_changes_cannot_change_action(monkeypatch):
    original = game()
    changed = original.clone()
    changed.hands[2][0], changed.deck.cards[-1] = (
        changed.deck.cards[-1],
        changed.hands[2][0],
    )
    changed.deck.cards.reverse()
    first, second = make_agent(monkeypatch), make_agent(monkeypatch)
    assert first.act(original, 1, []) == second.act(changed, 1, [])
    assert (
        first.last_metrics["public_search_fingerprint"]
        == second.last_metrics["public_search_fingerprint"]
    )


def test_hidden_collections_need_only_public_length_not_identity_access(monkeypatch):
    class LengthOnly(list):
        def __iter__(self):
            raise AssertionError("Unobserved identities were read")

        def __getitem__(self, index):
            raise AssertionError("Unobserved identities were read")

    state = game()
    state.hands[2] = LengthOnly(state.hands[2])
    state.deck.cards = LengthOnly(state.deck.cards)
    agent = make_agent(monkeypatch)
    move = agent.act(state, 1, [])
    assert move in state.get_legal_moves()


def test_public_query_fallback_does_not_insert_actual_hand(monkeypatch):
    state = game()
    agent = make_agent(monkeypatch)
    agent.act(state, 1, [])
    search = agent.last_search
    adapter, solver, selected = (
        search["adapter"],
        search["solver"],
        search["selected_iteration"],
    )
    before = [x.copy() for x in adapter.hands]
    # Pick a physical identity absent from the small public acting bank.
    card = int(np.flatnonzero(adapter.hands[0].sum(axis=0) == 0)[0])
    action = (card, 0, 1, False)
    belief, kind = agent._query_belief(adapter, solver, selected, action)
    assert kind == "unsupported_action"
    assert belief is selected.evaluation.beliefs[adapter.root.key]
    for old, new in zip(before, adapter.hands):
        np.testing.assert_array_equal(old, new)


def test_private_query_cannot_mutate_public_particle_bank(monkeypatch):
    class MutatingValue(ValueStub):
        def evaluate_query(self, adapter, action, belief, actual_hand):
            adapter.prior[:] = 0
            return 0.0

    agent = make_agent(monkeypatch, MutatingValue())
    with pytest.raises(RuntimeError, match="mutated the public"):
        agent.act(game(), 1, [])


def test_nonfinite_query_rejected_and_wrong_player_rejected(monkeypatch):
    class BadValue(ValueStub):
        def evaluate_query(self, adapter, action, belief, actual_hand):
            return np.nan

    agent = make_agent(monkeypatch, BadValue())
    with pytest.raises(ValueError, match="nonfinite"):
        agent.act(game(), 1, [])
    with pytest.raises(ValueError, match="current player"):
        agent.act(game(), 2, [])
    agent.reset(player=2)
    with pytest.raises(ValueError, match="does not match"):
        agent.act(game(), 1, [])


def test_no_legal_moves_returns_none_without_search(monkeypatch):
    state = game()
    state.game_over = True
    agent = make_agent(monkeypatch)
    assert agent.act(state, 1, []) is None
    assert agent.last_search is None and agent.last_query is None


def test_depth2_query_integrates_frozen_opponent_policy_and_conditional_weights(
    monkeypatch,
):
    from types import SimpleNamespace

    from src.public_belief_cfr import PublicBeliefCFR, PublicNode

    action, first, second = (0, 0, 1, False), (1, 1, 1, False), (2, 1, 2, False)
    root = PublicNode(
        "root",
        actor=0,
        children={
            action: PublicNode(
                "opponent",
                actor=1,
                children={
                    first: PublicNode("first"),
                    second: PublicNode("second"),
                },
            ),
        },
    )
    policies = {"root": np.ones((1, 1)), "opponent": np.eye(2)}
    solver = PublicBeliefCFR(root, [[0.5, 0.5]], lambda node, belief: 0.0)
    selected = SimpleNamespace(
        strategies=policies, evaluation=solver.evaluate(policies)
    )
    adapter = SimpleNamespace(root=root)
    calls = []

    class PathValue(ValueStub):
        def evaluate_query(self, *args):
            raise AssertionError("Depth2 query incorrectly used immediate leaf")

        def evaluate_query_path(self, adapter, actions, belief, hand, opponent_weights):
            calls.append((actions, opponent_weights.copy(), belief))
            return -1.0 if actions[-1] == first else 1.0

    agent = make_agent(monkeypatch, PathValue(), depth=2)
    belief = selected.evaluation.beliefs["opponent"]
    value, metrics = agent._evaluate_private(
        adapter,
        selected,
        action,
        belief,
        (),
        np.array([0.25, 0.75]),
    )
    assert value == pytest.approx(0.5)
    assert metrics["depth2_queries"] == 1
    assert metrics["depth2_response_evaluations"] == 2
    np.testing.assert_array_equal(calls[0][1], [1, 0])
    np.testing.assert_array_equal(calls[1][1], [0, 1])
    assert calls[0][2] is selected.evaluation.beliefs["first"]
    assert calls[1][2] is selected.evaluation.beliefs["second"]


def test_depth2_rejects_value_wrapper_without_continuation_support(monkeypatch):
    with pytest.raises(ValueError, match="evaluate_query_path"):
        make_agent(monkeypatch, depth=2)
