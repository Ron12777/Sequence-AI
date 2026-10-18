import itertools

import numpy as np
import pytest

from src.public_belief_cfr import LeafValues, PublicBeliefCFR, PublicNode


def kuhn():
    """One-card Kuhn poker: six equiprobable ordered deals, one bet maximum."""
    show = np.sign(np.arange(3)[:, None] - np.arange(3)[None, :])

    def leaf(key, payoff):
        return PublicNode(key, payoff=payoff)

    root = PublicNode("root", actor=0, children={
        "check": PublicNode("check", actor=1, children={
            "check": leaf("check/check", show),
            "bet": PublicNode("check/bet", actor=0, children={
                "fold": leaf("check/bet/fold", -1),
                "call": leaf("check/bet/call", 2 * show),
            }),
        }),
        "bet": PublicNode("bet", actor=1, children={
            "fold": leaf("bet/fold", 1),
            "call": leaf("bet/call", 2 * show),
        }),
    })
    return PublicBeliefCFR(root, np.ones((3, 3)) - np.eye(3))


def test_kuhn_cfr_converges_to_known_value_and_low_exploitability():
    solver = kuhn()
    result = solver.solve(4000)
    assert result.value == pytest.approx(-1 / 18, abs=0.004)
    assert solver.exploitability(result.strategies) < 0.012
    assert sum(result.evaluation.player_values) == pytest.approx(0, abs=1e-12)
    assert result.last.iteration == 4000
    assert all(p.shape == (3, 2) for p in result.strategies.values())


@pytest.mark.parametrize("player", [0, 1])
def test_exact_best_response_agrees_with_exhaustive_infoset_policies(player):
    solver = kuhn()
    rng = np.random.default_rng(123)
    policy = {key: rng.dirichlet([1, 1], size=3) for key in solver.masks}
    reaches = (np.array([0.2, 0.8, 1.0]), np.array([1.0, 0.1, 0.6]))
    keys = [key for key in policy if solver.nodes[key].actor == player]
    expected = -np.inf
    for choices in itertools.product((0, 1), repeat=3 * len(keys)):
        changed = dict(policy)
        for k, key in enumerate(keys):
            changed[key] = np.eye(2)[list(choices[3 * k:3 * k + 3])]
        expected = max(expected, solver.evaluate(changed, reaches).player_values[player])
    actual, response = solver.best_response(policy, player, reaches)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert solver.evaluate(policy | response, reaches).player_values[player] == pytest.approx(expected)


def test_matching_pennies_payoff_with_chance_hidden_bit_has_no_clairvoyance():
    # Player 0's bit is drawn privately by chance. Player 1 sees no signal.
    # This is NOT simultaneous strategic matching pennies: root actions are public.
    root = PublicNode("guess", actor=1, children={
        0: PublicNode("zero", payoff=np.array([[-1], [1]])),
        1: PublicNode("one", payoff=np.array([[1], [-1]])),
    })
    solver = PublicBeliefCFR(root, [[0.5], [0.5]])
    result = solver.solve(20)
    assert result.value == pytest.approx(0)
    assert solver.best_response(result.strategies, 1)[0] == pytest.approx(0)
    assert result.strategies["guess"].shape == (1, 2)
    # Per-deal maximization would incorrectly claim value +1 for the guesser.
    assert solver.exploitability(result.strategies) == pytest.approx(0)


def test_correlated_private_prior_signal_and_bayes_action_update():
    root = PublicNode("guess", actor=1, children={
        0: PublicNode("zero", payoff=np.array([[-1, -1], [1, 1]])),
        1: PublicNode("one", payoff=np.array([[1, 1], [-1, -1]])),
    })
    solver = PublicBeliefCFR(root, [[0.45, 0.05], [0.05, 0.45]])
    result = solver.solve(100)
    assert result.value == pytest.approx(-0.8, abs=0.01)
    assert solver.best_response(result.strategies, 1)[0] == pytest.approx(0.8)
    updated = solver.posterior_after_action(root, np.eye(2), 0)
    np.testing.assert_allclose(updated.joint, [[0.9, 0], [0.1, 0]])
    np.testing.assert_allclose(updated.ranges[0], [0.9, 0.1])
    assert updated.mass == pytest.approx(0.5)


def test_private_type_illegal_actions_never_receive_mass_or_best_response():
    root = PublicNode("root", actor=0, legal_mask=np.array([[1, 0], [0, 1]]), children={
        "a": PublicNode("a", payoff=np.array([[1, 1], [100, 100]])),
        "b": PublicNode("b", payoff=np.array([[100, 100], [3, 3]])),
    })
    solver = PublicBeliefCFR(root, np.ones((2, 2)))
    result = solver.solve(10)
    np.testing.assert_array_equal(result.strategies["root"], np.eye(2))
    assert result.value == pytest.approx(2)
    assert solver.best_response(result.strategies, 0)[0] == pytest.approx(2)
    with pytest.raises(ValueError, match="Invalid information-set"):
        solver.evaluate({"root": np.ones((2, 2)) / 2})


def test_inactive_unreachable_type_has_zero_policy_and_cf_value():
    root = PublicNode("root", actor=0, legal_mask=np.array([[1, 0], [0, 0]]), children={
        "a": PublicNode("a", payoff=1),
        "b": PublicNode("b", payoff=100),
    })
    solver = PublicBeliefCFR(root, [[0.5], [0.5]])
    reaches = (np.array([1.0, 0]), np.array([1.0]))
    result = solver.solve(5, reaches)
    np.testing.assert_array_equal(result.strategies["root"], [[1, 0], [0, 0]])
    np.testing.assert_array_equal(result.counterfactual_values[0], [0.5, 0])
    assert result.value == pytest.approx(1)
    value, policy = solver.best_response(result.strategies, 0, reaches)
    assert value == pytest.approx(1)
    np.testing.assert_array_equal(policy["root"], [[1, 0], [0, 0]])
    with pytest.raises(ValueError, match="no legal action"):
        solver.solve(1)


def test_conditional_leaf_callback_matches_payoff_matrix_with_correlated_reaches():
    prior = np.array([[0.4, 0.1], [0.2, 0.3]])
    payoff = np.array([[1, -2], [3, 0.5]])
    reaches = (np.array([0, 0.7]), np.array([0.8, 0.1]))
    exact = PublicBeliefCFR(PublicNode("leaf", payoff=payoff), prior)
    seen = []

    def network(node, belief):
        assert node.key == "leaf"
        seen.append(belief)
        cf0 = (prior * payoff) @ belief.reaches[1]
        cf1 = -(prior * payoff).T @ belief.reaches[0]
        return LeafValues(cf0 / belief.compatible_mass[0], cf1 / belief.compatible_mass[1])

    learned = PublicBeliefCFR(PublicNode("leaf"), prior, network)
    a = exact.evaluate({}, reaches)
    b = learned.evaluate({}, reaches)
    for x, y in zip(a.counterfactual_values, b.counterfactual_values):
        np.testing.assert_allclose(x, y)
    assert a.value == pytest.approx(b.value)
    assert seen[0].ranges[0][0] == 0  # Zero own reach still permits a CF hand target.
    assert b.counterfactual_values[0][0] != 0
    with pytest.raises(ValueError, match="full fixed-payoff"):
        learned.exploitability({})


def test_subtree_root_reaches_and_normalization_match_bayes_conditioning():
    root = PublicNode("root", actor=0, children={
        0: PublicNode("zero", payoff=np.array([[1, 0], [0, 2]])),
        1: PublicNode("one", payoff=-1),
    })
    prior = np.array([[0.4, 0.1], [0.2, 0.3]])
    solver = PublicBeliefCFR(root, prior)
    policy = {"root": np.array([[0.6, 0.4], [0.1, 0.9]])}
    belief = solver.posterior_after_action(root, policy["root"], 0)
    continuation = PublicBeliefCFR(root.children[0], prior).evaluate({}, belief.reaches)
    posterior_game = PublicBeliefCFR(root.children[0], belief.joint).evaluate({})
    assert continuation.value == pytest.approx(posterior_game.value)
    np.testing.assert_allclose(continuation.conditional_values, posterior_game.conditional_values)
    assert continuation.value == pytest.approx(np.sum(belief.joint * root.children[0].payoff))


def test_iteration_policy_belief_values_are_coupled_and_no_private_identity_input():
    solver = kuhn()
    iterations = list(solver.iterations(20))
    for step in iterations:
        replay = solver.evaluate(step.strategies)
        assert replay.value == pytest.approx(step.evaluation.value)
        for key in solver.nodes:
            np.testing.assert_allclose(replay.beliefs[key].joint, step.evaluation.beliefs[key].joint)
        # Public betting updates posterior using per-own-hand likelihood only.
        expected = solver.posterior_after_action(solver.root, step.strategies["root"], "bet")
        np.testing.assert_allclose(expected.joint, step.evaluation.beliefs["bet"].joint)
    result = solver.solve(20)
    expected = np.mean([x.evaluation.counterfactual_values[0] for x in iterations], axis=0)
    np.testing.assert_allclose(result.mean_node_counterfactual_values["root"][0], expected)


def test_unreachable_public_branch_has_zero_posterior_but_finite_counterfactuals():
    solver = kuhn()
    policy = {key: np.tile([1.0, 0.0], (3, 1)) for key in solver.masks}
    value = solver.evaluate(policy)
    assert value.beliefs["bet"].mass == 0
    assert np.all(value.beliefs["bet"].joint == 0)
    for vectors in value.node_values.values():
        assert all(np.isfinite(x).all() for x in vectors)


def test_rejects_duplicate_public_histories_and_invalid_prior():
    duplicate = PublicNode("x", payoff=0)
    with pytest.raises(ValueError, match="unique"):
        PublicBeliefCFR(PublicNode("root", actor=0, children={0: duplicate, 1: duplicate}), [[1]])
    for prior in ([[0]], [[-1]], [[np.nan]], [1, 2]):
        with pytest.raises(ValueError):
            PublicBeliefCFR(PublicNode("leaf", payoff=0), prior)


def test_optional_batched_leaf_callback_matches_sequential_cfr_and_preserves_terminals():
    root = PublicNode("root", actor=0, children={
        "a": PublicNode("a", actor=1, children={
            "a0": PublicNode("a0"), "a1": PublicNode("a1", payoff=1),
        }),
        "b": PublicNode("b"),
    })
    prior = np.array([[0.4, 0.1], [0.2, 0.3]])

    def sequential(node, belief):
        # Depend on reaches and public range so batching stale beliefs fails.
        v0 = np.array([0.2, -0.6]) + 0.1 * belief.reaches[1].sum()
        v1 = np.array([-0.1, 0.4]) - 0.2 * belief.ranges[0][0]
        return LeafValues(v0, v1)

    class Batched:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args):
            raise AssertionError("Sequential inference should not run")

        def evaluate_many(self, leaves):
            self.calls += 1
            assert {node.key for node, _ in leaves} == {"a0", "b"}
            return {node.key: sequential(node, belief) for node, belief in leaves}

    callback = Batched()
    first = PublicBeliefCFR(root, prior, sequential).solve(30)
    second = PublicBeliefCFR(root, prior, callback).solve(30)
    assert callback.calls == 31  # One per iteration plus average-policy evaluation.
    assert first.value == pytest.approx(second.value, abs=1e-14)
    for key in first.strategies:
        np.testing.assert_allclose(first.strategies[key], second.strategies[key], rtol=0, atol=0)
    for key in first.evaluation.node_values:
        for a, b in zip(first.evaluation.node_values[key], second.evaluation.node_values[key]):
            np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_batch_callback_rejects_missing_or_extra_leaf_keys():
    class Broken:
        def __call__(self, *args):
            return 0

        def evaluate_many(self, leaves):
            return {"wrong": 0}

    solver = PublicBeliefCFR(PublicNode("leaf"), [[1]], Broken())
    with pytest.raises(ValueError, match="exactly the requested"):
        solver.solve(1)
