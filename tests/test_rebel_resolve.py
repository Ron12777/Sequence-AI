"""Finite-tree checks for coupled recursive search and empirical evaluation."""

import numpy as np
import pytest

from scripts import rebel_resolve
from scripts.rebel_resolve import resolve_mixture, resolve_profile, truncate
from src.public_belief_cfr import LeafValues, PublicBeliefCFR, PublicNode


def tiny_game():
    # Two public stages, private correlated types, and different type preferences.
    return PublicNode(
        "root",
        actor=0,
        children={
            "left": PublicNode(
                "left",
                actor=1,
                children={
                    "x": PublicNode("left/x", payoff=np.array([[1, -1], [-1, 1]])),
                    "y": PublicNode("left/y", payoff=np.array([[-1, 1], [1, -1]])),
                },
            ),
            "right": PublicNode(
                "right",
                actor=1,
                children={
                    "x": PublicNode("right/x", payoff=np.array([[0.5, -1], [1, -0.5]])),
                    "y": PublicNode("right/y", payoff=np.array([[-0.5, 1], [-1, 0.5]])),
                },
            ),
        },
    )


PRIOR = np.array([[0.4, 0.1], [0.1, 0.4]])


def test_truncate_preserves_public_structure_and_isolates_arrays():
    root = tiny_game()
    root.legal_mask = np.array([[1, 1], [1, 0]], dtype=bool)
    root.data = {"public": "root"}
    cut = truncate(root, 1)
    assert cut.key == root.key and cut.data == root.data
    assert list(cut.children) == ["left", "right"]
    assert all(n.actor is None and n.payoff is None for n in cut.children.values())
    cut.legal_mask[0, 0] = False
    assert root.legal_mask[0, 0]
    terminal = root.children["left"].children["x"]
    copied = truncate(terminal, 0)
    copied.payoff[0, 0] = 100
    assert terminal.payoff[0, 0] == 1
    assert truncate(root, 0).actor is None
    assert truncate(root, 3).children["left"].children["x"].payoff is not None
    with pytest.raises(ValueError):
        truncate(root, -1)


def test_sampled_full_tree_policy_matches_exact_selected_iteration():
    root = tiny_game()
    seed, count = 21, 12
    selected = int(np.random.default_rng(seed).integers(1, count + 1))
    expected = list(PublicBeliefCFR(root, PRIOR).iterations(selected))[-1]
    actual = resolve_profile(root, PRIOR, None, depth=3, iterations=count, seed=seed)
    assert actual.metrics["sampled_iterations"] == [selected]
    assert actual.metrics["cfr_iterations"] == selected
    assert actual.metrics["leaf_evaluations"] == 0
    for key, policy in expected.strategies.items():
        np.testing.assert_array_equal(actual.strategies[key], policy)
    full_budget = resolve_profile(
        root,
        PRIOR,
        None,
        depth=3,
        iterations=count,
        seed=seed,
        run_full_iterations=True,
    )
    assert full_budget.metrics["cfr_iterations"] == count
    for key in actual.strategies:
        np.testing.assert_array_equal(
            actual.strategies[key], full_budget.strategies[key]
        )


def test_recursive_resolving_couples_cutoff_reaches_and_fills_unreachable_subgames():
    root = tiny_game()
    # This leaf evaluator makes root choose left after the first CFR update.
    seen = []

    def leaf(node, belief):
        seen.append((node.key, belief.reaches))
        return 1.0 if node.key == "left" else -1.0

    result = resolve_profile(root, PRIOR, leaf, depth=1, iterations=10, seed=3)
    np.testing.assert_array_equal(result.strategies["root"], [[1, 0], [1, 0]])
    assert result.metrics["search_calls"] == 2  # Root and reachable left continuation.
    assert result.metrics["zero_mass_fallback_subgames"] == 1
    assert result.metrics["fallback_information_sets"] == 1
    np.testing.assert_array_equal(result.strategies["right"], [[0.5, 0.5], [0.5, 0.5]])
    assert set(result.strategies) == {"root", "left", "right"}
    full = PublicBeliefCFR(root, PRIOR)
    result_value = full.evaluate(result.strategies)
    assert result_value.beliefs["right"].mass == 0
    assert np.isfinite(full.exploitability(result.strategies))
    assert seen and result.metrics["leaf_evaluations"] == len(seen)


def test_zero_own_reach_types_and_inactive_mask_rows_are_finite():
    # Left is possible only for type0. Type1's row there is structurally inactive.
    root = PublicNode(
        "root",
        actor=0,
        legal_mask=np.eye(2, dtype=bool),
        children={
            "left": PublicNode(
                "left",
                actor=0,
                legal_mask=np.array([[1, 1], [0, 0]], bool),
                children={
                    "x": PublicNode("left/x", payoff=1),
                    "y": PublicNode("left/y", payoff=-1),
                },
            ),
            "right": PublicNode(
                "right",
                actor=1,
                children={
                    "x": PublicNode("right/x", payoff=1),
                    "y": PublicNode("right/y", payoff=-1),
                },
            ),
        },
    )
    seen = []

    def leaf(node, belief):
        seen.append(belief)
        return LeafValues(np.zeros(2), np.zeros(2))

    result = resolve_profile(root, PRIOR, leaf, depth=1, iterations=4, seed=1)
    assert any(np.any(b.reaches[0] == 0) for b in seen)
    np.testing.assert_array_equal(result.strategies["left"][1], [0, 0])
    full = PublicBeliefCFR(root, PRIOR)
    assert np.isfinite(full.evaluate(result.strategies).value)
    assert np.isfinite(full.exploitability(result.strategies))
    zero = resolve_profile(root, PRIOR, leaf, root_reach=(np.zeros(2), np.ones(2)))
    assert zero.metrics["search_calls"] == 0
    assert zero.metrics["zero_mass_fallback_subgames"] == 1
    assert all(np.isfinite(p).all() for p in zero.strategies.values())


def test_mixture_uses_own_reach_not_arithmetic_policy_average(monkeypatch):
    root = PublicNode(
        "root",
        actor=0,
        children={
            "a": PublicNode(
                "a",
                actor=0,
                children={
                    "x": PublicNode("x", payoff=1),
                    "y": PublicNode("y", payoff=-1),
                },
            ),
            "b": PublicNode("b", payoff=0),
        },
    )
    policies = iter(
        [
            {"root": np.array([[1.0, 0.0]]), "a": np.array([[1.0, 0.0]])},
            {"root": np.array([[0.0, 1.0]]), "a": np.array([[0.0, 1.0]])},
        ]
    )

    def fake_resolve(*args, **kwargs):
        metrics = {
            name: 0
            for name in (
                "search_calls",
                "leaf_evaluations",
                "leaf_seconds",
                "cfr_iterations",
                "zero_mass_fallback_subgames",
                "fallback_information_sets",
            )
        }
        return rebel_resolve.ResolvedProfile(next(policies), metrics)

    monkeypatch.setattr(rebel_resolve, "resolve_profile", fake_resolve)
    result = resolve_mixture(root, [[1]], None, repeats=2)
    np.testing.assert_array_equal(result.strategies["root"], [[0.5, 0.5]])
    np.testing.assert_array_equal(result.strategies["a"], [[1, 0]])
    assert result.value == pytest.approx(0.5)
    assert result.exploitability == pytest.approx(0.25)
    assert not result.metrics["stochastic_resolver_exploitability_bound"]


def test_mixture_is_deterministic_complete_and_reports_exact_finite_profile_response():
    root = tiny_game()
    a = resolve_mixture(root, PRIOR, None, depth=1, iterations=8, repeats=4, seed=25)
    b = resolve_mixture(root, PRIOR, None, depth=1, iterations=8, repeats=4, seed=25)
    full = PublicBeliefCFR(root, PRIOR)
    assert set(a.strategies) == set(full.masks)
    for key in a.strategies:
        np.testing.assert_array_equal(a.strategies[key], b.strategies[key])
    assert a.value == b.value
    assert a.exploitability == full.exploitability(a.strategies)
    assert a.metrics["search_calls"] >= 4
    assert a.metrics["wall_seconds"] > 0
    assert len(a.metrics["profiles"]) == 4
    assert not a.metrics["safe_resolving_guarantee"]
    with pytest.raises(ValueError, match="positive root"):
        resolve_mixture(root, PRIOR, None, root_reach=(np.zeros(2), np.ones(2)))


@pytest.mark.parametrize(
    "options", [{"depth": 0}, {"iterations": 0}, {"iterations": True}]
)
def test_invalid_search_budgets_rejected(options):
    with pytest.raises(ValueError):
        resolve_profile(tiny_game(), PRIOR, None, **options)
