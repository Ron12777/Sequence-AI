"""Sampled recursive public-belief resolving for small finite test games.

This helper samples one CFR iteration per subgame and passes that iteration's
strategy AND reach factors to the next subgame. It does not condition on a
sampled private deal. Zero-probability subgames use an explicit uniform legal
fallback. Learned leaves and that fallback preclude a safety guarantee here.

Mixture evaluation computes exact best responses to the finite empirical
mixture of complete policies, not to the underlying stochastic resolver.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.public_belief_cfr import LeafEvaluator, PublicBeliefCFR, PublicNode


def _positive_integer(value, name, *, allow_zero=False):
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")


def truncate(root: PublicNode, depth: int) -> PublicNode:
    """Copy a public tree, replacing nonterminal depth-boundary nodes by leaves.

    Keys and PUBLIC data are preserved. Payoffs and masks are copied, so the
    resulting solver cannot mutate the full tree through either array.
    """
    _positive_integer(depth, "depth", allow_zero=True)
    if root.actor is None:
        payoff = None if root.payoff is None else np.array(root.payoff, copy=True)
        return PublicNode(root.key, payoff=payoff, data=root.data)
    if depth == 0:
        return PublicNode(root.key, data=root.data)
    mask = None if root.legal_mask is None else np.array(root.legal_mask, copy=True)
    return PublicNode(
        root.key,
        actor=root.actor,
        children={
            action: truncate(child, depth - 1)
            for action, child in root.children.items()
        },
        legal_mask=mask,
        data=root.data,
    )


def _uniform(mask):
    count = mask.sum(axis=1, keepdims=True)
    return np.divide(mask, count, out=np.zeros(mask.shape), where=count > 0)


def _root_reaches(shape, root_reach):
    if root_reach is None:
        return tuple(np.ones(n) for n in shape)
    if len(root_reach) != 2:
        raise ValueError("root_reach must contain two vectors")
    reaches = tuple(np.array(x, dtype=np.float64, copy=True) for x in root_reach)
    for reach, n in zip(reaches, shape):
        if reach.shape != (n,) or not np.isfinite(reach).all() or np.any(reach < 0):
            raise ValueError("root_reach must contain finite nonnegative type vectors")
    return reaches


@dataclass
class ResolvedProfile:
    strategies: dict
    metrics: dict


@dataclass
class ResolvedMixture:
    strategies: dict
    metrics: dict
    value: float
    exploitability: float


def resolve_profile(
    root: PublicNode,
    joint_prior,
    leaf_value: LeafEvaluator | None,
    *,
    depth=2,
    iterations=32,
    seed=0,
    root_reach=None,
    run_full_iterations=False,
) -> ResolvedProfile:
    """Construct one complete policy profile with coupled sampled CFR searches.

    A subgame's iteration is sampled uniformly from 1..iterations before solving.
    The default stops after evaluating that iteration: later iterations cannot
    change the sampled policy, so this preserves its distribution while saving
    work. ``run_full_iterations=True`` consumes the full nominal search budget.

    Each selected strategy is followed through the ENTIRE truncated public tree.
    Every cutoff is resolved using the reaches from that same selected iteration.
    All unreachable continuation subgames receive uniform legal policies, with
    inactive hand rows left zero. This convention is counted in metrics.
    """
    _positive_integer(depth, "depth")
    _positive_integer(iterations, "iterations")
    start = perf_counter()
    full = PublicBeliefCFR(root, joint_prior)
    if full.has_learned_leaves:
        raise ValueError("resolve_profile requires a complete fixed-payoff tree")
    reaches = _root_reaches(full.shape, root_reach)
    rng = np.random.default_rng(seed)
    strategies = {}
    metrics = {
        "search_calls": 0,
        "leaf_evaluations": 0,
        "leaf_seconds": 0.0,
        "cfr_iterations": 0,
        "zero_mass_fallback_subgames": 0,
        "fallback_information_sets": 0,
        "sampled_iterations": [],
        "nominal_iterations_per_search": iterations,
        "run_full_iterations": bool(run_full_iterations),
        "safe_resolving_guarantee": False,
        "zero_mass_convention": "uniform legal; inactive rows zero",
    }

    def evaluate_leaf(node, belief):
        metrics["leaf_evaluations"] += 1
        before = perf_counter()
        result = 0.0 if leaf_value is None else leaf_value(node, belief)
        metrics["leaf_seconds"] += perf_counter() - before
        return result

    def fill_uniform(node):
        if node.actor is None:
            return
        strategies[node.key] = _uniform(full.masks[node.key])
        metrics["fallback_information_sets"] += 1
        for child in node.children.values():
            fill_uniform(child)

    def resolve(node, reach):
        if node.actor is None:
            return
        mass = float(np.sum(full.prior * reach[0][:, None] * reach[1][None, :]))
        if mass <= 0:
            metrics["zero_mass_fallback_subgames"] += 1
            fill_uniform(node)
            return
        subgame = truncate(node, depth)
        solver = PublicBeliefCFR(subgame, full.prior, evaluate_leaf)
        sampled_iteration = int(rng.integers(1, iterations + 1))
        metrics["search_calls"] += 1
        metrics["sampled_iterations"].append(sampled_iteration)
        selected = None
        count = iterations if run_full_iterations else sampled_iteration
        for step in solver.iterations(count, root_reach=reach):
            metrics["cfr_iterations"] += 1
            if step.iteration == sampled_iteration:
                selected = step
        assert selected is not None
        strategies.update(
            {key: policy.copy() for key, policy in selected.strategies.items()}
        )

        def continue_cutoffs(part):
            if part.actor is None:
                if part.payoff is None:
                    coupled_reach = selected.evaluation.beliefs[part.key].reaches
                    resolve(full.nodes[part.key], coupled_reach)
                return
            for child in part.children.values():
                continue_cutoffs(child)

        continue_cutoffs(subgame)

    resolve(root, reaches)
    if strategies.keys() != full.masks.keys():
        raise RuntimeError("Recursive resolving did not produce a complete policy")
    metrics["policy_information_sets"] = len(strategies)
    metrics["wall_seconds"] = perf_counter() - start
    return ResolvedProfile(strategies, metrics)


def resolve_mixture(
    root: PublicNode,
    joint_prior,
    leaf_value: LeafEvaluator | None,
    *,
    depth=2,
    iterations=32,
    repeats=4,
    seed=0,
    root_reach=None,
    run_full_iterations=False,
) -> ResolvedMixture:
    """Resolve repeatedly and evaluate the own-reach-weighted empirical mixture.

    For each player this is the behavioral realization of the uniform mixture
    of their complete sampled policies. Players' empirical mixtures are then
    evaluated independently; this is not a shared-randomness correlated profile.
    The reported exploitability is exact for this finite behavioral profile. It
    is a sampled approximation to the resolver's expected-policy exploitability,
    not a rigorous upper/lower bound on the true stochastic algorithm.
    """
    _positive_integer(repeats, "repeats")
    start = perf_counter()
    full = PublicBeliefCFR(root, joint_prior)
    reaches = _root_reaches(full.shape, root_reach)
    if full.belief(reaches).mass <= 0:
        raise ValueError("Mixture evaluation requires positive root probability")
    sums = {key: np.zeros(mask.shape) for key, mask in full.masks.items()}
    rng = np.random.default_rng(seed)
    profiles = []
    sample_values = []
    for _ in range(repeats):
        sample_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        result = resolve_profile(
            root,
            full.prior,
            leaf_value,
            depth=depth,
            iterations=iterations,
            seed=sample_seed,
            root_reach=reaches,
            run_full_iterations=run_full_iterations,
        )
        evaluation = full.evaluate(result.strategies, root_reach=reaches)
        for key, policy in result.strategies.items():
            actor = full.nodes[key].actor
            own_reach = evaluation.beliefs[key].reaches[actor]
            sums[key] += own_reach[:, None] * policy
        profiles.append(result.metrics | {"seed": sample_seed})
        sample_values.append(evaluation.value)
    strategies = {}
    for key, weights in sums.items():
        total = weights.sum(axis=1, keepdims=True)
        strategies[key] = np.divide(
            weights,
            total,
            out=_uniform(full.masks[key]),
            where=total > 0,
        )
    evaluation = full.evaluate(strategies, root_reach=reaches)
    exploitability = full.exploitability(strategies, root_reach=reaches)
    metrics = {
        "repeats": repeats,
        "seed": seed,
        "search_calls": sum(p["search_calls"] for p in profiles),
        "leaf_evaluations": sum(p["leaf_evaluations"] for p in profiles),
        "leaf_seconds": sum(p["leaf_seconds"] for p in profiles),
        "cfr_iterations": sum(p["cfr_iterations"] for p in profiles),
        "zero_mass_fallback_subgames": sum(
            p["zero_mass_fallback_subgames"] for p in profiles
        ),
        "fallback_information_sets": sum(
            p["fallback_information_sets"] for p in profiles
        ),
        "profiles": profiles,
        "sample_profile_values": sample_values,
        "policy_information_sets": len(strategies),
        "safe_resolving_guarantee": False,
        "evaluation_scope": "Exact best response to finite empirical player-policy mixtures",
        "stochastic_resolver_exploitability_bound": False,
        "wall_seconds": perf_counter() - start,
    }
    return ResolvedMixture(strategies, metrics, evaluation.value, exploitability)
