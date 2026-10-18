"""Tabular CFR on a two-player, zero-sum public tree with private root types.

An information set is (public node, acting player's private type). Actions are
public; private types have perfect recall and may have a correlated joint prior.
There is no post-root chance in this representation. A node must have a distinct
key even if two public histories happen to produce the same visible board.

Full-tree, fixed-payoff CFR and best responses are exact for this finite game.
Reach-conditioned learned leaves are an approximation: this module is a ReBeL
component, not a complete ReBeL training or safe resolving implementation.
"""

from collections.abc import Callable, Hashable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PublicNode:
    key: Hashable
    actor: int | None = None
    children: dict[Hashable, "PublicNode"] = field(default_factory=dict)
    legal_mask: np.ndarray | None = None
    payoff: np.ndarray | float | None = None  # Player 0 utility.
    data: object = None  # Optional PUBLIC features for a learned leaf evaluator.


@dataclass(frozen=True)
class PublicBelief:
    joint: np.ndarray  # Normalized posterior, all zero at an unreachable node.
    ranges: tuple[np.ndarray, np.ndarray]
    reaches: tuple[np.ndarray, np.ndarray]  # Action likelihoods, excluding prior.
    mass: float
    compatible_mass: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LeafValues:
    """Per-hand conditional utilities, each in that player's own perspective.

    For player 0 hand i the conditioning measure on opponent hand j is
    prior[i,j]*reach1[j], normalized by its sum. Player 1 uses the transpose.
    Own reach is deliberately excluded: counterfactual values remain meaningful
    for hands whose own strategy assigned zero likelihood to this public history.
    The solver multiplies these values by the compatible opponent reach mass.
    """

    player0: np.ndarray
    player1: np.ndarray


@dataclass
class Evaluation:
    value: float  # Player 0, conditional on reaching the supplied root.
    player_values: tuple[float, float]
    conditional_values: tuple[np.ndarray, np.ndarray]
    counterfactual_values: tuple[np.ndarray, np.ndarray]
    node_values: dict[Hashable, tuple[np.ndarray, np.ndarray]]
    beliefs: dict[Hashable, PublicBelief]


@dataclass
class IterationResult:
    iteration: int
    strategies: dict[Hashable, np.ndarray]
    evaluation: Evaluation


@dataclass
class SolveResult:
    strategies: dict[Hashable, np.ndarray]
    evaluation: Evaluation
    last: IterationResult
    mean_node_counterfactual_values: dict[Hashable, tuple[np.ndarray, np.ndarray]]

    @property
    def value(self):
        return self.evaluation.value

    @property
    def conditional_values(self):
        return self.evaluation.conditional_values

    @property
    def counterfactual_values(self):
        return self.evaluation.counterfactual_values


LeafEvaluator = Callable[[PublicNode, PublicBelief], LeafValues | np.ndarray | float]


class PublicBeliefCFR:
    def __init__(self, root: PublicNode, joint_prior, leaf_value: LeafEvaluator | None = None):
        self.root = root
        self.prior = np.array(joint_prior, dtype=np.float64, copy=True)
        if (self.prior.ndim != 2 or min(self.prior.shape) == 0
                or not np.isfinite(self.prior).all() or (self.prior < 0).any()
                or self.prior.sum() <= 0):
            raise ValueError("joint_prior must be a finite, nonnegative, nonempty matrix")
        self.prior /= self.prior.sum()
        self.shape = self.prior.shape
        self.leaf_value = leaf_value
        self.nodes = {}
        self.masks = {}
        self.has_learned_leaves = False
        self._validate(root)

    def _validate(self, node):
        if node.key in self.nodes:
            raise ValueError("Public history keys must be unique; trees cannot share nodes")
        self.nodes[node.key] = node
        if node.actor is None:
            if node.children or node.legal_mask is not None:
                raise ValueError("Leaves cannot have actions or masks")
            if node.payoff is None:
                self.has_learned_leaves = True
                if self.leaf_value is None:
                    raise ValueError("A cutoff leaf requires leaf_value")
            else:
                self._payoff(node.payoff)
            return
        if node.actor not in (0, 1) or not node.children or node.payoff is not None:
            raise ValueError("Decision node requires actor 0/1, children and no payoff")
        shape = (self.shape[node.actor], len(node.children))
        mask = (np.ones(shape, dtype=bool) if node.legal_mask is None
                else np.asarray(node.legal_mask, dtype=bool))
        if mask.shape != shape:
            raise ValueError("Legal mask must have shape (own types, public actions)")
        self.masks[node.key] = mask.copy()
        for child in node.children.values():
            self._validate(child)

    def _payoff(self, value):
        try:
            payoff = np.broadcast_to(np.asarray(value, dtype=np.float64), self.shape)
        except ValueError as exc:
            raise ValueError("Payoff must broadcast to joint private-type shape") from exc
        if not np.isfinite(payoff).all():
            raise ValueError("Payoffs must be finite")
        return payoff

    def _reaches(self, root_reach):
        if root_reach is None:
            return tuple(np.ones(n) for n in self.shape)
        if len(root_reach) != 2:
            raise ValueError("root_reach must contain two vectors")
        result = tuple(np.array(x, dtype=np.float64, copy=True) for x in root_reach)
        for x, n in zip(result, self.shape):
            if x.shape != (n,) or not np.isfinite(x).all() or (x < 0).any():
                raise ValueError("Reach factors must be finite nonnegative type vectors")
        if np.sum(self.prior * result[0][:, None] * result[1][None, :]) <= 0:
            raise ValueError("The supplied root has zero probability")
        return result

    def belief(self, reaches=None):
        """Build a public posterior, preserving correlated type compatibility."""
        if reaches is None:
            reaches = tuple(np.ones(n) for n in self.shape)
        r0, r1 = reaches
        joint = self.prior * r0[:, None] * r1[None, :]
        mass = float(joint.sum())
        posterior = joint / mass if mass else np.zeros_like(joint)
        return PublicBelief(
            posterior, (posterior.sum(axis=1), posterior.sum(axis=0)),
            (r0.copy(), r1.copy()), mass, (self.prior @ r1, self.prior.T @ r0),
        )

    def posterior_after_action(self, node, strategy, action, reaches=None):
        """Bayes update by public action likelihood indexed ONLY by own hand."""
        if node.actor is None:
            raise ValueError("A leaf has no actions")
        if reaches is None:
            reaches = tuple(np.ones(n) for n in self.shape)
        checked = self._strategy(node, strategy)
        index = list(node.children).index(action)
        updated = [np.array(x, copy=True) for x in reaches]
        updated[node.actor] *= checked[:, index]
        return self.belief(tuple(updated))

    def _strategy(self, node, strategy):
        p = np.asarray(strategy, dtype=np.float64)
        mask = self.masks[node.key]
        if (p.shape != mask.shape or not np.isfinite(p).all() or (p < 0).any()
                or np.any(p[~mask] != 0)
                or not np.allclose(p.sum(axis=1), mask.any(axis=1), atol=1e-10, rtol=0)):
            raise ValueError(f"Invalid information-set strategy at {node.key!r}")
        return p

    def _policies(self, strategies):
        return {key: self._strategy(self.nodes[key], strategies[key]) for key in self.masks}

    def _match(self, scores, key):
        mask = self.masks[key]
        positive = np.maximum(scores, 0) * mask
        total = positive.sum(axis=1, keepdims=True)
        legal_count = mask.sum(axis=1, keepdims=True)
        fallback = np.divide(mask, legal_count, out=np.zeros(mask.shape), where=legal_count > 0)
        return np.divide(positive, total, out=fallback, where=total > 0)

    def _leaf(self, node, belief, batch_values=None):
        if node.payoff is not None:
            value = node.payoff
        elif batch_values is not None:
            value = batch_values[node.key]
        else:
            value = self.leaf_value(node, belief)
        if isinstance(value, LeafValues):
            if node.payoff is not None:
                raise ValueError("Terminal payoff must be a zero-sum payoff matrix")
            vectors = tuple(np.asarray(x, dtype=np.float64) for x in (value.player0, value.player1))
            if any(v.shape != (n,) or not np.isfinite(v).all() for v, n in zip(vectors, self.shape)):
                raise ValueError("LeafValues requires finite per-hand vectors")
            return tuple(v * m for v, m in zip(vectors, belief.compatible_mass))
        payoff = self._payoff(value)
        return ((self.prior * payoff) @ belief.reaches[1],
                -(self.prior * payoff).T @ belief.reaches[0])

    def _walk(self, strategies, reaches, regrets=None, sums=None):
        node_values, beliefs = {}, {}
        batch_values = None
        evaluate_many = getattr(self.leaf_value, "evaluate_many", None)
        if callable(evaluate_many) and self.has_learned_leaves:
            # Every policy is fixed for this simultaneous CFR iteration. Thus
            # every public belief can be constructed before any value inference.
            # An optional neural callback batches only learned cutoff leaves;
            # exact terminal payoffs always bypass the learned evaluator.
            leaves = []

            def collect(node, reach):
                public = self.belief(reach)
                beliefs[node.key] = public
                if node.actor is None:
                    if node.payoff is None:
                        leaves.append((node, public))
                    return
                actor = node.actor
                for index, child in enumerate(node.children.values()):
                    next_reach = list(reach)
                    next_reach[actor] = reach[actor] * strategies[node.key][:, index]
                    collect(child, tuple(next_reach))

            collect(self.root, reaches)
            batch_values = evaluate_many(leaves)
            if not isinstance(batch_values, Mapping) or set(batch_values) != {n.key for n, _ in leaves}:
                raise ValueError("evaluate_many must return exactly the requested cutoff leaf keys")

        def visit(node, reach):
            if node.key not in beliefs:
                beliefs[node.key] = self.belief(reach)
            public = beliefs[node.key]
            if node.actor is None:
                values = self._leaf(node, public, batch_values)
            else:
                actor = node.actor
                policy = strategies[node.key]
                inactive = ~self.masks[node.key].any(axis=1)
                if np.any(public.ranges[actor][inactive] > 0):
                    raise ValueError("A positive-probability private type has no legal action")
                child_values = []
                for index, child in enumerate(node.children.values()):
                    next_reach = list(reach)
                    next_reach[actor] = reach[actor] * policy[:, index]
                    child_values.append(visit(child, tuple(next_reach)))
                action_values = np.stack([v[actor] for v in child_values], axis=1)
                own = (policy * action_values).sum(axis=1)
                other = sum(v[1 - actor] for v in child_values)
                values = (own, other) if actor == 0 else (other, own)
                if regrets is not None:
                    regrets[node.key] += (action_values - own[:, None]) * self.masks[node.key]
                if sums is not None:
                    sums[node.key] += reach[actor][:, None] * policy
            node_values[node.key] = values
            return values

        cf = visit(self.root, reaches)
        public = beliefs[self.root.key]
        conditional = tuple(np.divide(v, m, out=np.zeros_like(v), where=m > 0)
                            for v, m in zip(cf, public.compatible_mass))
        expected = tuple(float(r @ v / public.mass) for r, v in zip(reaches, cf))
        return Evaluation(expected[0], expected, conditional, cf, node_values, beliefs)

    def evaluate(self, strategies: Mapping, root_reach=None):
        """Evaluate a policy. Learned-cutoff results are estimates, not game values.

        CFVs include normalized prior and opponent reach, excluding own reach.
        Conditional values divide by compatible opponent mass; undefined entries
        use zero. Root scalar values divide by public root probability.
        """
        return self._walk(self._policies(strategies), self._reaches(root_reach))

    def iterations(self, count: int, root_reach=None) -> Iterator[IterationResult]:
        """Fresh simultaneous-update vanilla CFR run; yield each pre-update policy.

        The yielded strategy and beliefs/values come from the SAME iteration.
        For ReBeL-style trajectory generation, sample one iteration in advance
        (or reservoir sample) and couple its acting policy with its public belief.
        This iterator does not itself implement the outer ReBeL sampling scheme.
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        reaches = self._reaches(root_reach)
        regrets = {key: np.zeros(mask.shape) for key, mask in self.masks.items()}
        for iteration in range(1, count + 1):
            strategies = {key: self._match(v, key) for key, v in regrets.items()}
            evaluation = self._walk(strategies, reaches, regrets=regrets)
            yield IterationResult(iteration, strategies, evaluation)

    def solve(self, iterations: int, root_reach=None):
        reaches = self._reaches(root_reach)
        sums = {key: np.zeros(mask.shape) for key, mask in self.masks.items()}
        means = {}
        last = None
        for last in self.iterations(iterations, reaches):
            for key, policy in last.strategies.items():
                actor = self.nodes[key].actor
                sums[key] += last.evaluation.beliefs[key].reaches[actor][:, None] * policy
            for key, values in last.evaluation.node_values.items():
                if key not in means:
                    means[key] = [np.zeros_like(v) for v in values]
                for player in (0, 1):
                    means[key][player] += values[player] / iterations
        averaged = {key: self._match(v, key) for key, v in sums.items()}
        return SolveResult(averaged, self.evaluate(averaged, reaches), last,
                           {key: tuple(v) for key, v in means.items()})

    def best_response(self, strategies, player: int, root_reach=None):
        """Exact infoset-respecting full-game best response, in player's utility.

        Returns (conditional scalar value, responding player's policy mapping).
        A response action maximizes a sum over compatible opponent hands. It
        NEVER maximizes each matrix entry independently (strategy fusion).
        """
        if self.has_learned_leaves:
            raise ValueError("Exact best responses require full fixed-payoff trees")
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        policies = self._policies(strategies)
        reaches = self._reaches(root_reach)
        response = {}

        def visit(node, reach):
            if node.actor is None:
                return self._leaf(node, self.belief(reach))[player]
            actor = node.actor
            children = []
            for index, child in enumerate(node.children.values()):
                next_reach = list(reach)
                if actor != player:
                    next_reach[actor] = reach[actor] * policies[node.key][:, index]
                children.append(visit(child, tuple(next_reach)))
            if actor != player:
                return sum(children)
            values = np.stack(children, axis=1)
            choice = np.where(self.masks[node.key], values, -np.inf).argmax(axis=1)
            policy = np.zeros_like(values)
            active = self.masks[node.key].any(axis=1)
            policy[np.arange(len(choice))[active], choice[active]] = 1
            response[node.key] = policy
            return np.where(active, values[np.arange(len(choice)), choice], 0)

        cf = visit(self.root, reaches)
        value = float(reaches[player] @ cf / self.belief(reaches).mass)
        return value, response

    def exploitability(self, strategies, root_reach=None):
        """NashConv/2 in utility units, only for full zero-sum finite games."""
        values = [self.best_response(strategies, p, root_reach)[0] for p in (0, 1)]
        return max(0.0, sum(values) / 2)
