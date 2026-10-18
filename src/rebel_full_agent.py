"""Full-game action selection using public particle search and a private query.

Search never inserts the actual hand into the public particle bank. After a
uniformly sampled CFR iteration is fixed, the real hand is evaluated separately
against that public search state. This off-support extrapolation and finite
particle belief approximation do not carry ReBeL's equilibrium guarantee.
"""

import hashlib
from time import perf_counter

import numpy as np

from src.public_belief_cfr import PublicBeliefCFR


def load_value(checkpoint, device):
    from src.rebel_full_value import load_value as load

    return load(checkpoint, device=device)


def make_adapter(game, *, particles, seed, depth, events):
    from src.rebel_full_game import FullSequenceBeliefGame

    return FullSequenceBeliefGame(
        game,
        particles=particles,
        seed=seed,
        depth=depth,
        events=events,
    )


def _action_key(move):
    return (move.card.to_int(), move.row, move.col, bool(move.is_removal))


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _public_fingerprint(adapter, selected):
    digest = hashlib.sha256()
    digest.update(np.asarray(adapter.prior, dtype=np.float64).tobytes())
    for hands in adapter.hands:
        digest.update(np.asarray(hands).tobytes())
    for key, policy in selected.strategies.items():
        digest.update(repr(key).encode())
        digest.update(np.asarray(policy, dtype=np.float64).tobytes())
    for key, belief in selected.evaluation.beliefs.items():
        digest.update(repr(key).encode())
        digest.update(np.asarray(belief.joint, dtype=np.float64).tobytes())
        for reach in belief.reaches:
            digest.update(np.asarray(reach, dtype=np.float64).tobytes())
    return digest.hexdigest()


class ReBeLAgent:
    """Experimental public-belief search agent for complete Sequence games.

    ``checkpoint`` is a learned public-value model. The wrapper provides
    evaluate_leaf(adapter, node, belief) -> LeafValues and
    evaluate_query(adapter, action_key, belief, actual_hand) -> scalar utility
    from the current/root player's perspective.

    The public solver runs all requested iterations for training targets. One
    iteration, selected independently in advance, supplies the query beliefs.
    Root targets average iteration CFVs, not a reevaluation of average policies.
    """

    def __init__(
        self,
        checkpoint,
        device="cpu",
        particles=8,
        iterations=16,
        depth=1,
        seed=0,
    ):
        _positive_int(particles, "particles")
        _positive_int(iterations, "iterations")
        _positive_int(depth, "depth")
        if depth not in (1, 2):
            raise ValueError("Public search supports depth=1 or depth=2")
        self.checkpoint = checkpoint
        self.device = device
        self.particles = particles
        self.iterations = iterations
        self.depth = depth
        self.model = load_value(checkpoint, device)
        if depth == 2 and not hasattr(self.model, "evaluate_query_path"):
            raise ValueError("Depth2 requires a value wrapper with evaluate_query_path")
        self.reset(seed)

    @property
    def evaluation_models(self):
        return self.model.evaluation_models

    def reset(self, seed=0, player=None):
        if player is not None and player not in (1, 2):
            raise ValueError("player must be 1 or 2")
        self.seed = int(seed)
        self.player = player
        self.rng = np.random.default_rng(self.seed)
        self.decisions = 0
        self.last_search = None
        self.last_query = None
        self.last_metrics = {}

    def _query_belief(self, adapter, solver, selected, action):
        """Use coupled search posterior; explicitly count off-support fallbacks.

        A zero-likelihood action first conditions on public card legality alone.
        If the bank contains no compatible hand, retain its pre-action public
        belief. This approximation is PUBLIC: it depends only on the played card,
        not on other cards in the actual hand, and never changes the bank.
        """
        child = adapter.root.children.get(action)
        if child is not None:
            belief = selected.evaluation.beliefs[child.key]
            if belief.mass > 0:
                return belief, None
        actor = adapter.root.actor
        if child is not None:
            index = list(adapter.root.children).index(action)
            support = solver.masks[adapter.root.key][:, index]
        else:
            card = int(action[0])
            support = np.asarray(adapter.hands[actor])[:, card] > 0
        root_belief = selected.evaluation.beliefs[adapter.root.key]
        reaches = [np.array(r, copy=True) for r in root_belief.reaches]
        reaches[actor] *= support
        belief = solver.belief(tuple(reaches))
        if belief.mass > 0:
            return belief, "public_legality"
        return root_belief, "unsupported_action"

    def _evaluate_private(
        self, adapter, selected, action, belief, actual_hand, weights
    ):
        """Two-ply queries average frozen opponent actions, never per-deal maxima."""
        metrics = {
            "depth2_queries": 0,
            "depth2_response_evaluations": 0,
            "depth2_one_ply_fallbacks": 0,
            "depth2_public_posterior_fallbacks": 0,
        }
        child = adapter.root.children.get(action)
        if self.depth == 1 or child is None or child.actor is None or not weights.sum():
            if self.depth == 2 and (child is None or not weights.sum()):
                metrics["depth2_one_ply_fallbacks"] += 1
            return float(
                self.model.evaluate_query(adapter, action, belief, actual_hand)
            ), metrics
        opponent = 1 - adapter.root.actor
        if child.actor != opponent or child.key not in selected.strategies:
            raise RuntimeError("Depth2 query lacks its frozen opponent policy")
        policy = selected.strategies[child.key]
        response_mass = weights @ policy
        if not np.isfinite(response_mass).all() or not np.isclose(
            response_mass.sum(), 1
        ):
            raise RuntimeError(
                "Frozen opponent response is not a complete conditional policy"
            )
        expected = 0.0
        metrics["depth2_queries"] += 1
        for index, (response, leaf) in enumerate(child.children.items()):
            probability = float(response_mass[index])
            if probability <= 0:
                continue
            conditional = weights * policy[:, index] / probability
            public = selected.evaluation.beliefs[leaf.key]
            if public.mass <= 0:
                # Off-support public observations retain the pre-response public
                # bank. Query-compatible opponent weights still update. This
                # explicit finite-bank fallback never adds the real hand.
                public = belief
                metrics["depth2_public_posterior_fallbacks"] += 1
            if leaf.payoff is not None:
                payoff = np.asarray(leaf.payoff)
                if payoff.ndim != 0:
                    raise ValueError("Full Sequence terminal query needs public scalar payoff")
                continuation = float(payoff) * (1 if adapter.root.actor == 0 else -1)
            else:
                continuation = float(
                    self.model.evaluate_query_path(
                        adapter,
                        (action, response),
                        public,
                        actual_hand,
                        opponent_weights=conditional,
                    )
                )
            if not np.isfinite(continuation):
                raise ValueError("Private continuation returned a nonfinite value")
            expected += probability * continuation
            metrics["depth2_response_evaluations"] += 1
        return expected, metrics

    def act(self, game, player, events):
        start = perf_counter()
        if player not in (1, 2) or game.current_player != player:
            raise ValueError("act must be called for the current player")
        if self.player is not None and self.player != player:
            raise ValueError("agent player does not match reset(player=...)")
        legal = game.get_legal_moves()
        if not legal:
            self.last_search = self.last_query = None
            self.last_metrics = {
                "legal_moves": 0,
                "wall_seconds": perf_counter() - start,
            }
            return None
        # This seed depends only on the agent's public random stream and turn
        # count; neither hand identities nor the real deck are consulted.
        public_seed = int(self.rng.integers(0, np.iinfo(np.int64).max))
        selected_index = int(self.rng.integers(1, self.iterations + 1))
        adapter = make_adapter(
            game,
            particles=self.particles,
            seed=public_seed,
            depth=self.depth,
            events=events,
        )
        leaf_calls = 0

        def leaf(node, belief):
            nonlocal leaf_calls
            leaf_calls += 1
            return self.model.evaluate_leaf(adapter, node, belief)

        if hasattr(self.model, "evaluate_many"):

            def leaf_many(pairs):
                nonlocal leaf_calls
                leaf_calls += len(pairs)
                return self.model.evaluate_many(adapter, pairs)

            leaf.evaluate_many = leaf_many

        solver = PublicBeliefCFR(adapter.root, adapter.prior, leaf)
        means = tuple(np.zeros(n) for n in solver.shape)
        selected = None
        for step in solver.iterations(self.iterations):
            for target, value in zip(means, step.evaluation.counterfactual_values):
                target += value / self.iterations
            if step.iteration == selected_index:
                selected = step
        assert selected is not None
        public_seconds = perf_counter() - start
        fingerprint = _public_fingerprint(adapter, selected)
        root_belief = selected.evaluation.beliefs[adapter.root.key]
        conditional = tuple(
            np.divide(value, mass, out=np.zeros_like(value), where=mass > 0)
            for value, mass in zip(means, root_belief.compatible_mass)
        )
        actual_hand = tuple(game.hands[player])
        query_start = perf_counter()
        chosen_move = chosen_action = chosen_belief = None
        chosen_value = -np.inf
        chosen_support_valid = False
        unsupported_private_support = 0
        fallback_counts = {"public_legality": 0, "unsupported_action": 0}
        depth_metrics = {
            "depth2_queries": 0,
            "depth2_response_evaluations": 0,
            "depth2_one_ply_fallbacks": 0,
            "depth2_public_posterior_fallbacks": 0,
        }
        for move in sorted(legal, key=_action_key):
            action = _action_key(move)
            belief, fallback = self._query_belief(adapter, solver, selected, action)
            if fallback is not None:
                fallback_counts[fallback] += 1
            query_weights = adapter.query_support(
                adapter.root.actor, actual_hand, belief.reaches[1 - adapter.root.actor]
            )
            support_valid = bool(np.any(query_weights > 0))
            unsupported_private_support += int(not support_valid)
            value, query_metrics = self._evaluate_private(
                adapter,
                selected,
                action,
                belief,
                actual_hand,
                query_weights,
            )
            for key, number in query_metrics.items():
                depth_metrics[key] += number
            if not np.isfinite(value):
                raise ValueError("Private query returned a nonfinite value")
            if value > chosen_value:
                chosen_support_valid = support_valid
                chosen_move, chosen_action, chosen_belief, chosen_value = (
                    move,
                    action,
                    belief,
                    value,
                )
        # Query code must never feed private observations back into public search.
        if fingerprint != _public_fingerprint(adapter, selected):
            raise RuntimeError("Private query mutated the public particle search")
        self.decisions += 1
        self.last_query = (chosen_action, chosen_belief, actual_hand)
        self.last_metrics = {
            "public_seed": public_seed,
            "selected_iteration": selected_index,
            "cfr_iterations": self.iterations,
            "leaf_evaluations": leaf_calls,
            "query_evaluations": len(legal),
            "public_legality_fallbacks": fallback_counts["public_legality"],
            "unsupported_action_fallbacks": fallback_counts["unsupported_action"],
            "unsupported_actual_hand_queries": unsupported_private_support,
            "public_search_seconds": public_seconds,
            "private_query_seconds": perf_counter() - query_start,
            "wall_seconds": perf_counter() - start,
            "public_search_fingerprint": fingerprint,
            "particle_count": self.particles,
            "depth": self.depth,
            **depth_metrics,
            "private_query_method": "off-support value response to frozen public search",
            "equilibrium_guarantee": False,
        }
        self.last_search = {
            "adapter": adapter,
            "solver": solver,
            "selected_iteration": selected,
            "mean_root_counterfactual_values": means,
            "mean_root_conditional_values": conditional,
            "root_belief": root_belief,
            "root_player": player,
            "chosen_value": chosen_value,
            "query_support_valid": chosen_support_valid,
            "last_query": self.last_query,
            "metrics": self.last_metrics,
        }
        return chosen_move
