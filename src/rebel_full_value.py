"""Compressed public-belief values for the full Sequence experiment.

Both players' public ranges are inputs. The queried private hand is never added
to either range. A frozen trained board critic initializes values; the learned
residual is fitted to CFR targets and full-game Monte Carlo returns. This is a
particle/feature approximation, not the exact ReBeL algorithm or its guarantee.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.cards import Card
from src.history_policy import load_policy
from src.public_belief_cfr import LeafValues

FEATURES = 1128


class BeliefResidual(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.width = width
        self.layers = nn.Sequential(
            nn.Linear(FEATURES, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, base):
        return (
            torch.atanh(base.clamp(-0.995, 0.995)) + self.layers(features).flatten()
        ).tanh()


def cards_from_counts(counts):
    return [Card.from_int(i) for i, n in enumerate(counts) for _ in range(int(n))]


def moments(hands, weights):
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    weights = weights / total if total else np.zeros(len(hands))
    mean = weights @ hands
    variance = weights @ (hands * hands) - mean * mean
    return mean, np.maximum(variance, 0)


class FullBeliefValue:
    """Cache only public/static data; recompute all reach-dependent features."""

    def __init__(self, teacher, residual=None, device="cpu"):
        self.device = torch.device(device)
        self.teacher = teacher.to(self.device).eval()
        self.teacher.requires_grad_(False)
        self.residual = (residual or BeliefResidual()).to(self.device).eval()
        self._adapter = None
        self._base_cache = {}
        self._node_cache = {}
        self._query_cache = {}
        self._leaf_cache = {}
        self._residual_versions = None
        self.query_support_fallbacks = 0
        self.calls = 0
        self.teacher_batches = 0
        self.teacher_rows = 0

    @property
    def evaluation_models(self):
        return (self.teacher, self.residual)

    def _teacher_values(self, boards):
        values = []
        for offset in range(0, len(boards), 128):
            data = np.stack(boards[offset : offset + 128]).astype(
                np.float32, copy=False
            )
            with torch.inference_mode():
                _, result = self.teacher(torch.from_numpy(data).to(self.device))
            values.append(result.flatten().cpu().numpy().copy())
            self.teacher_batches += 1
            self.teacher_rows += len(data)
        return np.concatenate(values) if values else np.empty(0, np.float32)

    @staticmethod
    def _board_inputs(state, records):
        # state is an isolated public snapshot. Copy once for all hypothetical
        # hands, rather than cloning the board and all sequences for every row.
        game = state.clone()
        game.hands = {1: [], 2: []}
        result = []
        for player, counts in records:
            game.hands[player + 1] = cards_from_counts(counts)
            result.append(game.get_state_tensor(player + 1))
        return result

    def _start(self, adapter):
        if self._adapter is adapter:
            return
        self._adapter = adapter
        self._base_cache.clear()
        self._node_cache.clear()
        self._query_cache.clear()
        self._leaf_cache.clear()
        # Batch the frozen board critic once across this entire public tree.
        # Invalid private types have exactly zero counterfactual mass: their
        # critic values cannot affect CFR and are explicitly left at zero.
        unique, indices, assignments = [], {}, []
        for node in adapter.nodes.values():
            hands = adapter.hand_counts(node)
            valid = adapter.valid_types(node)
            records, positions = [], []
            cursor = 0
            base = np.zeros(sum(len(bank) for bank in hands), np.float32)
            self._base_cache[("node", node.key)] = base
            for player in (0, 1):
                for hand, possible in zip(hands[player], valid[player]):
                    if possible:
                        records.append((player, hand))
                        positions.append(cursor)
                    cursor += 1
            for position, board in zip(
                positions, self._board_inputs(adapter.state(node), records)
            ):
                encoded = board.tobytes()
                index = indices.get(encoded)
                if index is None:
                    index = len(unique)
                    indices[encoded] = index
                    unique.append(board)
                assignments.append((base, position, index))
        values = self._teacher_values(unique)
        for base, position, index in assignments:
            base[position] = values[index]

    def _base(self, adapter, state, hands, cache_key):
        self._start(adapter)
        if cache_key not in self._base_cache:
            self._base_cache[cache_key] = self._teacher_values(
                self._board_inputs(state, hands)
            )
        return self._base_cache[cache_key]

    @staticmethod
    def _pending(adapter, node):
        pending = getattr(adapter, "pending_draws", {})
        return pending.get(node.key, (0, 0)) if isinstance(pending, dict) else (0, 0)

    @staticmethod
    def _static_features(adapter, state, player, pending):
        other = 1 - player
        board = state.board
        protected = []
        for owner in (player + 1, other + 1):
            plane = np.zeros((10, 10), np.float32)
            for sequence in state.completed_sequences[owner]:
                for row, col in sequence:
                    plane[row, col] = 1
            protected.append(plane)
        result = np.zeros(FEATURES, np.float32)
        result[:600] = np.stack(
            (
                board == player + 1,
                board == other + 1,
                board == -1,
                board == 0,
                *protected,
            )
        ).flatten()
        for index, owner in enumerate((player + 1, other + 1)):
            result[600 + index * 52 : 652 + index * 52] = (
                np.bincount(
                    [card.to_int() for card in state.discard_piles[owner]], minlength=52
                )
                / 2
            )
        result[704:756] = adapter.available_counts / 2
        result[1120:1126] = (
            pending[player],
            pending[other],
            int(state.current_player == player + 1),
            adapter.deck_size / 104,
            len(state.completed_sequences[player + 1]) / 2,
            len(state.completed_sequences[other + 1]) / 2,
        )
        return result

    @staticmethod
    def _range_features(hands, ranges, player):
        result = []
        for owner in (player, 1 - player):
            mean, var = moments(hands[owner], ranges[owner])
            result.extend((mean / 2, var / 4))
        return np.concatenate(result)

    def _features(
        self, adapter, state, hands, belief, player, own, conditional, pending=(0, 0)
    ):
        """One private query. Both public ranges stay separate from own hand."""
        result = self._static_features(adapter, state, player, pending)
        mean, var = moments(hands[1 - player], conditional)
        result[756:808] = own / 2
        result[808:860], result[860:912] = mean / 2, var / 4
        result[912:1120] = self._range_features(hands, belief.ranges, player)
        result[1126:] = own.sum() / 7, mean.sum() / 7
        if not np.isfinite(result).all():
            raise ValueError("Invalid full public-belief feature vector")
        return result

    def examples(self, adapter, node, belief):
        self._start(adapter)
        if node.key not in self._node_cache:
            state, hands = adapter.state(node), adapter.hand_counts(node)
            pending = self._pending(adapter, node)
            templates = []
            for player in (0, 1):
                template = np.repeat(
                    self._static_features(adapter, state, player, pending)[None],
                    len(hands[player]),
                    axis=0,
                )
                template[:, 756:808] = hands[player] / 2
                template[:, 1126] = hands[player].sum(axis=1) / 7
                templates.append(template)
            self._node_cache[node.key] = hands, tuple(templates)
        hands, templates = self._node_cache[node.key]
        rows = []
        for player in (0, 1):
            other = 1 - player
            weights = (
                adapter.prior if player == 0 else adapter.prior.T
            ) * belief.reaches[other][None, :]
            mass = weights.sum(axis=1, keepdims=True)
            normalized = np.divide(
                weights,
                mass,
                out=np.zeros_like(weights),
                where=mass > 0,
            )
            mean = normalized @ hands[other]
            var = np.maximum(normalized @ (hands[other] ** 2) - mean**2, 0)
            features = templates[player].copy()
            features[:, 808:860], features[:, 860:912] = mean / 2, var / 4
            features[:, 912:1120] = self._range_features(hands, belief.ranges, player)
            features[:, 1127] = mean.sum(axis=1) / 7
            rows.append(features)
        result = np.concatenate(rows)
        if not np.isfinite(result).all():
            raise ValueError("Invalid full public-belief feature matrix")
        return result, self._base_cache[("node", node.key)]

    def predict(self, features, base):
        with torch.inference_mode():
            result = self.residual(
                torch.from_numpy(features).to(self.device),
                torch.from_numpy(base).to(self.device),
            )
        return result.cpu().numpy().astype(np.float64)

    def _begin_values(self, adapter):
        self._start(adapter)
        versions = tuple(parameter._version for parameter in self.residual.parameters())
        if versions != self._residual_versions:
            self._leaf_cache.clear()
            self._residual_versions = versions

    @staticmethod
    def _belief_key(node, belief):
        normalized = [
            reach / reach.sum() if reach.sum() else reach for reach in belief.reaches
        ]
        return (node.key, *(array.tobytes() for array in (*belief.ranges, *normalized)))

    @staticmethod
    def _reconcile(adapter, belief, values):
        n = len(adapter.hands[0])
        v0, v1 = values[:n], values[n:]
        residual = (belief.ranges[0] @ v0 + belief.ranges[1] @ v1) / 2
        return LeafValues(
            (v0 - residual) / (1 + abs(residual)), (v1 - residual) / (1 + abs(residual))
        )

    def evaluate_leaf(self, adapter, node, belief):
        self.calls += 1
        self._begin_values(adapter)
        key = self._belief_key(node, belief)
        if key not in self._leaf_cache:
            features, base = self.examples(adapter, node, belief)
            self._leaf_cache[key] = self._reconcile(
                adapter, belief, self.predict(features, base)
            )
        return self._leaf_cache[key]

    def evaluate_many(self, adapter, pairs):
        """One neural batch for the fixed policies of one CFR iteration."""
        self.calls += len(pairs)
        self._begin_values(adapter)
        result, pending, feature_rows, bases = {}, [], [], []
        for node, belief in pairs:
            key = self._belief_key(node, belief)
            if key in self._leaf_cache:
                result[node.key] = self._leaf_cache[key]
            else:
                features, base = self.examples(adapter, node, belief)
                feature_rows.append(features)
                bases.append(base)
                pending.append((node, belief, key, len(base)))
        if pending:
            predictions = self.predict(
                np.concatenate(feature_rows), np.concatenate(bases)
            )
            offset = 0
            for node, belief, key, size in pending:
                values = self._reconcile(
                    adapter, belief, predictions[offset : offset + size]
                )
                self._leaf_cache[key] = result[node.key] = values
                offset += size
        return result

    def _query_context(self, adapter, actions, belief, actual_hand, opponent_weights):
        from types import SimpleNamespace

        from src.rebel_full_game import card_counts, public_transition

        if len(actions) not in (1, 2):
            raise ValueError(
                "Private query path must contain one or two public actions"
            )
        player = adapter.root.actor
        state, own = adapter.query_after_action(actions[0], actual_hand)
        if len(actions) == 2:
            state = public_transition(state, actions[1])
        spent = np.zeros((2, 52), dtype=np.int64)
        pending = np.zeros(2, dtype=np.int64)
        for offset, action in enumerate(actions):
            actor = (player + offset) % 2
            if action[0] >= 0:
                spent[actor, action[0]] += 1
                pending[actor] += int(adapter.deck_size > 0)
        hands = tuple(
            np.maximum(bank - spent[p], 0) for p, bank in enumerate(adapter.hands)
        )
        # Project only on PUBLIC consumed cards; never condition public ranges on
        # the actual queried hand. Unsupported banks remain all-zero summaries.
        ranges = []
        for owner in (0, 1):
            valid = np.all(adapter.hands[owner] >= spent[owner], axis=1)
            weights = np.asarray(belief.ranges[owner], dtype=np.float64) * valid
            ranges.append(weights / weights.sum() if weights.sum() else weights)
        public = SimpleNamespace(ranges=tuple(ranges))
        if opponent_weights is None:
            conditional = adapter.query_support(
                player, card_counts(actual_hand), belief.reaches[1 - player]
            )
        else:
            conditional = np.asarray(opponent_weights, dtype=np.float64)
        if (
            conditional.shape != (len(hands[1 - player]),)
            or not np.isfinite(conditional).all()
            or (conditional < 0).any()
        ):
            raise ValueError("Invalid private query opponent weights")
        conditional = conditional * np.all(
            adapter.hands[1 - player] >= spent[1 - player], axis=1
        )
        return state, own, hands, public, conditional, pending

    def _query_encoding(self, adapter, actions, context):
        state, own, hands, public, conditional, pending = context
        player = adapter.root.actor
        features = self._features(
            adapter, state, hands, public, player, own, conditional, pending
        )
        base = self._base(
            adapter, state, [(player, own)], ("query", tuple(actions), tuple(own))
        )[0]
        return features, np.float32(base)

    def query_example(self, adapter, action, belief, actual_hand):
        actions = (action,)
        context = self._query_context(adapter, actions, belief, actual_hand, None)
        return self._query_encoding(adapter, actions, context)

    def evaluate_query_path(
        self, adapter, actions, belief, actual_hand, opponent_weights=None
    ):
        """Actual-hand values after one/two PUBLIC moves, before deferred draws.

        Opponent weights may incorporate the frozen public opponent policy.
        The public range summaries cannot depend on these private query weights.
        Empty conditional support explicitly falls back to the frozen critic.
        """
        context = self._query_context(
            adapter, tuple(actions), belief, actual_hand, opponent_weights
        )
        state = context[0]
        if state.game_over:
            return state.get_result(adapter.root.actor + 1)
        features, base = self._query_encoding(adapter, tuple(actions), context)
        if context[4].sum() <= 0:
            self.query_support_fallbacks += 1
            return float(base)
        return float(self.predict(features[None], np.array([base], np.float32))[0])

    def evaluate_query(self, adapter, action, belief, actual_hand):
        return self.evaluate_query_path(adapter, (action,), belief, actual_hand)


def load_value(path, device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("kind") == "full_sequence_belief_value":
        teacher_path = data["teacher_checkpoint"]
        teacher, _ = load_policy(teacher_path, device)
        residual = BeliefResidual(data["width"])
        residual.load_state_dict(data["residual"])
        return FullBeliefValue(teacher.base, residual, device)
    teacher, _ = load_policy(path, device)
    return FullBeliefValue(teacher.base, device=device)


def save_value(value, path, teacher_checkpoint, training_state=None, metadata=None):
    path = Path(path)
    payload = {
        "kind": "full_sequence_belief_value",
        "width": value.residual.width,
        "teacher_checkpoint": str(teacher_checkpoint),
        "residual": value.residual.state_dict(),
        "training_state": training_state,
        "metadata": metadata or {},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
