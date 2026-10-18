"""Full-board Sequence public-belief support and shallow public-tree adapter.

All 52 identities, Jacks, legal placements/removals, protected sequences and
terminal wins use the real rules. The sampled public prior conditions on card
counts and public hand sizes, not on an inverse model of historical behavior.
Finite type support is an approximation; it is never conditioned on the actual
acting hand. That hand is queried separately after solving the public game.

Leaves precede replacement draws. At depth two, the first player's private draw
is deferred to the leaf as well: this deliberately omits publicly skipped dead
cards and their information before the second action. It is an approximation,
not an exact full-game ReBeL subgame. No sampled future card enters a strategy.
"""

from math import comb, log

import numpy as np

from src.cards import Card, Deck
from src.game import Move, SequenceGame
from src.public_belief_cfr import PublicBelief, PublicNode

PASS = (-1, -1, -1, False)
FEATURE_SIZE = 562


def action_key(move):
    return (move.card.to_int(), move.row, move.col, bool(move.is_removal))


def card_counts(hand):
    array = np.asarray(hand)
    if array.shape == (52,) and np.issubdtype(array.dtype, np.number):
        counts = array.astype(np.int64)
        if not np.array_equal(array, counts) or (counts < 0).any():
            raise ValueError("Hand counts must be nonnegative integers")
        return counts
    counts = np.zeros(52, dtype=np.int64)
    for card in hand:
        index = card.to_int() if isinstance(card, Card) else int(card)
        if not 0 <= index < 52:
            raise ValueError("Invalid card identity")
        counts[index] += 1
    return counts


def counts_to_cards(counts):
    return [Card.from_int(i) for i, number in enumerate(counts) for _ in range(int(number))]


def public_available_counts(game):
    counts = np.full(52, 2, dtype=np.int64)
    for player in (1, 2):
        for card in game.discard_piles[player]:
            counts[card.to_int()] -= 1
    if (counts < 0).any():
        raise ValueError("Public discard history exceeds physical card counts")
    return counts


def public_copy(game):
    """Copy only public fields; do not invoke clone(), which reads hidden cards."""
    state = SequenceGame.__new__(SequenceGame)
    state.board_layout = game.board_layout
    state.card_to_positions = game.card_to_positions
    state.board = np.array(game.board, copy=True)
    state.current_player = game.current_player
    state.completed_sequences = {
        p: [set(sequence) for sequence in game.completed_sequences[p]] for p in (1, 2)
    }
    state.discard_piles = {p: list(game.discard_piles[p]) for p in (1, 2)}
    state.game_over = game.game_over
    state.winner = game.winner
    state.hands = {1: [], 2: []}
    state.deck = Deck.__new__(Deck)
    state.deck.num_decks = 2
    state.deck.cards = []
    state.id = "public-belief-snapshot"
    return state


def public_transition(state, action):
    """Real board/rule transition, before the unknown replacement draw."""
    child = public_copy(state)
    if action == PASS:
        # This repository's no-legal-action convention is an immediate draw,
        # not the official exchange/pass procedure. PASS is only a tree marker.
        child.game_over = True
        child.winner = None
        return child
    card, row, col, removal = action
    move = Move(Card.from_int(int(card)), int(row), int(col), bool(removal))
    child.hands[child.current_player] = [move.card]
    if not child.make_move(move):
        raise ValueError("Illegal public board transition")
    child.hands = {1: [], 2: []}
    return child


def _log_hand_mass(available, hand, total, size):
    if np.any(hand < 0) or np.any(hand > available) or int(hand.sum()) != size:
        return -np.inf
    if total < size:
        return -np.inf
    return sum(log(comb(int(n), int(k))) for n, k in zip(available, hand)) - log(comb(total, size))


def _ratio(available, known, other, other_log_marginal):
    """P(other hand | known hand) / P(other hand), physical-card prior."""
    if np.any(known > available):
        return 0.0
    conditional = _log_hand_mass(available - known, other,
                                 int(available.sum() - known.sum()), int(other.sum()))
    return float(np.exp(conditional - other_log_marginal)) if np.isfinite(conditional) else 0.0


class FullSequenceBeliefGame:
    def __init__(self, game, particles=8, seed=0, depth=1, events=None, build_tree=True):
        if not isinstance(particles, int) or particles < 1 or depth not in (1, 2):
            raise ValueError("particles must be positive; depth must be 1 or 2")
        self.available_counts = public_available_counts(game)
        self.hand_sizes = (len(game.hands[1]), len(game.hands[2]))
        self.deck_size = len(game.deck.cards)
        if int(self.available_counts.sum()) != sum(self.hand_sizes) + self.deck_size:
            raise ValueError("Public hand sizes, deck size and discard counts are inconsistent")
        self.depth = depth
        self.events = tuple(events or ())  # Public history only, preserved for auditing.
        self.seed = int(seed)
        self.states = {}
        self.spent = {}
        self.pending_draws = {}
        self.nodes = {}
        self.root_player = game.current_player - 1
        self.hands, self.frequencies = self._sample_banks(particles)
        total = int(self.available_counts.sum())
        self.log_marginals = tuple(np.array([
            _log_hand_mass(self.available_counts, hand, total, size) for hand in bank
        ]) for bank, size in zip(self.hands, self.hand_sizes))
        weights = np.array([
            [_ratio(self.available_counts, first, second, self.log_marginals[1][j])
             for j, second in enumerate(self.hands[1])]
            for first in self.hands[0]
        ]) * self.frequencies[0][:, None] * self.frequencies[1][None, :]
        if weights.sum() <= 0:
            raise RuntimeError("Sampled public hand support has no compatible deals")
        self.prior = weights / weights.sum()
        self.build_tree = bool(build_tree)
        initial = public_copy(game)
        spent = np.zeros((2, 52), dtype=np.int64)
        pending = np.zeros(2, dtype=np.int64)
        if self.build_tree:
            self.root = self._build(initial, (), spent, pending)
        else:
            # A feature-only descriptor preserves the player metadata consumed
            # by private query evaluators, without expanding any public actions.
            # Deliberately not a valid CFR decision tree: passing it to a solver
            # raises, rather than silently treating an unexpanded state as a game.
            self.root = PublicNode((), actor=self.root_player, data={"feature_only": True})
            self.states[()] = initial
            self.spent[()] = spent
            self.pending_draws[()] = pending
            self.nodes[()] = self.root

    def _sample_banks(self, particles):
        # Each sampled joint deal supplies marginal hand proposals. Cross-product
        # compatibility reweighting avoids equating one own hand to one opponent
        # hand. Paired proposals guarantee nonempty support in deck-empty states.
        # Correlated finite proposal banks are a support approximation, not an
        # unbiased posterior estimator or an exact behavioral belief update.
        physical = np.repeat(np.arange(52), self.available_counts)
        rng = np.random.default_rng(self.seed)
        banks = [[], []]
        for _ in range(particles):
            shuffled = rng.permutation(physical)
            offset = 0
            for player, size in enumerate(self.hand_sizes):
                banks[player].append(np.bincount(shuffled[offset:offset + size], minlength=52))
                offset += size
        unique, frequencies = [], []
        for bank in banks:
            types, frequency = np.unique(np.asarray(bank), axis=0, return_counts=True)
            unique.append(types)
            frequencies.append(frequency.astype(np.float64) / particles)
        return tuple(unique), tuple(frequencies)

    def state(self, node):
        return self.states[node.key]

    def public_belief(self, reaches=None):
        """Public prior/action posterior, also usable without building a tree."""
        if reaches is None:
            reaches = tuple(np.ones(len(bank)) for bank in self.hands)
        reaches = tuple(np.asarray(r, dtype=np.float64) for r in reaches)
        if len(reaches) != 2 or any(
            r.shape != (len(bank),) or not np.isfinite(r).all() or (r < 0).any()
            for r, bank in zip(reaches, self.hands)
        ):
            raise ValueError("Invalid public reach vectors")
        joint = self.prior * reaches[0][:, None] * reaches[1][None, :]
        mass = float(joint.sum())
        joint = joint / mass if mass else np.zeros_like(joint)
        return PublicBelief(joint, (joint.sum(axis=1), joint.sum(axis=0)),
                            tuple(r.copy() for r in reaches), mass,
                            (self.prior @ reaches[1], self.prior.T @ reaches[0]))

    def hand_counts(self, node):
        return tuple(np.maximum(hand - self.spent[node.key][p], 0)
                     for p, hand in enumerate(self.hands))

    def valid_types(self, node):
        return tuple(np.all(hand >= self.spent[node.key][p], axis=1)
                     for p, hand in enumerate(self.hands))

    def _build(self, state, history, spent, pending):
        key = history
        self.states[key], self.spent[key], self.pending_draws[key] = state, spent, pending
        if state.game_over or len(history) == self.depth or history[-2:] == (PASS, PASS):
            terminal = state.get_result(1) if state.game_over else (0.0 if history[-2:] == (PASS, PASS) else None)
            node = PublicNode(key, payoff=terminal)
            self.nodes[key] = node
            return node
        actor = state.current_player - 1
        per_type = []
        for original in self.hands[actor]:
            remaining = original - spent[actor]
            if (remaining < 0).any():
                per_type.append(set())
                continue
            state.hands[actor + 1] = counts_to_cards(remaining)
            legal = {action_key(move) for move in state.get_legal_moves()}
            per_type.append(legal or {PASS})
        state.hands = {1: [], 2: []}
        actions = sorted(set().union(*per_type))
        if not actions:
            # Entire structurally impossible branch; it has zero policy reach.
            node = PublicNode(key, payoff=0.0)
            self.nodes[key] = node
            return node
        mask = np.array([[action in legal for action in actions] for legal in per_type])
        children = {}
        for action in actions:
            child = public_transition(state, action)
            next_spent, next_pending = spent.copy(), pending.copy()
            if action != PASS:
                next_spent[actor, action[0]] += 1
                next_pending[actor] += int(self.deck_size > 0)
            children[action] = self._build(child, history + (action,), next_spent, next_pending)
        node = PublicNode(key, actor=actor, children=children, legal_mask=mask)
        self.nodes[key] = node
        return node

    def query_support(self, player, own_counts, opponent_reach=None):
        """Conditional opponent bank for a private query, without changing PBS.

        player is zero based. No compatible support returns all zeros: callers
        must explicitly fall back, not secretly add the actual hand to the bank.
        """
        if player not in (0, 1):
            raise ValueError("player must be zero based 0 or 1")
        known = card_counts(own_counts)
        if int(known.sum()) != self.hand_sizes[player] or np.any(known > self.available_counts):
            raise ValueError("Private query is inconsistent with public card counts")
        other = 1 - player
        weights = np.array([_ratio(self.available_counts, known, hand, marginal)
                            for hand, marginal in zip(self.hands[other], self.log_marginals[other])])
        weights *= self.frequencies[other]
        if opponent_reach is not None:
            reach = np.asarray(opponent_reach)
            if reach.shape != weights.shape or not np.isfinite(reach).all() or (reach < 0).any():
                raise ValueError("Invalid opponent reach")
            weights *= reach
        return weights / weights.sum() if weights.sum() else weights

    def query_after_action(self, action, actualhand):
        """Isolated actual-hand successor; never inserts a public private type."""
        known = card_counts(actualhand)
        actor = self.root_player
        if int(known.sum()) != self.hand_sizes[actor] or np.any(known > self.available_counts):
            raise ValueError("Private query is inconsistent with public card counts")
        state = public_copy(self.state(self.root))
        state.hands[actor + 1] = counts_to_cards(known)
        legal = {action_key(move) for move in state.get_legal_moves()} or {PASS}
        if action not in legal:
            raise ValueError("Action is not legal for the queried private hand")
        state.hands = {1: [], 2: []}
        if action != PASS:
            known = known.copy()
            known[action[0]] -= 1
        return public_transition(state, action), known

    def features_for_hand(self, state, player, own_counts, opponent_mean, pending_draws=None):
        """Player-relative hand-value features, no unobserved identity accesses."""
        if pending_draws is None:
            pending_draws = np.zeros(2)
        own, other = player + 1, 2 - player
        board = np.stack([state.board == own, state.board == other,
                          state.board == -1, state.board == 0]).astype(np.float32)
        result = np.concatenate((
            board.ravel(), np.asarray(own_counts) / 2, np.asarray(opponent_mean) / 2,
            public_available_counts(state) / 2,
            [self.deck_size / 104, float(state.current_player != own),
             len(state.completed_sequences[own]) / 2, len(state.completed_sequences[other]) / 2,
             pending_draws[player] / 7, pending_draws[1 - player] / 7],
        )).astype(np.float32)
        assert result.shape == (FEATURE_SIZE,)
        return result

    def features(self, node, belief):
        hands = self.hand_counts(node)
        features = []
        for player in (0, 1):
            joint = self.prior if player == 0 else self.prior.T
            weights = joint * belief.reaches[1 - player][None, :]
            mass = weights.sum(axis=1, keepdims=True)
            conditional = np.divide(weights, mass, out=np.zeros_like(weights), where=mass > 0)
            means = conditional @ hands[1 - player]
            features.append(np.stack([
                self.features_for_hand(self.state(node), player, hand, mean, self.pending_draws[node.key])
                for hand, mean in zip(hands[player], means)
            ]))
        return tuple(features)
