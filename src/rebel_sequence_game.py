"""Exact, restricted Sequence endgame for public-belief research.

This is a synthetic benchmark, not a replacement for full-game Sequence. It
uses the real 10x10 rules, two copies of each of six card identities, two hidden
cards per player, an empty draw pile, and hidden unused cards. There are no
Jacks. Four turns exhaust every playable hand; a dead hand passes publicly.
Both players' complete initial private hand types are retained for recall.
"""

from itertools import combinations_with_replacement
from math import comb

import numpy as np

from src.cards import Card
from src.game import SequenceGame
from src.public_belief_cfr import PublicNode

PASS = (-1, -1, -1, False)
POOL = ("10D", "2H", "5H", "10C", "3H", "AD")


class SequenceBeliefGame:
    def __init__(self, variant=0):
        self.cards = tuple(Card.from_string(s) for s in POOL)
        self.types = tuple(combinations_with_replacement(range(len(self.cards)), 2))
        self.counts = np.array(
            [[hand.count(c) for c in range(len(self.cards))] for hand in self.types]
        )
        self.prior = self._prior()
        self.games = {}
        self.spent = {}
        self.nodes = {}
        self.variant = variant
        game = SequenceGame()
        game.deck.cards = []
        game.hands = {1: [], 2: []}
        for player, row, columns in [(1, 8, range(4, 9)), (2, 9, range(1, 6))]:
            cells = {(row, c) for c in columns}
            for r, c in cells:
                game.board[r, c] = player
            game.completed_sequences[player] = [cells]
        for c in (1, 2, 3):
            game.board[4, c] = 1
        for r in (1, 2, 3):
            game.board[r, 4] = 2
        # Variants alter public threats, never reveal a sampled private hand.
        if variant == 1:
            game.board[4, 1] = 0
            game.board[4, 6] = 1
        elif variant == 2:
            game.board[1, 4] = 0
            game.board[6, 4] = 2
        elif variant != 0:
            raise ValueError("Unknown restricted benchmark variant")
        self.root = self._build(game, (), ((), ()))

    def _prior(self):
        n = len(self.types)
        prior = np.zeros((n, n), dtype=np.float64)
        for i, a in enumerate(self.counts):
            for j, b in enumerate(self.counts):
                if np.any(a + b > 2):
                    continue
                weight = 1
                for ca, cb in zip(a, b):
                    weight *= comb(2, int(ca)) * comb(2 - int(ca), int(cb))
                prior[i, j] = weight
        return prior / prior.sum()

    def _build(self, game, history, spent):
        key = (self.variant, history)
        self.games[key] = game
        self.spent[key] = spent
        if game.game_over or len(history) == 4:
            payoff = game.get_result(1) if game.game_over else 0.0
            node = PublicNode(key=key, payoff=payoff)
            self.nodes[key] = node
            return node
        actor = game.current_player - 1
        spent_counts = [np.bincount(s, minlength=len(self.cards)) for s in spent]
        valid = [np.all(self.counts >= sc, axis=1) for sc in spent_counts]
        compatible = (self.prior > 0) & valid[0][:, None] & valid[1][None, :]
        valid = [compatible.any(axis=1), compatible.any(axis=0)]
        per_type = []
        action_moves = {}
        for h, counts in enumerate(self.counts):
            if not valid[actor][h]:
                per_type.append(set())
                continue
            remaining = counts - spent_counts[actor]
            game.hands[actor + 1] = [
                card for card, count in zip(self.cards, remaining) for _ in range(count)
            ]
            legal = game.get_legal_moves()
            keys = set()
            for move in legal:
                action = (move.card.to_int(), move.row, move.col, move.is_removal)
                keys.add(action)
                action_moves[action] = move
            per_type.append(keys or {PASS})
        game.hands = {1: [], 2: []}
        actions = sorted(set().union(*per_type))
        if not actions:
            raise RuntimeError("Unreachable public node was expanded")
        legal_mask = np.array([[a in keys for a in actions] for keys in per_type])
        children = {}
        for action in actions:
            child = game.clone()
            child_spent = [tuple(x) for x in spent]
            if action == PASS:
                child.current_player = 3 - child.current_player
            else:
                move = action_moves[action]
                child.hands[actor + 1] = [move.card]
                assert child.make_move(move)
                child.hands = {1: [], 2: []}
                card_id = self.cards.index(move.card)
                child_spent[actor] += (card_id,)
            # Filter structural impossible branches caused by public pass: the
            # solver's legal mask gives zero reach to incompatible private types.
            children[action] = self._build(
                child, history + (action,), tuple(child_spent)
            )
        node = PublicNode(
            key=key, actor=actor, children=children, legal_mask=legal_mask
        )
        self.nodes[key] = node
        return node

    def features(self, node, belief):
        """Only public board/history and public joint belief, never true hands."""
        game = self.games[node.key]
        spent = self.spent[node.key]
        board = np.stack([(game.board == x) for x in (1, 2, -1, 0)]).astype(np.float32)
        used = np.concatenate(
            [np.bincount(s, minlength=len(self.cards)) for s in spent]
        )
        return np.concatenate(
            (
                board.ravel(),
                [game.current_player - 1, len(node.key[1]) / 4],
                used / 2,
                np.asarray(belief.joint).ravel(),
                *[r / max(float(r.sum()), 1e-30) for r in belief.reaches],
            )
        ).astype(np.float32)

    def summary(self):
        return {
            "variant": self.variant,
            "private_hand_types_per_player": len(self.types),
            "compatible_private_deals": int(np.count_nonzero(self.prior)),
            "public_nodes": len(self.nodes),
            "terminal_nodes": sum(n.payoff is not None for n in self.nodes.values()),
            "pool": list(POOL),
            "physical_copies_per_identity": 2,
            "scope": "Synthetic restricted Sequence endgame; not full-game training or deployment",
        }
