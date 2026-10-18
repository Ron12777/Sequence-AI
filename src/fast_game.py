"""Python card/deck interface backed by the Rust rules engine."""

import numpy as np
import sequence_native
from .cards import Card, Deck
from .game import SequenceGame, Move


class FastSequenceGame:
    MAX_HAND_SIZE = 7
    STATE_CHANNELS = 11

    def __init__(self):
        reference = SequenceGame()
        self.board_layout = reference.board_layout
        self.card_to_positions = reference.card_to_positions
        self.reset()

    @property
    def current_player(self):
        return self.c_state.get_info()[0]

    @property
    def winner(self):
        return self.c_state.get_info()[1] or None

    @property
    def game_over(self):
        return self.c_state.get_info()[2]

    @property
    def board(self):
        return np.frombuffer(self.c_state.get_tensor(), dtype=np.int8).reshape(10, 10)

    @property
    def completed_sequences(self):
        return {
            p + 1: [{divmod(i, 10) for i in seq} for seq in seqs]
            for p, seqs in enumerate(self.c_state.get_sequences())
        }

    def reset(self, num_players=2):
        if num_players != 2:
            raise ValueError("Only two-player games are supported")
        self.c_state = sequence_native.GameState()
        self.deck = Deck()
        self.discard_piles = {1: [], 2: []}
        self.hands = {1: self.deck.draw_many(7), 2: self.deck.draw_many(7)}

    def get_legal_moves(self, player=None):
        player = self.current_player if player is None else player
        if player != self.current_player:
            raise ValueError("Native legal moves require the current player")
        return [
            Move(Card.from_int(card), r, c, removal)
            for card, r, c, removal in self.c_state.get_legal_moves(
                [card.to_int() for card in self.hands[player]]
            )
        ]

    def make_move(self, move):
        player = self.current_player
        if move.card not in self.hands[player]:
            return False
        if not self.c_state.apply_move(
            move.card.to_int(), move.row, move.col, int(move.is_removal)
        ):
            return False
        self.hands[player].remove(move.card)
        self.discard_piles[player].append(move.card)
        while (card := self.deck.draw()) is not None:
            if card.is_jack() or any(
                self.board[r, c] == 0 for r, c in self.card_to_positions[card]
            ):
                self.hands[player].append(card)
                break
            self.discard_piles[player].append(card)
        return True

    def get_state_tensor(self, player):
        data = self.c_state.state_tensor(
            [c.to_int() for c in self.hands[player]], player
        )
        return np.frombuffer(data, dtype=np.float32).reshape(11, 10, 10).copy()

    get_state_numpy = get_state_tensor

    def get_result(self, player):
        if not self.game_over or self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def clone(self):
        result = self.__class__.__new__(self.__class__)
        result.board_layout = self.board_layout
        result.card_to_positions = self.card_to_positions
        result.c_state = self.c_state.copy()
        result.hands = {p: list(hand) for p, hand in self.hands.items()}
        result.discard_piles = {
            p: list(cards) for p, cards in self.discard_piles.items()
        }
        result.deck = Deck.__new__(Deck)
        result.deck.num_decks = self.deck.num_decks
        result.deck.cards = list(self.deck.cards)
        return result
