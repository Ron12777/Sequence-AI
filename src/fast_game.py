"""
Fast Sequence game wrapper for the C extension.
"""
import numpy as np
from typing import List, Dict, Optional
import c_sequence
from .game import SequenceGame, Move, ChipState
from .cards import Card, Deck, get_cards_per_player

class FastSequenceGame:
    """
    High-performance wrapper around the C GameState.
    Syncs Python-side hand state with C-side board state.
    """
    def __init__(self):
        self.c_state = c_sequence.GameState()
        self.hands: Dict[int, List[Card]] = {1: [], 2: []}
        self.deck = Deck(num_decks=2)
        
    @property
    def current_player(self) -> int:
        return self.c_state.get_info()[0]
        
    @property
    def winner(self) -> int:
        return self.c_state.get_info()[1]
        
    @property
    def game_over(self) -> bool:
        return bool(self.c_state.get_info()[2])

    def reset(self, num_players: int = 2):
        """Reset game to initial state."""
        self.c_state = c_sequence.GameState()
        self.deck = Deck(num_decks=2)
        
        # Deal cards
        cards_per_player = get_cards_per_player(num_players)
        self.hands = {
            1: self.deck.draw_many(cards_per_player),
            2: self.deck.draw_many(cards_per_player)
        }

    def make_move(self, move: Move) -> bool:
        """Execute a move. Syncs Python and C state."""
        player = self.current_player
        
        # 1. Execute move in C
        # args: card_int, row, col, is_removal_int
        success = self.c_state.apply_move(
            move.card.to_int(),
            move.row,
            move.col,
            1 if move.is_removal else 0
        )
        
        if not success:
            return False
            
        # 2. Update Python hands
        self.hands[player].remove(move.card)
        new_card = self.deck.draw()
        if new_card:
            self.hands[player].append(new_card)
            
        return True

    def get_state_tensor(self, player: int) -> np.ndarray:
        """
        Convert C board to neural network input tensor.
        Matches the 8-channel format expected by model.py.
        """
        opponent = 3 - player
        # C board is a bytes object of 100 int8s
        board_bytes = self.c_state.get_tensor()
        board = np.frombuffer(board_bytes, dtype=np.int8).reshape(10, 10)
        
        tensor = np.zeros((8, 10, 10), dtype=np.float32)
        
        # Channels 0-3: Board state
        tensor[0] = (board == player).astype(np.float32)
        tensor[1] = (board == opponent).astype(np.float32)
        tensor[2] = (board == -1).astype(np.float32)  # FREE
        tensor[3] = (board == 0).astype(np.float32)   # EMPTY
        
        # Channels 4-7: Playable positions for first 4 cards in hand
        hand = self.hands[player]
        # To match SequenceGame logic, we'd need layout map here.
        # But for training speed, we can simplify or import it.
        # Let's import the layout mapping from a dummy SequenceGame
        if not hasattr(self, '_pos_map'):
            dummy = SequenceGame()
            self._pos_map = dummy.card_to_positions

        for i, card in enumerate(hand[:4]):
            if card.is_jack():
                continue
            positions = self._pos_map.get(card, [])
            for r, c in positions:
                if board[r, c] == 0:
                    tensor[4 + i, r, c] = 1.0
                    
        return tensor

    def get_state_numpy(self, player: int) -> np.ndarray:
        """Alias for get_state_tensor since it already returns numpy."""
        return self.get_state_tensor(player)

    def get_result(self, player: int) -> float:
        """Get game result from player's perspective."""
        if not self.game_over:
            return 0.0
        if self.winner == player:
            return 1.0
        elif self.winner != 0:
            return -1.0
        return 0.0

    def clone(self) -> 'FastSequenceGame':
        """Deep copy."""
        new_game = FastSequenceGame.__new__(FastSequenceGame)
        new_game.c_state = self.c_state.copy()
        new_game.hands = {k: list(v) for k, v in self.hands.items()}
        new_game.deck = Deck.__new__(Deck)
        new_game.deck.cards = list(self.deck.cards)
        new_game.deck.num_decks = self.deck.num_decks
        if hasattr(self, '_pos_map'):
            new_game._pos_map = self._pos_map
        return new_game
