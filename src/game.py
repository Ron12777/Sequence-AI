"""
Sequence game logic - board state, moves, and win detection.
"""
import csv
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from .cards import Card, Deck, get_cards_per_player


class ChipState(IntEnum):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2
    FREE = -1  # Corner spaces count for everyone


@dataclass
class Move:
    """Represents a game move."""
    card: Card
    row: int
    col: int
    is_removal: bool = False  # True for one-eyed Jack removals
    
    def __str__(self) -> str:
        action = "remove" if self.is_removal else "place"
        return f"{self.card} -> {action} ({self.row}, {self.col})"


class SequenceGame:
    """
    Full Sequence game state and logic.
    """
    BOARD_SIZE = 10
    SEQUENCE_LENGTH = 5
    
    def __init__(self, board_csv_path: Optional[str] = None):
        # Load board layout from CSV
        if board_csv_path is None:
            board_csv_path = Path(__file__).parent.parent / "BoardDesign.csv"
        
        self.board_layout = self._load_board_layout(board_csv_path)
        self.card_to_positions = self._build_card_position_map()
        
        # Game state
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.current_player = 1  # 1 or 2
        self.hands: Dict[int, List[Card]] = {1: [], 2: []}
        self.deck = Deck(num_decks=2)
        self.discard_piles: Dict[int, List[Card]] = {1: [], 2: []}
        self.completed_sequences: Dict[int, List[Set[Tuple[int, int]]]] = {1: [], 2: []}
        self.game_over = False
        self.winner = None
        
        # Mark corner spaces as FREE
        for r, c in [(0, 0), (0, 9), (9, 0), (9, 9)]:
            self.board[r, c] = ChipState.FREE
    
    def _load_board_layout(self, csv_path: str) -> List[List[Optional[Card]]]:
        """Load board layout from CSV file."""
        layout = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                layout_row = []
                for cell in row:
                    cell = cell.strip()
                    if cell == 'FREE':
                        layout_row.append(None)  # Corner
                    else:
                        layout_row.append(Card.from_string(cell))
                layout.append(layout_row)
        return layout
    
    def _build_card_position_map(self) -> Dict[Card, List[Tuple[int, int]]]:
        """Build mapping from card to board positions."""
        mapping = {}
        for r, row in enumerate(self.board_layout):
            for c, card in enumerate(row):
                if card is not None:
                    if card not in mapping:
                        mapping[card] = []
                    mapping[card].append((r, c))
        return mapping
    
    def reset(self, num_players: int = 2):
        """Reset game to initial state."""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        for r, c in [(0, 0), (0, 9), (9, 0), (9, 9)]:
            self.board[r, c] = ChipState.FREE
        
        self.current_player = 1
        self.deck = Deck(num_decks=2)
        self.discard_piles = {1: [], 2: []}
        self.completed_sequences = {1: [], 2: []}
        self.game_over = False
        self.winner = None
        
        # Deal cards
        cards_per_player = get_cards_per_player(num_players)
        self.hands = {
            1: self.deck.draw_many(cards_per_player),
            2: self.deck.draw_many(cards_per_player)
        }
    
    def get_legal_moves(self, player: Optional[int] = None) -> List[Move]:
        """Get all legal moves for a player."""
        if player is None:
            player = self.current_player
        
        if self.game_over:
            return []
        
        moves = []
        hand = self.hands[player]
        
        for card in hand:
            if card.is_red_jack():
                # Red Jack (Wild): can place on any empty non-corner space
                for r in range(self.BOARD_SIZE):
                    for c in range(self.BOARD_SIZE):
                        if self.board[r, c] == ChipState.EMPTY:
                            moves.append(Move(card=card, row=r, col=c))
            
            elif card.is_black_jack():
                # Black Jack (Remove): can remove any opponent chip not in a sequence
                opponent = 3 - player
                protected = self._get_protected_positions(opponent)
                for r in range(self.BOARD_SIZE):
                    for c in range(self.BOARD_SIZE):
                        if self.board[r, c] == opponent and (r, c) not in protected:
                            moves.append(Move(card=card, row=r, col=c, is_removal=True))
            
            else:
                # Normal card: place on corresponding empty position
                positions = self.card_to_positions.get(card, [])
                for r, c in positions:
                    if self.board[r, c] == ChipState.EMPTY:
                        moves.append(Move(card=card, row=r, col=c))
        
        return moves
    
    def _get_protected_positions(self, player: int) -> Set[Tuple[int, int]]:
        """Get positions that are part of completed sequences (protected from removal)."""
        protected = set()
        for seq in self.completed_sequences[player]:
            protected.update(seq)
        return protected
    
    def is_dead_card(self, card: Card, player: int) -> bool:
        """Check if card has no valid plays (both positions covered)."""
        if card.is_jack():
            return False  # Jacks always have potential plays
        
        positions = self.card_to_positions.get(card, [])
        for r, c in positions:
            if self.board[r, c] == ChipState.EMPTY:
                return False
        return True
    
    def make_move(self, move: Move) -> bool:
        """Execute a move. Returns True if successful."""
        if self.game_over:
            return False
        
        player = self.current_player
        
        # Validate move
        if move.card not in self.hands[player]:
            return False
        
        legal_moves = self.get_legal_moves(player)
        if not any(m.card == move.card and m.row == move.row and 
                   m.col == move.col and m.is_removal == move.is_removal 
                   for m in legal_moves):
            return False
        
        # Execute move
        self.hands[player].remove(move.card)
        self.discard_piles[player].append(move.card)
        
        if move.is_removal:
            self.board[move.row, move.col] = ChipState.EMPTY
        else:
            self.board[move.row, move.col] = player
        
        # Check for new sequences
        if not move.is_removal:
            new_sequences = self._find_new_sequences(player, move.row, move.col)
            self.completed_sequences[player].extend(new_sequences)
            
            # Check win condition (2 sequences for 2 players)
            if len(self.completed_sequences[player]) >= 2:
                self.game_over = True
                self.winner = player
        
        # Draw new card
        new_card = self.deck.draw()
        if new_card:
            self.hands[player].append(new_card)
        
        # Switch player
        self.current_player = 3 - player
        
        return True
    
    def _find_new_sequences(self, player: int, row: int, col: int) -> List[Set[Tuple[int, int]]]:
        """Find any new sequences formed by placing a chip at (row, col)."""
        new_sequences = []
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal down-right
            (1, -1),  # Diagonal down-left
        ]
        
        existing_sequences = self.completed_sequences[player]
        
        for dr, dc in directions:
            sequence = self._find_sequence_in_direction(player, row, col, dr, dc)
            if sequence and len(sequence) >= self.SEQUENCE_LENGTH:
                # Check it's not already counted (or overlaps by more than 1)
                is_new = True
                for existing in existing_sequences:
                    overlap = sequence & existing
                    if len(overlap) > 1:  # Can share exactly 1 chip
                        is_new = False
                        break
                
                if is_new:
                    new_sequences.append(sequence)
        
        return new_sequences
    
    def _find_sequence_in_direction(self, player: int, row: int, col: int, 
                                     dr: int, dc: int) -> Optional[Set[Tuple[int, int]]]:
        """Find sequence containing (row, col) in given direction."""
        positions = []
        
        # Scan backward
        r, c = row - dr, col - dc
        while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE:
            if self._is_player_chip(player, r, c):
                positions.append((r, c))
                r -= dr
                c -= dc
            else:
                break
        
        positions.reverse()
        positions.append((row, col))
        
        # Scan forward
        r, c = row + dr, col + dc
        while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE:
            if self._is_player_chip(player, r, c):
                positions.append((r, c))
                r += dr
                c += dc
            else:
                break
        
        if len(positions) >= self.SEQUENCE_LENGTH:
            return set(positions[:self.SEQUENCE_LENGTH])
        return None
    
    def _is_player_chip(self, player: int, row: int, col: int) -> bool:
        """Check if position has player's chip or is a free corner."""
        state = self.board[row, col]
        return state == player or state == ChipState.FREE
    
    def get_state_tensor(self, player: int) -> np.ndarray:
        """
        Convert game state to neural network input tensor.
        Shape: (8, 10, 10)
        Channels:
            0: Current player's chips
            1: Opponent's chips  
            2: Free corner spaces
            3: Empty spaces
            4-7: Playable positions for hand cards (up to 4 cards shown)
        """
        opponent = 3 - player
        tensor = np.zeros((8, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        
        # Channel 0: Current player chips
        tensor[0] = (self.board == player).astype(np.float32)
        
        # Channel 1: Opponent chips
        tensor[1] = (self.board == opponent).astype(np.float32)
        
        # Channel 2: Free corners
        tensor[2] = (self.board == ChipState.FREE).astype(np.float32)
        
        # Channel 3: Empty spaces
        tensor[3] = (self.board == ChipState.EMPTY).astype(np.float32)
        
        # Channels 4-7: Playable positions for hand cards
        hand = self.hands[player]
        for i, card in enumerate(hand[:4]):  # Show up to 4 cards
            if card.is_jack():
                continue  # Jacks handled separately
            positions = self.card_to_positions.get(card, [])
            for r, c in positions:
                if self.board[r, c] == ChipState.EMPTY:
                    tensor[4 + i, r, c] = 1.0
        
        return tensor
    
    def clone(self) -> 'SequenceGame':
        """Create a deep copy of the game state."""
        new_game = SequenceGame.__new__(SequenceGame)
        new_game.board_layout = self.board_layout
        new_game.card_to_positions = self.card_to_positions
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.hands = {k: list(v) for k, v in self.hands.items()}
        new_game.deck = Deck.__new__(Deck)
        new_game.deck.cards = list(self.deck.cards)
        new_game.deck.num_decks = self.deck.num_decks
        new_game.discard_piles = {k: list(v) for k, v in self.discard_piles.items()}
        new_game.completed_sequences = {k: [set(s) for s in v] 
                                         for k, v in self.completed_sequences.items()}
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game
    
    def get_result(self, player: int) -> float:
        """Get game result from player's perspective. 1=win, -1=loss, 0=draw/ongoing."""
        if not self.game_over:
            return 0.0
        if self.winner == player:
            return 1.0
        elif self.winner is not None:
            return -1.0
        return 0.0
    
    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {
            ChipState.EMPTY: '.',
            ChipState.PLAYER1: 'X',
            ChipState.PLAYER2: 'O',
            ChipState.FREE: '*'
        }
        lines = []
        for r in range(self.BOARD_SIZE):
            row_str = ' '.join(symbols[self.board[r, c]] for c in range(self.BOARD_SIZE))
            lines.append(row_str)
        return '\n'.join(lines)
