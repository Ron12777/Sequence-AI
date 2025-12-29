"""
Unit tests for Sequence game logic.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game import SequenceGame, Move, ChipState
from src.cards import Card, Deck, Rank, Suit


class TestCards:
    """Tests for card handling."""
    
    def test_card_from_string(self):
        card = Card.from_string("AS")
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES
        
        card = Card.from_string("10H")
        assert card.rank == Rank.TEN
        assert card.suit == Suit.HEARTS
        
        card = Card.from_string("KD")
        assert card.rank == Rank.KING
        assert card.suit == Suit.DIAMONDS
    
    def test_card_to_string(self):
        card = Card(rank=Rank.QUEEN, suit=Suit.CLUBS)
        assert str(card) == "QC"
        
        card = Card(rank=Rank.TEN, suit=Suit.SPADES)
        assert str(card) == "10S"
    
    def test_jack_types(self):
        # Two-eyed jacks (wild)
        jc = Card(rank=Rank.JACK, suit=Suit.CLUBS)
        jd = Card(rank=Rank.JACK, suit=Suit.DIAMONDS)
        assert jc.is_two_eyed_jack()
        assert jd.is_two_eyed_jack()
        assert not jc.is_one_eyed_jack()
        
        # One-eyed jacks (remove)
        jh = Card(rank=Rank.JACK, suit=Suit.HEARTS)
        js = Card(rank=Rank.JACK, suit=Suit.SPADES)
        assert jh.is_one_eyed_jack()
        assert js.is_one_eyed_jack()
        assert not jh.is_two_eyed_jack()
    
    def test_deck_size(self):
        deck = Deck(num_decks=2)
        # 2 decks * 52 cards = 104 cards
        assert len(deck) == 104
    
    def test_deck_draw(self):
        deck = Deck(num_decks=1)
        initial_size = len(deck)
        
        card = deck.draw()
        assert card is not None
        assert len(deck) == initial_size - 1


class TestGame:
    """Tests for game logic."""
    
    def test_game_initialization(self):
        game = SequenceGame()
        game.reset()
        
        # Check board size
        assert game.board.shape == (10, 10)
        
        # Check corners are free
        assert game.board[0, 0] == ChipState.FREE
        assert game.board[0, 9] == ChipState.FREE
        assert game.board[9, 0] == ChipState.FREE
        assert game.board[9, 9] == ChipState.FREE
        
        # Check hands are dealt
        assert len(game.hands[1]) == 7  # 2 players = 7 cards each
        assert len(game.hands[2]) == 7
    
    def test_legal_moves_basic(self):
        game = SequenceGame()
        game.reset()
        
        moves = game.get_legal_moves(1)
        assert len(moves) > 0
        
        # Each move should have a card from player's hand
        for move in moves:
            assert move.card in game.hands[1] or move.card.is_jack()
    
    def test_make_move(self):
        game = SequenceGame()
        game.reset()
        
        moves = game.get_legal_moves(1)
        assert len(moves) > 0
        
        move = moves[0]
        initial_hand_size = len(game.hands[1])
        
        result = game.make_move(move)
        assert result is True
        
        # Chip should be placed (unless removal)
        if not move.is_removal:
            assert game.board[move.row, move.col] == 1
        
        # Hand size should stay same (drew new card)
        assert len(game.hands[1]) == initial_hand_size
        
        # Should be player 2's turn
        assert game.current_player == 2
    
    def test_sequence_detection_horizontal(self):
        game = SequenceGame()
        game.reset()
        
        # Manually place 5 in a row horizontally (row 1, cols 1-5)
        for c in range(1, 6):
            game.board[1, c] = ChipState.PLAYER1
        
        # Check sequence detection
        sequences = game._find_new_sequences(1, 1, 5)
        assert len(sequences) >= 1
        assert len(list(sequences[0])) == 5
    
    def test_sequence_detection_with_corner(self):
        game = SequenceGame()
        game.reset()
        
        # Place 4 chips near a corner to complete sequence
        # Corner at (0, 0) counts as player's chip
        game.board[0, 1] = ChipState.PLAYER1
        game.board[0, 2] = ChipState.PLAYER1
        game.board[0, 3] = ChipState.PLAYER1
        game.board[0, 4] = ChipState.PLAYER1
        
        # Check that corner + 4 chips = sequence
        sequences = game._find_new_sequences(1, 0, 4)
        assert len(sequences) >= 1
    
    def test_clone(self):
        game = SequenceGame()
        game.reset()
        
        clone = game.clone()
        
        # Modify original
        moves = game.get_legal_moves(1)
        if moves:
            game.make_move(moves[0])
        
        # Clone should be unaffected
        assert clone.current_player == 1
    
    def test_state_tensor(self):
        game = SequenceGame()
        game.reset()
        
        tensor = game.get_state_tensor(1)
        
        # Check shape
        assert tensor.shape == (8, 10, 10)
        
        # Check data type
        assert tensor.dtype.name.startswith('float')


class TestIntegration:
    """Integration tests."""
    
    def test_full_game_random(self):
        """Play a random game to completion."""
        import random
        
        game = SequenceGame()
        game.reset()
        
        max_moves = 200
        moves_made = 0
        
        while not game.game_over and moves_made < max_moves:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            
            move = random.choice(legal_moves)
            game.make_move(move)
            moves_made += 1
        
        # Game should have finished or reached move limit
        assert moves_made > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
