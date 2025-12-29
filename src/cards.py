"""
Card and deck management for Sequence game.
"""
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13


# Card notation mapping
RANK_SYMBOLS = {
    'A': Rank.ACE, '2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR,
    '5': Rank.FIVE, '6': Rank.SIX, '7': Rank.SEVEN, '8': Rank.EIGHT,
    '9': Rank.NINE, '10': Rank.TEN, 'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING
}

SUIT_SYMBOLS = {
    'C': Suit.CLUBS, 'D': Suit.DIAMONDS, 'H': Suit.HEARTS, 'S': Suit.SPADES
}

RANK_TO_SYMBOL = {v: k for k, v in RANK_SYMBOLS.items()}
SUIT_TO_SYMBOL = {v: k for k, v in SUIT_SYMBOLS.items()}


@dataclass(frozen=True)
class Card:
    """Represents a playing card."""
    rank: Rank
    suit: Suit
    
    def __str__(self) -> str:
        return f"{RANK_TO_SYMBOL[self.rank]}{SUIT_TO_SYMBOL[self.suit]}"
    
    def __repr__(self) -> str:
        return str(self)
    
    @classmethod
    def from_string(cls, s: str) -> 'Card':
        """Parse card from string like '10C', 'AS', 'KH'."""
        s = s.strip().upper()
        if s == 'FREE':
            return None  # Corner spaces
        
        # Handle 10 specially (two-digit rank)
        if s.startswith('10'):
            rank_str = '10'
            suit_str = s[2:]
        else:
            rank_str = s[:-1]
            suit_str = s[-1]
        
        rank = RANK_SYMBOLS.get(rank_str)
        suit = SUIT_SYMBOLS.get(suit_str)
        
        if rank is None or suit is None:
            raise ValueError(f"Invalid card string: {s}")
        
        return cls(rank=rank, suit=suit)
    
    def is_red_jack(self) -> bool:
        """Red Jacks (Hearts, Diamonds) are wild cards."""
        return self.rank == Rank.JACK and self.suit in (Suit.HEARTS, Suit.DIAMONDS)
    
    def is_black_jack(self) -> bool:
        """Black Jacks (Clubs, Spades) remove opponent chips."""
        return self.rank == Rank.JACK and self.suit in (Suit.CLUBS, Suit.SPADES)
    
    def is_jack(self) -> bool:
        return self.rank == Rank.JACK
    
    def to_int(self) -> int:
        """Convert card to unique integer (0-51)."""
        return self.suit * 13 + (self.rank - 1)
    
    @classmethod
    def from_int(cls, n: int) -> 'Card':
        """Create card from integer (0-51)."""
        suit = Suit(n // 13)
        rank = Rank((n % 13) + 1)
        return cls(rank=rank, suit=suit)


class Deck:
    """Manages the deck of cards (2 standard decks for Sequence)."""
    
    def __init__(self, num_decks: int = 2):
        self.num_decks = num_decks
        self.cards: List[Card] = []
        self.reset()
    
    def reset(self):
        """Reset deck with all cards, excluding Jacks (used as special cards in hand only)."""
        self.cards = []
        for _ in range(self.num_decks):
            for suit in Suit:
                for rank in Rank:
                    # Include all cards - Jacks are dealt but not on the board
                    self.cards.append(Card(rank=rank, suit=suit))
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def draw(self) -> Optional[Card]:
        """Draw a card from the deck."""
        if self.cards:
            return self.cards.pop()
        return None
    
    def draw_many(self, n: int) -> List[Card]:
        """Draw multiple cards."""
        return [self.draw() for _ in range(n) if self.cards]
    
    def __len__(self) -> int:
        return len(self.cards)
    
    def cards_remaining(self) -> int:
        return len(self.cards)


def get_cards_per_player(num_players: int) -> int:
    """Get number of cards to deal based on player count."""
    if num_players == 2:
        return 7
    elif num_players <= 4:
        return 6
    elif num_players == 6:
        return 5
    elif num_players <= 9:
        return 4
    else:
        return 3
