"""
Focused tests for the C training backend.
"""

import numpy as np
import pytest

from src.cards import Card, Rank, Suit
from src.game import SequenceGame


c_sequence = pytest.importorskip("sequence_native")


def setup_c_layout():
    game = SequenceGame()
    counts = {}
    for r, row in enumerate(game.board_layout):
        for c, card in enumerate(row):
            if card is None:
                continue
            card_int = card.to_int()
            idx = counts.get(card_int, 0)
            c_sequence.setup_layout(card_int, idx, r, c)
            counts[card_int] = idx + 1


def test_c_mcts_leaf_tensor_includes_hand_channels():
    setup_c_layout()
    game = SequenceGame()
    game.reset()
    game.hands[1] = [
        Card(rank=Rank.ACE, suit=Suit.SPADES),
        Card(rank=Rank.JACK, suit=Suit.CLUBS),
    ]

    mcts = c_sequence.CMCTS()
    mcts.reset(game.board.tobytes(), [card.to_int() for card in game.hands[1]], 1)

    _, tensor_bytes, _, is_terminal = mcts.select_leaf()
    assert is_terminal == 0

    tensor = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(11, 10, 10)
    expected_positions = game.card_to_positions[game.hands[1][0]]

    assert tensor[4].sum() > 0
    assert all(tensor[4, row, col] == 1.0 for row, col in expected_positions)
    assert tensor[5].sum() > 0.0


def test_c_mcts_training_policy_is_normalized_after_visits():
    setup_c_layout()
    game = SequenceGame()
    game.reset()
    game.hands[1] = [Card(rank=Rank.JACK, suit=Suit.CLUBS)]

    mcts = c_sequence.CMCTS()
    mcts.reset(game.board.tobytes(), [card.to_int() for card in game.hands[1]], 1)

    leaf, _, value, is_terminal = mcts.select_leaf()
    assert is_terminal == 0
    policy = np.ones(100, dtype=np.float32) / 100.0
    mcts.backpropagate(leaf, policy.tobytes(), value)

    leaf, _, value, is_terminal = mcts.select_leaf()
    mcts.backpropagate(leaf, policy.tobytes() if not is_terminal else b"", value)

    _, _, _, _, policy_bytes = mcts.get_action(0.0)
    visit_policy = np.frombuffer(policy_bytes, dtype=np.float32)

    assert np.isclose(visit_policy.sum(), 1.0)
