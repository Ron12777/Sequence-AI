"""Contract, safety and full-game parity checks for the Rust backend."""

import random
import numpy as np
import pytest
import sequence_native as native
from src.cards import Card
from src.game import SequenceGame, Move
from src.fast_game import FastSequenceGame


def move_key(move):
    return move.card.to_int(), move.row, move.col, move.is_removal


def search_for(game, seed=1):
    search = native.CMCTS(seed)
    p = game.current_player
    sequences = [
        [sorted(r * 10 + c for r, c in seq) for seq in game.completed_sequences[player]]
        for player in (1, 2)
    ]
    unseen = [c.to_int() for c in game.deck.cards + game.hands[3 - p]]
    search.reset(
        game.board.tobytes(),
        [c.to_int() for c in game.hands[p]],
        p,
        sequences,
        unseen,
        len(game.hands[3 - p]),
    )
    return search


@pytest.mark.parametrize("seed", range(12))
def test_full_game_parity(seed):
    random.seed(seed)
    python = SequenceGame()
    python.reset()
    rust = FastSequenceGame()
    rust.hands = {p: list(hand) for p, hand in python.hands.items()}
    rust.deck.cards = list(python.deck.cards)
    rng = random.Random(seed)
    for _ in range(200):
        assert rust.current_player == python.current_player
        assert rust.game_over == python.game_over
        assert rust.winner == python.winner
        assert rust.completed_sequences == python.completed_sequences
        assert rust.hands == python.hands
        assert rust.deck.cards == python.deck.cards
        np.testing.assert_array_equal(rust.board, python.board)
        for player in (1, 2):
            np.testing.assert_array_equal(
                rust.get_state_tensor(player), python.get_state_tensor(player)
            )
        moves = python.get_legal_moves()
        assert set(map(move_key, moves)) == set(map(move_key, rust.get_legal_moves()))
        if not moves:
            break
        move = rng.choice(moves)
        assert python.make_move(move)
        assert rust.make_move(move)


def test_two_separate_sequences_and_protection():
    game = native.GameState()
    # Player 1 builds a horizontal sequence. Player 2 plays isolated fillers.
    for col in range(1, 6):
        assert game.apply_move(10, 2, col, 0)
        assert game.apply_move(10, 8, col * 2 % 9, 0)
    assert len(game.get_sequences()[0]) == 1
    assert not game.apply_move(36, 2, 1, 1)  # wrong target for player 1
    assert game.apply_move(10, 4, 1, 0)
    assert not game.apply_move(36, 2, 1, 1)  # protected from player 2
    for col in range(2, 6):
        assert game.apply_move(10, 7, col * 2 % 9, 0)
        assert game.apply_move(10, 4, col, 0)
    assert game.get_info()[1:] == (1, True)


@pytest.mark.parametrize(
    "args",
    [
        (-1, 0, 1, 0),
        (52, 0, 1, 0),
        (10, -1, 0, 0),
        (10, 10, 0, 0),
        (10, 0, 1000, 0),
        (10, 0, 0, 0),
        (0, 4, 4, 0),
    ],
)
def test_invalid_move_does_not_mutate(args):
    game = native.GameState()
    before = game.get_tensor(), game.get_info()
    assert not game.apply_move(*args)
    assert (game.get_tensor(), game.get_info()) == before


def test_search_contract_and_stale_tokens():
    game = SequenceGame()
    game.reset()
    search = search_for(game)
    token, data, value, terminal = search.select_leaf()
    assert not terminal
    np.testing.assert_array_equal(
        np.frombuffer(data, np.float32).reshape(11, 10, 10), game.get_state_tensor(1)
    )
    with pytest.raises(RuntimeError):
        search.select_leaf()
    with pytest.raises(ValueError):
        search.backpropagate(token, b"", 0)
    policy = np.full(100, 0.01, np.float32).tobytes()
    search.backpropagate(token, policy, value)
    with pytest.raises(ValueError):
        search.backpropagate(token, policy, value)
    for _ in range(15):
        token, _, value, terminal = search.select_leaf()
        search.backpropagate(token, b"" if terminal else policy, value)
    for temp in (0, 0.01, 0.1, 1):
        card, r, c, removal, data = search.get_action(temp)
        assert (card, r, c, removal) in set(map(move_key, game.get_legal_moves()))
        pi = np.frombuffer(data, np.float32)
        assert np.isfinite(pi).all() and np.isclose(pi.sum(), 1)
    with pytest.raises(ValueError):
        search.get_action(float("nan"))


def test_invalid_reset_and_uninitialized_search():
    search = native.CMCTS()
    with pytest.raises(RuntimeError):
        search.select_leaf()
    for board, hand, player in [
        (b"", [], 1),
        (bytes(100), [], 1),
        (native.GameState().get_tensor(), [255], 1),
        (native.GameState().get_tensor(), [], 3),
    ]:
        with pytest.raises(ValueError):
            search.reset(board, hand, player)


def test_long_line_can_share_one_chip():
    game = SequenceGame()
    game.reset()
    game.board[3, :9] = 1
    game.completed_sequences[1] = [{(3, c) for c in range(5)}]
    assert game._find_new_sequences(1, 3, 8) == [{(3, c) for c in range(4, 9)}]


def test_fast_game_rejects_missing_hand_card_before_mutating():
    game = FastSequenceGame()
    game.hands[1] = []
    board = game.board.copy()
    assert not game.make_move(Move(Card.from_int(10), 1, 1))
    np.testing.assert_array_equal(game.board, board)


def test_search_recognizes_winning_extension_of_protected_sequence():
    game = SequenceGame()
    game.reset()
    game.board[3, :8] = 1
    game.completed_sequences[1] = [{(3, c) for c in range(5)}]
    game.hands[1] = [Card.from_int(10)]
    game.hands[2] = []
    game.deck.cards = []
    search = search_for(game)
    token, _, _, _ = search.select_leaf()
    policy = np.zeros(100, np.float32)
    policy[38] = 1
    search.backpropagate(token, policy.tobytes(), 0.0)
    token, data, value, terminal = search.select_leaf()
    assert terminal and data is None and value == -1.0
    search.backpropagate(token, b"", value)
    assert search.get_action(0)[:4] == (10, 3, 8, False)


def test_search_rejects_impossible_card_pool_and_sequence_history():
    game = SequenceGame()
    game.reset()
    search = native.CMCTS()
    with pytest.raises(ValueError, match="two copies"):
        search.reset(game.board.tobytes(), [10], 1, None, [10] * 8, 7)
    game.board[2, :5] = 1
    with pytest.raises(ValueError, match="sequence history"):
        search.reset(game.board.tobytes(), [10], 1, [[[20, 20, 21, 22, 23]], []], [], 0)
