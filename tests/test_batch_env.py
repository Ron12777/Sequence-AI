"""Full-game parity for Rust-owned decks, public history and exact-card actions."""

import random

import numpy as np
import pytest
import sequence_native

from src.fast_game import FastSequenceGame
from src.history_policy import apply_public_move, observe


def initial_deck(game):
    return [
        c.to_int()
        for c in (
            game.deck.cards
            + list(reversed(game.hands[2]))
            + list(reversed(game.hands[1]))
        )
    ]


def unpack(batch, ids, canonical=True):
    raw = batch.observe(ids, canonical=canonical)
    n = len(ids)
    return (
        raw,
        np.frombuffer(raw["board"], np.uint8).reshape(n, 11, 10, 10),
        np.frombuffer(raw["context"], "<f4").reshape(n, 157),
        np.frombuffer(raw["history"], np.uint8).reshape(n, 128, 4),
        np.frombuffer(raw["actions"], "<i2").reshape(n, raw["width"]),
    )


def assert_snapshot(batch, index, game, events):
    actual = batch.debug_snapshot(index)
    np.testing.assert_array_equal(
        np.frombuffer(actual["board"], np.int8).reshape(10, 10), game.board
    )
    assert actual["hands"] == [[c.to_int() for c in game.hands[p]] for p in (1, 2)]
    assert actual["discards"] == [
        [c.to_int() for c in game.discard_piles[p]] for p in (1, 2)
    ]
    assert actual["deck"] == [c.to_int() for c in game.deck.cards]
    assert actual["events"] == events
    assert actual["player"] == game.current_player
    assert actual["winner"] == (game.winner or 0)
    assert actual["done"] == (game.game_over or not game.get_legal_moves())
    assert actual["sequences"] == game.c_state.get_sequences()
    assert actual["legal"] == sorted(
        {m.card.to_int() * 100 + m.row * 10 + m.col for m in game.get_legal_moves()}
    )


@pytest.mark.parametrize("first_seed", [0, 100, 200, 300])
def test_entire_random_games_match_python_rules_cards_and_observations(first_seed):
    references = []
    for seed in range(first_seed, first_seed + 8):
        random.seed(seed)
        references.append(FastSequenceGame())
    batch = sequence_native.BatchEnv.from_decks([initial_deck(g) for g in references])
    events = [[] for _ in references]
    rng = random.Random(first_seed + 9000)
    kinds = set()
    for _ in range(110):
        ids = batch.active_ids()
        if not ids:
            break
        for canonical in (True, False):
            raw, boards, contexts, histories, actions = unpack(batch, ids, canonical)
            for row, index in enumerate(ids):
                game = references[index]
                original_hand = game.hands[game.current_player]
                if canonical:
                    game.hands[game.current_player] = sorted(
                        original_hand, key=lambda c: c.to_int()
                    )
                expected = observe(game, events[index])
                game.hands[game.current_player] = original_hand
                for actual, wanted in zip(
                    (boards[row], contexts[row], histories[row]), expected
                ):
                    np.testing.assert_array_equal(actual, wanted)
                legal = sorted(
                    {
                        m.card.to_int() * 100 + m.row * 10 + m.col
                        for m in game.get_legal_moves()
                    }
                )
                assert actions[row, : raw["counts"][row]].tolist() == legal
                assert (actions[row, raw["counts"][row] :] == -1).all()
                assert raw["players"][row] == game.current_player
        chosen = []
        for index in ids:
            game = references[index]
            move = rng.choice(game.get_legal_moves())
            chosen.append(move.card.to_int() * 100 + move.row * 10 + move.col)
            apply_public_move(game, events[index], move)
            kinds.update(event[3] for event in events[index])
        result = batch.step(ids, chosen)
        assert result["active"] == batch.active_ids()
        for index in ids:
            assert_snapshot(batch, index, references[index], events[index])
    assert not batch.active_ids(), "No truncated games allowed in parity test"
    assert (
        0 in kinds and 1 in kinds and 2 in kinds
    )  # placement, removal, dead-card skips
    for player in (1, 2):
        assert batch.results(player) == [g.get_result(player) for g in references]


def test_parallel_and_serial_batches_identical_and_subset_order_preserved():
    serial = sequence_native.BatchEnv(128, seed=175, threads=1)
    parallel = sequence_native.BatchEnv(128, seed=175, threads=4)
    for _ in range(50):
        ids = serial.active_ids()[::-1]
        assert ids == parallel.active_ids()[::-1]
        left = serial.observe(ids)
        right = parallel.observe(ids)
        assert left == right
        if not ids:
            break
        actions = np.frombuffer(left["actions"], "<i2").reshape(len(ids), left["width"])
        selected = [int(row[0]) for row in actions]
        assert serial.step(ids, selected) == parallel.step(ids, selected)
    for i in range(128):
        assert serial.debug_snapshot(i) == parallel.debug_snapshot(i)
    assert serial.observe([])["width"] == 0


def test_invalid_action_batch_is_atomic_and_duplicate_indices_rejected():
    batch = sequence_native.BatchEnv(2, seed=4)
    before = [batch.debug_snapshot(i) for i in range(2)]
    valid = before[0]["legal"][0]
    with pytest.raises(ValueError, match="Illegal exact action"):
        batch.step([0, 1], [valid, -1])
    assert [batch.debug_snapshot(i) for i in range(2)] == before
    with pytest.raises(ValueError, match="Duplicate"):
        batch.step([0, 0], [valid, valid])
    with pytest.raises(ValueError, match="length"):
        batch.step([0], [])
    with pytest.raises(ValueError, match="range"):
        batch.observe([2])


def test_explicit_deck_validation_and_reproducible_seed():
    with pytest.raises(ValueError, match="104"):
        sequence_native.BatchEnv.from_decks([[0] * 103])
    with pytest.raises(ValueError, match="exactly two"):
        sequence_native.BatchEnv.from_decks([[0] * 104])
    first, second = sequence_native.BatchEnv(2, 1234), sequence_native.BatchEnv(3, 1234)
    assert first.debug_snapshot(0) == second.debug_snapshot(0)
    assert first.debug_snapshot(1) == second.debug_snapshot(1)


def test_search_states_match_public_python_inputs_and_ignore_hidden_identities():
    from src.belief import unseen_cards

    random.seed(980)
    game = FastSequenceGame()
    original = initial_deck(game)
    changed = original.copy()
    # Own player-one cards are the final seven; permute hidden deck + other hand.
    changed[:97] = reversed(changed[:97])
    batch = sequence_native.BatchEnv.from_decks([original, changed])
    first, second = batch.search_states([0, 1])
    assert first == second
    assert first["board"] == game.c_state.get_tensor()
    assert first["hand"] == [c.to_int() for c in game.hands[1]]
    assert first["unseen"] == unseen_cards(game, 1)
    assert first["legal"] == [
        m.card.to_int() * 100 + m.row * 10 + m.col for m in game.get_legal_moves()
    ]
    events = []
    for _ in range(20):
        move = game.get_legal_moves()[0]
        batch.step([0], [move.card.to_int() * 100 + move.row * 10 + move.col])
        apply_public_move(game, events, move)
        state = batch.search_states([0])[0]
        assert state["board"] == game.c_state.get_tensor()
        assert state["sequences"] == game.c_state.get_sequences()
        assert state["unseen"] == unseen_cards(game, game.current_player)
        assert state["hand"] == [c.to_int() for c in game.hands[game.current_player]]
