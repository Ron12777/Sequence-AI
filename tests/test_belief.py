"""Public-information boundaries, resampling, and value backup regressions."""

import random
from collections import Counter
import numpy as np
import sequence_native
from src.belief import unseen_cards
from src.fast_game import FastSequenceGame


def initialize(game, seed=17):
    search = sequence_native.EnsembleMCTS(seed)
    p = game.current_player
    search.reset(
        game.board.tobytes(),
        [c.to_int() for c in game.hands[p]],
        p,
        [
            [sorted(r * 10 + c for r, c in seq) for seq in game.completed_sequences[q]]
            for q in (1, 2)
        ],
        unseen_cards(game, p),
        len(game.hands[3 - p]),
    )
    return search


def test_public_discards_reconstruct_hidden_multiset_through_a_game():
    random.seed(71)
    game = FastSequenceGame()
    for _ in range(150):
        for p in (1, 2):
            assert Counter(unseen_cards(game, p)) == Counter(
                c.to_int() for c in game.deck.cards + game.hands[3 - p]
            )
        moves = game.get_legal_moves()
        if game.game_over or not moves:
            break
        assert game.make_move(random.choice(moves))


def test_search_does_not_read_actual_hidden_partition_or_order():
    random.seed(19)
    game = FastSequenceGame()
    altered = game.clone()
    hidden = altered.hands[2] + altered.deck.cards
    random.shuffle(hidden)
    altered.hands[2], altered.deck.cards = hidden[:7], hidden[7:]
    # The public helper works even if access to the actual deck is forbidden.
    altered.deck = None
    assert unseen_cards(game, 1) == unseen_cards(altered, 1)
    searches = [initialize(g) for g in (game, altered)]
    policy = np.full(100, 0.01, np.float32).tobytes()
    for _ in range(48):
        leaves = [s.select_leaf() for s in searches]
        assert leaves[0][1:] == leaves[1][1:]
        for search, (token, data, value, terminal) in zip(searches, leaves):
            search.backpropagate(token, b"" if terminal else policy, value)
    assert searches[0].get_action(0) == searches[1].get_action(0)


def test_four_deals_share_known_hand_but_sample_different_opponent_hands():
    random.seed(37)
    game = FastSequenceGame()
    search = initialize(game)
    policy = np.full(100, 0.01, np.float32).tobytes()
    observations = []
    for _ in range(8):
        token, data, value, terminal = search.select_leaf()
        assert not terminal
        observations.append(data)
        search.backpropagate(token, policy, value)
    assert len(set(observations[:4])) == 1  # Four roots see the same own hand.
    assert len(set(observations[4:])) == 4  # Next leaves see independent hidden hands.
    result = search.get_action(0)
    assert np.isclose(np.frombuffer(result[4], np.float32).sum(), 1)
