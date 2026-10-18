from collections import Counter
from itertools import combinations

import numpy as np
import pytest

from src.public_belief_cfr import PublicBeliefCFR
from src.rebel_sequence_game import PASS, SequenceBeliefGame


@pytest.fixture(scope="module")
def restricted():
    return SequenceBeliefGame(0)


def test_correlated_prior_matches_enumeration_of_physical_cards(restricted):
    game = restricted
    physical = tuple((identity, copy) for identity in range(6) for copy in range(2))
    ordered_hand_pairs = Counter()
    for first in combinations(physical, 2):
        remaining = tuple(card for card in physical if card not in first)
        for second in combinations(remaining, 2):
            ordered_hand_pairs[(tuple(sorted(c[0] for c in first)),
                                tuple(sorted(c[0] for c in second)))] += 1
    expected = np.array([[ordered_hand_pairs[(a, b)] for b in game.types] for a in game.types])
    np.testing.assert_allclose(game.prior, expected / expected.sum(), rtol=0, atol=1e-16)
    assert expected.sum() == 66 * 45
    assert game.prior.sum() == pytest.approx(1)
    np.testing.assert_array_equal(game.prior, game.prior.T)


def test_two_physical_copies_allow_shared_identity_but_exclude_third_copy(restricted):
    game = restricted
    both = game.types.index((0, 1))
    double = game.types.index((0, 0))
    assert game.prior[both, both] > 0
    assert game.prior[double, both] == 0
    assert game.prior[double, double] == 0


def test_features_do_not_read_actual_hidden_hands(restricted):
    game = restricted
    solver = PublicBeliefCFR(game.root, game.prior)
    belief = solver.belief()
    original = game.features(game.root, belief)
    state = game.games[game.root.key]
    try:
        state.hands = {1: [game.cards[0], game.cards[0]], 2: [game.cards[1], game.cards[2]]}
        np.testing.assert_array_equal(game.features(game.root, belief), original)
        state.hands = {1: [game.cards[4]], 2: [game.cards[3], game.cards[3]]}
        np.testing.assert_array_equal(game.features(game.root, belief), original)
    finally:
        state.hands = {1: [], 2: []}
    # Reach factors keep off-policy hand-value queries distinguishable even
    # when own-zero reach erased their rows from the normalized joint posterior.
    first = (np.array([1.0] + [0.0] * 20), np.ones(21))
    second = (first[0], np.array([2.0] + [1.0] * 20))
    assert not np.array_equal(game.features(game.root, solver.belief(first)),
                              game.features(game.root, solver.belief(second)))


def test_root_public_transitions_match_real_sequence_rules_for_every_legal_hand(restricted):
    game = restricted
    root = game.root
    initial = game.games[root.key]
    assert len(root.children) == root.legal_mask.shape[1]
    for hand_id, identities in enumerate(game.types):
        state = initial.clone()
        state.hands = {1: [game.cards[i] for i in identities], 2: []}
        legal = {(m.card.to_int(), m.row, m.col, m.is_removal): m for m in state.get_legal_moves()}
        expected_actions = set(legal) or {PASS}
        actual_actions = {a for index, a in enumerate(root.children) if root.legal_mask[hand_id, index]}
        assert actual_actions == expected_actions
        for action, move in legal.items():
            transitioned = state.clone()
            assert transitioned.make_move(move)
            public = game.games[root.children[action].key]
            np.testing.assert_array_equal(transitioned.board, public.board)
            assert transitioned.current_player == public.current_player
            assert transitioned.game_over == public.game_over
            assert transitioned.completed_sequences == public.completed_sequences
            assert public.hands == {1: [], 2: []}


def test_complete_public_tree_has_four_turn_horizon_and_no_stored_actual_hands(restricted):
    game = restricted
    for key, node in game.nodes.items():
        assert len(key[1]) <= 4
        assert game.games[key].hands == {1: [], 2: []}
        if node.actor is None:
            assert game.games[key].game_over or len(key[1]) == 4
            assert node.payoff in (-1, 0, 1)
        else:
            assert node.actor == game.games[key].current_player - 1
