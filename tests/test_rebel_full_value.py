import copy
import random

import numpy as np
import pytest
import torch

from src.fast_game import FastSequenceGame
from src.model import SequenceNet
from src.public_belief_cfr import PublicBeliefCFR
from src.rebel_full_game import FullSequenceBeliefGame, action_key, counts_to_cards
from src.rebel_full_value import BeliefResidual, FullBeliefValue


def setup(seed=72, *, empty_deck=False):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    game = FastSequenceGame()
    if empty_deck:
        game.discard_piles[1].extend(game.deck.cards)
        game.deck.cards.clear()
    adapter = FullSequenceBeliefGame(game, particles=4, seed=seed, depth=1)
    model = FullBeliefValue(
        SequenceNet(num_filters=8, num_res_blocks=1), BeliefResidual(16)
    )
    solver = PublicBeliefCFR(adapter.root, adapter.prior, lambda node, belief: 0.0)
    return game, adapter, model, solver


def test_vectorized_features_match_individual_and_own_public_range_changes_input():
    _, adapter, value, solver = setup()
    node = adapter.root
    belief = solver.belief()
    features, base = value.examples(adapter, node, belief)
    hands = adapter.hand_counts(node)
    rows = []
    for player in (0, 1):
        weights = (adapter.prior if player == 0 else adapter.prior.T) * belief.reaches[
            1 - player
        ][None]
        for index, own in enumerate(hands[player]):
            rows.append(
                value._features(
                    adapter,
                    adapter.state(node),
                    hands,
                    belief,
                    player,
                    own,
                    weights[index],
                )
            )
    np.testing.assert_allclose(features, np.stack(rows), atol=1e-7)
    assert base.shape == (sum(map(len, hands)),)
    changed = solver.belief(
        (np.arange(1, len(hands[0]) + 1, dtype=float), np.ones(len(hands[1])))
    )
    changed_features, _ = value.examples(adapter, node, changed)
    assert not np.array_equal(features[0, 912:1016], changed_features[0, 912:1016])
    np.testing.assert_array_equal(features[0, 756:808], changed_features[0, 756:808])


def test_private_query_does_not_change_public_ranges_or_insert_actual_hand():
    game, adapter, value, solver = setup()
    belief = solver.belief()
    before = (
        adapter.prior.copy(),
        tuple(x.copy() for x in adapter.hands),
        tuple(x.copy() for x in belief.ranges),
    )
    move = game.get_legal_moves()[0]
    actual = list(game.hands[1])
    altered = actual.copy()
    index = next(i for i, card in enumerate(actual) if card != move.card)
    counts = np.bincount([c.to_int() for c in actual], minlength=52)
    replacement = next(
        i for i in range(52) if counts[i] < 2 and i != actual[index].to_int()
    )
    from src.cards import Card

    altered[index] = Card.from_int(replacement)
    first, _ = value.query_example(adapter, action_key(move), belief, actual)
    second, _ = value.query_example(adapter, action_key(move), belief, altered)
    assert not np.array_equal(first[756:808], second[756:808])
    np.testing.assert_array_equal(first[912:1120], second[912:1120])
    np.testing.assert_array_equal(adapter.prior, before[0])
    for after, expected in zip(adapter.hands, before[1]):
        np.testing.assert_array_equal(after, expected)
    for after, expected in zip(belief.ranges, before[2]):
        np.testing.assert_array_equal(after, expected)


@pytest.mark.parametrize("empty_deck", [False, True])
def test_pending_draw_feature_and_empty_query_support_fallback(empty_deck):
    game, adapter, value, solver = setup(empty_deck=empty_deck)
    belief = solver.belief()
    action = action_key(game.get_legal_moves()[0])
    features, base = value.query_example(adapter, action, belief, game.hands[1])
    assert features[1120] == int(not empty_deck)
    assert features[1121] == 0
    assert features[756:808].sum() == 3  # Six post-play cards, encoded /2.
    result = value.evaluate_query_path(
        adapter,
        (action,),
        belief,
        game.hands[1],
        opponent_weights=np.zeros(len(adapter.hands[1])),
    )
    assert result == float(base)
    assert value.query_support_fallbacks == 1


def test_values_bounded_reconciled_batch_equal_and_optimizer_invalidates_cache():
    _, adapter, value, solver = setup()
    belief = solver.belief()
    pairs = [
        (adapter.root, belief),
        (next(iter(adapter.root.children.values())), belief),
    ]
    before = [value.evaluate_leaf(adapter, node, b) for node, b in pairs]
    batches_before = value.teacher_batches
    value._leaf_cache.clear()
    batched = value.evaluate_many(adapter, pairs)
    assert value.teacher_batches == batches_before
    for (node, b), expected in zip(pairs, before):
        actual = batched[node.key]
        for a, e in zip(
            (actual.player0, actual.player1), (expected.player0, expected.player1)
        ):
            assert np.isfinite(a).all() and (np.abs(a) <= 1).all()
            np.testing.assert_allclose(a, e, atol=1e-7)
        assert abs(b.ranges[0] @ actual.player0 + b.ranges[1] @ actual.player1) < 1e-7
    with torch.no_grad():
        value.residual.layers[-1].bias.add_(0.7)
    after = value.evaluate_leaf(adapter, adapter.root, belief)
    assert not np.allclose(after.player0, before[0].player0)


def test_two_public_move_query_keeps_public_ranges_and_defers_both_draws():
    game, adapter, value, solver = setup()
    first = action_key(game.get_legal_moves()[0])
    state, _ = adapter.query_after_action(first, game.hands[1])
    state.hands[2] = counts_to_cards(adapter.hands[1][0])
    second = action_key(state.get_legal_moves()[0])
    belief = solver.belief()
    frozen = copy.deepcopy(belief)
    weights = np.zeros(len(adapter.hands[1]))
    weights[0] = 1
    context = value._query_context(
        adapter, (first, second), belief, game.hands[1], weights
    )
    features, _ = value._query_encoding(adapter, (first, second), context)
    np.testing.assert_array_equal(features[1120:1122], [1, 1])
    result = value.evaluate_query_path(
        adapter, (first, second), belief, game.hands[1], weights
    )
    assert np.isfinite(result) and -1 <= result <= 1
    for a, b in zip(belief.ranges, frozen.ranges):
        np.testing.assert_array_equal(a, b)
