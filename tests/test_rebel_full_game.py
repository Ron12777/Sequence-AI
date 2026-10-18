import numpy as np
import pytest

from src.cards import Card
from src.game import SequenceGame
from src.public_belief_cfr import LeafValues, PublicBeliefCFR
from src.rebel_full_game import (
    PASS,
    FullSequenceBeliefGame,
    action_key,
    card_counts,
    counts_to_cards,
    public_copy,
)


class HiddenIdentities:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        raise AssertionError("Hidden card identities were inspected")

    def __getitem__(self, key):
        raise AssertionError("Hidden card identities were inspected")


def test_public_bank_never_reads_either_actual_hand_or_deck_identities():
    game = SequenceGame()
    game.reset()
    reference = FullSequenceBeliefGame(game, particles=8, seed=123)
    game.hands = {p: HiddenIdentities(len(cards)) for p, cards in game.hands.items()}
    game.deck.cards = HiddenIdentities(len(game.deck.cards))
    hidden = FullSequenceBeliefGame(game, particles=8, seed=123)
    np.testing.assert_array_equal(reference.prior, hidden.prior)
    for a, b in zip(reference.hands, hidden.hands):
        np.testing.assert_array_equal(a, b)
    assert list(reference.root.children) == list(hidden.root.children)
    assert all(state.hands == {1: [], 2: []} and state.deck.cards == []
               for state in hidden.states.values())


def test_public_discards_include_played_and_skipped_cards_and_exhaustion():
    game = SequenceGame()
    game.reset()
    first = game.deck.cards.pop()
    second = game.deck.cards.pop()
    game.discard_piles[1].append(first)
    game.discard_piles[2].append(second)
    adapter = FullSequenceBeliefGame(game, particles=16, seed=4)
    expected = np.full(52, 2)
    expected[first.to_int()] -= 1
    expected[second.to_int()] -= 1
    np.testing.assert_array_equal(adapter.available_counts, expected)
    assert adapter.prior.sum() == pytest.approx(1)
    for i, a in enumerate(adapter.hands[0]):
        for j, b in enumerate(adapter.hands[1]):
            if np.any(a + b > expected):
                assert adapter.prior[i, j] == 0
    game.deck.cards.pop()
    with pytest.raises(ValueError, match="inconsistent"):
        FullSequenceBeliefGame(game)


def test_private_query_jacks_use_exact_rules_and_do_not_modify_public_belief():
    game = SequenceGame()
    game.reset()
    game.board[3, 3] = 2
    protected = {(6, col) for col in range(1, 6)}
    for cell in protected:
        game.board[cell] = 2
    game.completed_sequences[2] = [protected]
    adapter = FullSequenceBeliefGame(game, particles=4, seed=8)
    actual = [Card.from_string(c) for c in ("JC", "JH", "2C", "3C", "4C", "5C", "6C")]
    snapshot = public_copy(game)
    snapshot.hands[1] = actual
    legal = snapshot.get_legal_moves()
    assert any(m.is_removal and (m.row, m.col) == (3, 3) for m in legal)
    assert not any(m.is_removal and (m.row, m.col) in protected for m in legal)
    before_prior = adapter.prior.copy()
    before_hands = tuple(bank.copy() for bank in adapter.hands)
    before_nodes = tuple(adapter.nodes)
    for move in legal:
        child, counts = adapter.query_after_action(action_key(move), actual)
        exact = snapshot.clone()
        assert exact.make_move(move)
        np.testing.assert_array_equal(child.board, exact.board)
        assert child.completed_sequences == exact.completed_sequences
        assert child.current_player == exact.current_player
        assert child.game_over == exact.game_over
        expected = card_counts(actual)
        expected[move.card.to_int()] -= 1
        np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(adapter.prior, before_prior)
    for original, after in zip(before_hands, adapter.hands):
        np.testing.assert_array_equal(original, after)
    assert tuple(adapter.nodes) == before_nodes


def test_root_type_legal_masks_match_actual_rules_and_leaf_features():
    game = SequenceGame()
    game.reset()
    adapter = FullSequenceBeliefGame(game, particles=4, seed=14)
    actor = adapter.root_player
    state = public_copy(game)
    for i, counts in enumerate(adapter.hands[actor]):
        state.hands[actor + 1] = counts_to_cards(counts)
        expected = {action_key(m) for m in state.get_legal_moves()}
        actual = {a for j, a in enumerate(adapter.root.children) if adapter.root.legal_mask[i, j]}
        assert expected == actual
    solver = PublicBeliefCFR(adapter.root, adapter.prior, lambda n, b: LeafValues(
        np.zeros(len(adapter.hands[0])), np.zeros(len(adapter.hands[1]))))
    result = solver.solve(3)
    assert result.value == pytest.approx(0)
    for node in adapter.nodes.values():
        features = adapter.features(node, result.last.evaluation.beliefs[node.key])
        assert [f.shape for f in features] == [(len(h), 562) for h in adapter.hands]
        assert all(np.isfinite(f).all() for f in features)


def test_query_conditioning_respects_compatibility_and_unavailable_support():
    game = SequenceGame()
    game.reset()
    adapter = FullSequenceBeliefGame(game, particles=16, seed=17)
    own = adapter.hands[0][0]
    conditional = adapter.query_support(0, own)
    assert conditional.sum() == pytest.approx(1)
    for weight, other in zip(conditional, adapter.hands[1]):
        if np.any(own + other > adapter.available_counts):
            assert weight == 0
    np.testing.assert_array_equal(adapter.query_support(0, own, np.zeros(len(conditional))), 0)


def test_empty_deck_public_banks_retain_at_least_one_compatible_deal():
    game = SequenceGame()
    game.reset()
    game.discard_piles[1].extend(game.deck.cards)
    game.deck.cards = []
    adapter = FullSequenceBeliefGame(game, particles=4, seed=99)
    assert adapter.deck_size == 0
    assert adapter.prior.sum() == pytest.approx(1)
    assert np.count_nonzero(adapter.prior) >= 1


def test_no_legal_action_is_immediate_draw_even_if_other_player_can_play():
    game = SequenceGame()
    dead, live = Card.from_string("2C"), Card.from_string("3C")
    game.hands = {1: [dead], 2: [live]}
    game.deck.cards = []
    available = np.zeros(52, dtype=int)
    available[dead.to_int()] = available[live.to_int()] = 1
    game.discard_piles = {1: counts_to_cards(2 - available), 2: []}
    for position in game.card_to_positions[dead]:
        game.board[position] = 1
    assert not game.get_legal_moves(1)
    assert game.get_legal_moves(2)
    adapter = FullSequenceBeliefGame(game, particles=16, seed=39, depth=2)
    terminal = adapter.root.children[PASS]
    assert terminal.actor is None
    assert terminal.payoff == 0.0
    assert terminal.children == {}
    assert adapter.state(terminal).game_over
    assert adapter.state(terminal).winner is None


def test_feature_only_adapter_preserves_public_banks_and_isolated_private_queries():
    game = SequenceGame()
    game.reset()
    full = FullSequenceBeliefGame(game, particles=4, seed=81)
    feature_only = FullSequenceBeliefGame(game, particles=4, seed=81, build_tree=False)
    assert len(feature_only.nodes) == 1
    assert feature_only.root.actor == full.root.actor
    assert feature_only.root.children == {}
    np.testing.assert_array_equal(full.prior, feature_only.prior)
    for first, second in zip(full.hands, feature_only.hands):
        np.testing.assert_array_equal(first, second)
    for move in game.get_legal_moves():
        state_a, hand_a = full.query_after_action(action_key(move), game.hands[game.current_player])
        state_b, hand_b = feature_only.query_after_action(action_key(move), game.hands[game.current_player])
        np.testing.assert_array_equal(state_a.board, state_b.board)
        np.testing.assert_array_equal(hand_a, hand_b)
        assert state_a.completed_sequences == state_b.completed_sequences
    with pytest.raises(ValueError, match="Decision node requires"):
        PublicBeliefCFR(feature_only.root, feature_only.prior, lambda *args: 0)
