import copy

import pytest
import torch

from scripts.architecture_experiments import direct_moves
from scripts.evaluate_rebel_full import evaluate_full
from src.history_policy import HistoryPolicy
from src.model import SequenceNet


class DirectAgent:
    def __init__(self, model):
        self.model = model
        self.resets = []

    def reset(self, *, seed, player):
        self.resets.append((seed, player))

    def act(self, game, player, events):
        assert player == game.current_player
        assert isinstance(events, tuple)
        return direct_moves(self.model, torch.device("cpu"), [game], [events], [0])[0]


def models():
    torch.set_num_threads(1)
    torch.manual_seed(472)
    policy = HistoryPolicy(SequenceNet(num_filters=8, num_res_blocks=1), "gru", 16)
    return policy, copy.deepcopy(policy)


def test_identical_direct_policies_complete_paired_games_and_resume():
    candidate, reference = models()
    agent = DirectAgent(candidate)
    progress = []
    result = evaluate_full(
        agent,
        reference,
        games=4,
        seed=721,
        frozen_models=[candidate],
        on_pair=lambda records: progress.append(len(records)),
    )
    assert result["score"] == 0.5
    assert progress == [2, 4]
    assert result["parameters_and_buffers_unchanged"]
    assert result["frozen_model_count"] == 2
    assert [player for _, player in agent.resets] == [1, 2, 1, 2]
    for first, second in zip(result["game_records"][::2], result["game_records"][1::2]):
        assert first["initial_state_sha256"] == second["initial_state_sha256"]
        assert first["score"] + second["score"] == 1
        assert first["termination"] in {"terminal", "no_legal_moves"}
        assert first["plies"] == len(first["moves"]) > 0
        assert all(move["seconds"] >= 0 for move in first["moves"])
    resumed_agent = DirectAgent(candidate)
    resumed = evaluate_full(
        resumed_agent,
        reference,
        games=4,
        seed=721,
        frozen_models=[candidate],
        completed_games=result["game_records"][:2],
    )
    assert resumed["outcomes"] == result["outcomes"]
    assert len(resumed_agent.resets) == 2
    assert [
        [(m["card"], m["row"], m["col"], m["is_removal"]) for m in g["moves"]]
        for g in resumed["game_records"]
    ] == [
        [(m["card"], m["row"], m["col"], m["is_removal"]) for m in g["moves"]]
        for g in result["game_records"]
    ]


def test_search_reference_completes_games_without_model_mutation():
    candidate, _ = models()
    reference = copy.deepcopy(candidate.base)
    result = evaluate_full(
        DirectAgent(candidate),
        reference,
        games=2,
        seed=812,
        opponent="champion",
        simulations=2,
        frozen_models=[candidate],
    )
    assert result["games"] == 2
    assert result["parameters_and_buffers_unchanged"]
    assert result["candidate_latency"]["moves"] > 0
    assert result["opponent_latency"]["moves"] > 0


def test_live_state_mutation_is_rejected_before_move_execution():
    candidate, reference = models()

    class MutatingAgent(DirectAgent):
        def act(self, game, player, events):
            result = super().act(game, player, events)
            game.deck.cards.pop()
            return result

    with pytest.raises(RuntimeError, match="mutated the live game"):
        evaluate_full(
            MutatingAgent(candidate),
            reference,
            games=2,
            seed=721,
            frozen_models=[candidate],
        )


def test_frozen_candidate_buffer_mutation_is_rejected():
    candidate, reference = models()

    class MutatingAgent(DirectAgent):
        def act(self, game, player, events):
            result = super().act(game, player, events)
            self.model.base.input_conv[1].running_mean.add_(0.01)
            return result

    with pytest.raises(RuntimeError, match="parameters or buffers"):
        evaluate_full(
            MutatingAgent(candidate),
            reference,
            games=2,
            seed=721,
            frozen_models=[candidate],
        )


def test_position_only_action_is_rejected():
    candidate, reference = models()

    class IncompleteAgent(DirectAgent):
        def act(self, game, player, events):
            return 42

    with pytest.raises(RuntimeError, match="illegal or incomplete"):
        evaluate_full(
            IncompleteAgent(candidate),
            reference,
            games=2,
            seed=721,
            frozen_models=[candidate],
        )
