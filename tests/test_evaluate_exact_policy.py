"""Full paired matches, native reference parity and immutable evaluation evidence."""

import json
import random

import numpy as np
import pytest
import sequence_native
import torch

from scripts import evaluate_exact_policy as runner
from scripts.architecture_experiments import direct_moves
from scripts.arena import choose_moves
from src.exact_policy import ExactPolicy, save_exact_policy
from src.fast_game import FastSequenceGame
from src.history_policy import HistoryPolicy, apply_public_move
from src.model import SequenceNet


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model():
    torch.manual_seed(33)
    teacher = HistoryPolicy(
        SequenceNet(num_filters=8, num_res_blocks=1), "board", width=8
    )
    return ExactPolicy(teacher, card_width=8).eval()


def setup_reference(seed=97):
    random.seed(seed)
    game = FastSequenceGame()
    deck = [
        c.to_int()
        for c in game.deck.cards
        + list(reversed(game.hands[2]))
        + list(reversed(game.hands[1]))
    ]
    return game, sequence_native.BatchEnv.from_decks([deck])


def encoded(move):
    return move.card.to_int() * 100 + move.row * 10 + move.col


def test_complete_identical_exact_pairs_and_chunk_deals_match():
    net = model()
    result = runner.evaluate_exact(net, net, games=4, seed=931, kind="exact")
    assert result["score"] == 0.5
    assert result["outcomes"][0] + result["outcomes"][1] == 1
    assert result["outcomes"][2] + result["outcomes"][3] == 1
    assert result["parameters_and_buffers_unchanged"]
    records = result["game_records"]
    assert all(0 < r["plies"] == len(r["exact_actions"]) < 105 for r in records)
    assert records[0]["exact_actions"] == records[1]["exact_actions"]
    second = runner.evaluate_exact(
        net, net, games=2, seed=932, kind="exact", game_offset=2
    )
    assert second["game_records"] == records[2:]


@pytest.mark.parametrize("zero", [False, True])
def test_legacy_original_hand_order_and_ties_match_existing_player(zero):
    game, env = setup_reference()
    net = HistoryPolicy(
        SequenceNet(num_filters=8, num_res_blocks=1), "gru", width=8
    ).eval()
    if zero:
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
    events = []
    for _ in range(12):
        expected = direct_moves(net, "cpu", [game], [events], [0])[0]
        actual = runner.legacy_moves(net, env, [0], "cpu", "history")[0]
        assert actual == encoded(expected)
        env.step([0], [actual])
        apply_public_move(game, events, expected)


def test_existing_ensemble_search_receives_identical_public_inputs_and_rng():
    game, env = setup_reference()
    net = SequenceNet(num_filters=8, num_res_blocks=1).eval()
    events = []
    for turn in range(3):
        expected = choose_moves(
            net, torch.device("cpu"), [game], [0], 16, 551, [turn], "ensemble"
        )[0]
        actual = runner.search_moves(
            net, env, [0], torch.device("cpu"), 16, 551, [turn]
        )[0]
        assert actual == encoded(expected)
        env.step([0], [actual])
        apply_public_move(game, events, expected)


def test_confirmation_checks_hashes_and_resumes_without_rerunning(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "exact.pt"
    save_exact_policy(model(), checkpoint, {"test": True})
    ref = {"checkpoint": str(checkpoint), "sha256": runner.digest(checkpoint)}
    plan = {
        "candidate": ref,
        "opponents": [
            {**ref, "label": "control", "kind": "exact", "games": 4, "seed": 174}
        ],
        "batch_size": 2,
    }
    output = tmp_path / "result.json"
    report = runner.confirm(plan, output)
    assert report["status"] == "complete"
    assert len(report["evaluations"][0]["chunks"]) == 2
    assert json.loads(output.read_text()) == report
    monkeypatch.setattr(
        runner,
        "evaluate_exact",
        lambda *a, **k: pytest.fail("reran complete evaluation"),
    )
    assert runner.confirm(plan, output) == report
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.confirm(plan, output)


def test_mutated_parameters_or_buffers_are_rejected():
    net = model()
    before = runner.freeze(net)
    with torch.no_grad():
        next(net.parameters()).add_(1)
    with pytest.raises(RuntimeError, match="mutated"):
        runner.verify_frozen(net, before)


def test_interrupted_confirmation_resumes_only_remaining_fresh_pairs(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "exact.pt"
    save_exact_policy(model(), checkpoint, {})
    ref = {"checkpoint": str(checkpoint), "sha256": runner.digest(checkpoint)}
    plan = {
        "candidate": ref,
        "opponents": [
            {**ref, "label": "control", "kind": "exact", "games": 4, "seed": 174}
        ],
        "batch_size": 2,
    }
    output = tmp_path / "result.json"
    original = runner.evaluate_exact
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(kwargs["seed"])
        if len(calls) == 2:
            raise RuntimeError("simulated process interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_exact", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.confirm(plan, output)
    saved = json.loads(output.read_text())
    assert len(saved["active"]["outcomes"]) == 2
    completed = saved["active"]["chunks"][0]
    resumed_calls = []

    def resume(*args, **kwargs):
        resumed_calls.append(kwargs["seed"])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_exact", resume)
    report = runner.confirm(plan, output)
    assert resumed_calls == [175]
    assert report["evaluations"][0]["chunks"][0] == completed
    assert report["status"] == "complete"


def test_categorical_sampler_masks_padding_and_uses_exact_card_ids():
    logits = np.array([[0.0, np.log(3.0), 1e30]] * 4)
    actions = np.array([[100, 1000, -1]] * 4)  # Distinct cards at the same square.
    assert runner.categorical_actions(logits, actions, [0.0, 0.2, 0.3, 0.999]) == [100, 100, 1000, 1000]
    # A zero draw must skip zero-probability leading actions after underflow.
    assert runner.categorical_actions(np.array([[-1000.0, 0.0, 1e30]]), actions[:1], [0.0]) == [1000]
    with pytest.raises(ValueError, match="finite legal logits"):
        runner.categorical_actions(logits[:1], np.full((1, 3), -1), [0.5])


def test_action_draws_are_stateless_and_do_not_consume_global_rng():
    random.seed(553)
    np.random.seed(661)
    torch.manual_seed(779)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    keys = [(23, 0), (4, 8), (23, 1), (2, 3)]
    draws = [runner.action_uniform(123456, game, decision) for game, decision in keys]
    reversed_draws = [runner.action_uniform(123456, game, decision) for game, decision in reversed(keys)]
    assert draws == reversed_draws[::-1]
    assert len(set(draws)) == len(draws) and all(0 <= draw < 1 for draw in draws)
    assert draws != [runner.action_uniform(123457, game, decision) for game, decision in keys]
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_sampled_full_games_match_across_batching_and_preserve_models():
    candidate, opponent = model(), model()
    before = [runner.freeze(net) for net in (candidate, opponent)]
    torch_rng = torch.get_rng_state().clone()
    full = runner.evaluate_exact(
        candidate, opponent, games=8, seed=8133, kind="exact",
        candidate_action_mode="sample", action_seed=991237,
    )
    assert torch.equal(torch.get_rng_state(), torch_rng)
    chunks, outcomes = [], []
    for offset in (0, 2, 4, 6):
        result = runner.evaluate_exact(
            candidate, opponent, games=2, seed=8133 + offset // 2, kind="exact",
            game_offset=offset, candidate_action_mode="sample", action_seed=991237,
        )
        chunks.extend(result["game_records"])
        outcomes.extend(result["outcomes"])
    assert full["game_records"] == chunks
    assert full["outcomes"] == outcomes
    assert full["candidate_action_mode"] == "sample"
    assert full["candidate_temperature"] == 1.0
    assert full["opponent_action_mode"] == "greedy"
    assert all(record["action_seed"] == 991237 for record in chunks)
    assert all(record["candidate_decisions"] == (record["plies"] + (record["candidate_player"] == 1)) // 2 for record in chunks)
    for net, snapshot in zip((candidate, opponent), before):
        runner.verify_frozen(net, snapshot)


def test_default_greedy_records_unchanged_and_opponent_never_samples(monkeypatch):
    candidate, opponent = model(), model()
    implicit = runner.evaluate_exact(candidate, opponent, games=2, seed=775, kind="exact")
    explicit = runner.evaluate_exact(
        candidate, opponent, games=2, seed=775, kind="exact", candidate_action_mode="greedy",
    )
    assert implicit.keys() == explicit.keys()
    assert "candidate_action_mode" not in implicit
    assert implicit["game_records"] == explicit["game_records"]
    assert implicit["outcomes"] == explicit["outcomes"]
    original = runner.exact_moves
    modes = []

    def track(actor, *args, **kwargs):
        modes.append((actor is candidate, kwargs.get("action_mode", "greedy")))
        return original(actor, *args, **kwargs)

    monkeypatch.setattr(runner, "exact_moves", track)
    runner.evaluate_exact(
        candidate, opponent, games=2, seed=775, kind="exact",
        candidate_action_mode="sample", action_seed=4481,
    )
    assert modes and all(mode == ("sample" if is_candidate else "greedy") for is_candidate, mode in modes)


def test_sampled_confirmation_resume_retains_draws_and_trace(tmp_path, monkeypatch):
    path = tmp_path / "candidate.pt"
    save_exact_policy(model(), path, {})
    ref = {"checkpoint": str(path), "sha256": runner.digest(path)}
    plan = {
        "candidate": ref,
        "opponents": [{**ref, "label": "reference", "kind": "exact", "games": 6, "seed": 913}],
        "batch_size": 2, "candidate_action_mode": "sample", "action_seed": 761977,
    }
    uninterrupted = runner.confirm(plan, tmp_path / "uninterrupted.json")
    original = runner.evaluate_exact
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_exact", interrupt)
    output = tmp_path / "resumed.json"
    with pytest.raises(RuntimeError, match="deliberate"):
        runner.confirm(plan, output)
    saved_chunk = json.loads(output.read_text())["active"]["chunks"][0]
    monkeypatch.setattr(runner, "evaluate_exact", original)
    resumed = runner.confirm(plan, output)
    assert resumed["evaluations"][0]["chunks"][0] == saved_chunk
    for a, b in zip(resumed["evaluations"][0]["chunks"], uninterrupted["evaluations"][0]["chunks"]):
        assert a["game_records"] == b["game_records"]
        assert a["action_seed"] == b["action_seed"] == 761977
    assert resumed["evaluations"][0]["score"] == uninterrupted["evaluations"][0]["score"]


@pytest.mark.parametrize("mode,seed", [("bad", 1), ("sample", None), ("sample", 1.5), ("sample", True)])
def test_invalid_sampling_configuration_rejected_before_evaluation(mode, seed):
    with pytest.raises(ValueError):
        runner.evaluate_exact(None, None, games=2, seed=1, candidate_action_mode=mode, action_seed=seed)
