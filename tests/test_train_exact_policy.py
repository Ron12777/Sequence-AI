"""Complete-game training semantics and resumability, using small real networks."""

import copy
import random

import numpy as np
import pytest
import torch

from scripts.distill_exact_policy import teacher_targets
from scripts.train_exact_policy import (
    RECIPE,
    checkpoint,
    evaluate,
    initialize,
    native_environment,
    observations,
    optimizer_for,
    restore,
    restore_rnad,
    rollout,
    tensors,
    update,
    update_rnad,
)
from src.exact_policy import ExactPolicy, load_exact_policy
from src.history_policy import HistoryPolicy
from src.model import SequenceNet
from src.rnad import RNaDConfig, RNaDState


@pytest.fixture(autouse=True)
def deterministic_small_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(112)
    np.random.seed(112)
    random.seed(112)
    yield
    torch.set_num_threads(previous)


def model(unbounded=False):
    teacher = HistoryPolicy(SequenceNet(num_filters=8, num_res_blocks=1), "board", width=8)
    return ExactPolicy.from_history_policy(teacher, value_mode="unbounded" if unbounded else "bounded").eval()


def test_complete_trajectory_rewards_and_behavior_are_consistent():
    actor = model(True)
    data, metrics = rollout(actor, "cpu", 4, 114, full_trajectory=True)
    assert metrics["decisions"] == metrics["positions"]
    torch.testing.assert_close(data["behavior_policy"].sum(1), torch.ones(len(data["action"])))
    assert not data["behavior_policy"][data["actions"] < 0].any()
    assert (data["actions"].gather(1, data["action"][:, None]) >= 0).all()
    for game in data["game_index"].unique():
        ids = torch.where(data["game_index"] == game)[0]
        ids = ids[data["ply"][ids].argsort()]
        assert torch.equal(data["ply"][ids], torch.arange(len(ids)))
        assert not data["rewards"][ids[:-1]].any()
        outcome = data["rewards"][ids[-1]]
        assert outcome.sum() == 0
        torch.testing.assert_close(data["return"][ids], outcome[data["player"][ids]])


@pytest.mark.parametrize("algorithm", ["ppo", "rnad"])
def test_optimizer_rng_and_reference_resume_match_next_update(tmp_path, algorithm):
    actor = model(algorithm == "rnad")
    recipe = {**RECIPE, "batch_size": 64, "epochs": 1,
              "rnad_games_per_minibatch": 2, "rnad_reference_interval": 2}
    optimizer = optimizer_for(actor, recipe, algorithm)
    rnad = RNaDState(actor, RNaDConfig(reference_interval=2)) if algorithm == "rnad" else None
    data, _ = rollout(actor, "cpu", 4, 771, full_trajectory=algorithm == "rnad")

    def perform(m, opt, state):
        return (update_rnad(m, opt, data, state, "cpu", recipe) if state is not None
                else update(m, opt, data, "cpu", recipe))

    perform(actor, optimizer, rnad)
    path = tmp_path / "resume.pt"
    checkpoint(actor, optimizer, path, {"algorithm": algorithm}, recipe, rnad)
    expected_rng = (random.random(), np.random.rand(), torch.rand(3))
    # The public loader must accept complete training checkpoints safely.
    load_exact_policy(path, "cpu")
    restored, restored_optimizer, _ = restore(path, "cpu", recipe)
    restored_rnad = restore_rnad(restored, path, recipe) if rnad is not None else None
    assert random.random() == expected_rng[0]
    assert np.random.rand() == expected_rng[1]
    assert torch.equal(torch.rand(3), expected_rng[2])
    perform(restored, restored_optimizer, restored_rnad)
    expected = copy.deepcopy(restored.state_dict())
    again, again_optimizer, _ = restore(path, "cpu", recipe)
    again_rnad = restore_rnad(again, path, recipe) if rnad is not None else None
    random.random()
    np.random.rand()
    torch.rand(3)
    metrics = perform(again, again_optimizer, again_rnad)
    assert metrics["updates"] > 0
    assert all(value is None or np.isfinite(value) for value in metrics.values())
    for name, value in again.state_dict().items():
        assert torch.equal(value, expected[name]), name
    if rnad is not None:
        assert again_rnad.learner_steps == restored_rnad.learner_steps
        for name, value in again_rnad.target.state_dict().items():
            assert torch.equal(value, restored_rnad.target.state_dict()[name])


def test_all_contenders_start_from_identical_distilled_predictions(tmp_path):
    initial = model()
    # Simulate nonzero learned residuals: copied common weights must survive.
    with torch.no_grad():
        for name, parameter in initial.named_parameters():
            if not name.startswith("backbone."):
                parameter.add_(torch.randn_like(parameter) * 0.005)
    env = native_environment(4, 112)
    inputs = tensors(observations(env, env.active_ids()), "cpu")
    with torch.no_grad():
        expected = initial(*inputs)
    for contender in RECIPE["contenders"]:
        directory = tmp_path / contender
        initialize(initial, contender, directory, RECIPE, "cpu")
        restored, _ = load_exact_policy(directory / "latest.pt", "cpu")
        with torch.no_grad():
            actual = restored(*inputs)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_paired_identical_policy_is_exactly_even_without_mutation():
    actor = model()
    result = evaluate(actor, copy.deepcopy(actor), "cpu", 8, 445)
    assert result["score"] == 0.5
    assert result["parameters_and_buffers_unchanged"]


def test_distillation_targets_preserve_positions_and_select_legal_exact_cards():
    spatial = torch.zeros(1, 100)
    actions = torch.tensor([[100, 1000, 111, -1]])  # Normal and Jack share square 0.
    targets, selected = teacher_targets(spatial, actions)
    torch.testing.assert_close(targets, torch.tensor([[0.5, 0., 0.5, 0.]]))
    assert selected.item() == 0
