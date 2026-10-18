import copy
import random

import pytest
import torch

from scripts.direct_selfplay import rollout, update
from scripts.sustained_selfplay import RECIPE, checkpoint, restore, schedule
from src.history_policy import HistoryPolicy
from src.model import SequenceNet


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_mixed_opponents_and_exact_optimizer_rng_resume(tmp_path, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(11)
    random.seed(11)
    device = torch.device(device_name)
    model = HistoryPolicy(SequenceNet(num_filters=8, num_res_blocks=1), "gru", 16).to(
        device
    )
    opponent = copy.deepcopy(model)
    original = {k: v.clone() for k, v in opponent.state_dict().items()}
    data, metrics = rollout(model, device, 8, 77, opponents=[opponent, opponent])
    assert metrics["opponent_moves"] > 0
    assert metrics["positions"] > metrics["opponent_moves"]
    assert data["mask"][torch.arange(len(data["action"])), data["action"]].all()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    update(
        model,
        optimizer,
        data,
        device,
        epochs=1,
        batch_size=128,
        entropy_coefficient=0.003,
        clip=0.1,
        target_kl=0.015,
    )
    path = tmp_path / "resume.pt"
    checkpoint(
        model, optimizer, path, {"training_seconds": 12, "iterations": 1}, RECIPE
    )
    python_draw = random.random()
    torch_draw = torch.rand(4, device=device)
    expected = update(model, optimizer, data, device, epochs=1, batch_size=128)
    resumed, resumed_optimizer, metadata = restore(path, device)
    assert metadata["training_seconds"] == 12
    assert random.random() == python_draw
    torch.testing.assert_close(torch.rand(4, device=device), torch_draw, rtol=0, atol=0)
    actual = update(resumed, resumed_optimizer, data, device, epochs=1, batch_size=128)
    assert actual["updates"] == expected["updates"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            resumed.state_dict()[key], value, rtol=1e-5, atol=1e-6
        )
    for key, value in opponent.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_schedule_is_continuous_in_training_time_and_bounded():
    start = schedule(0)
    middle = schedule(4 * 3600)
    end = schedule(8 * 3600)
    assert start[0] > middle[0] > end[0]
    assert start[1] > middle[1] > end[1]
    assert schedule(100 * 3600) == end
    assert end == (RECIPE["minimum_lr"], RECIPE["final_entropy_coefficient"])


@pytest.mark.parametrize("simulations", [0, 4])
def test_evaluation_freezes_reference_batchnorm_and_preserves_all_buffers(simulations):
    from scripts.architecture_experiments import evaluate_policy

    torch.set_num_threads(1)
    teacher = SequenceNet(num_filters=8, num_res_blocks=1)
    candidate = HistoryPolicy(teacher, "gru", width=16)
    assert teacher.training and candidate.training
    original = {key: value.clone() for key, value in teacher.state_dict().items()}
    candidate_original = {
        key: value.clone() for key, value in candidate.state_dict().items()
    }
    result = evaluate_policy(
        candidate, teacher, torch.device("cpu"), 4, 871, "champion", simulations
    )
    assert result["games"] == 4
    assert not teacher.training and not candidate.training
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    for key, value in candidate.state_dict().items():
        torch.testing.assert_close(value, candidate_original[key], rtol=0, atol=0)


def test_prediction_rejects_training_mode_before_mutating_batchnorm():
    import numpy as np

    from scripts.evaluate import predict

    teacher = SequenceNet(num_filters=8, num_res_blocks=1)
    before = teacher.input_conv[1].running_mean.clone()
    with pytest.raises(ValueError, match="model.eval"):
        predict(teacher, torch.device("cpu"), [np.zeros((11, 10, 10), np.float32)])
    torch.testing.assert_close(
        teacher.input_conv[1].running_mean, before, rtol=0, atol=0
    )


def test_history_opponents_are_frozen_and_identical_models_split_paired_deals():
    from scripts.architecture_experiments import evaluate_policy
    from src.history_policy import HistoryPolicy
    from src.model import SequenceNet

    torch.set_num_threads(1)
    torch.manual_seed(871)
    candidate = HistoryPolicy(
        SequenceNet(num_filters=8, num_res_blocks=1), "gru", width=16
    )
    teacher = HistoryPolicy(
        SequenceNet(num_filters=8, num_res_blocks=1), "gru", width=16
    )
    teacher.load_state_dict(candidate.state_dict())
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    result = evaluate_policy(
        candidate, teacher, torch.device("cpu"), 8, 212381, opponent="history"
    )
    assert result["score"] == 0.5
    assert all(
        a + b == 1 for a, b in zip(result["outcomes"][::2], result["outcomes"][1::2])
    )
    assert not candidate.training and not teacher.training
    for model in [candidate, teacher]:
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="direct policy"):
        evaluate_policy(
            candidate,
            teacher,
            torch.device("cpu"),
            8,
            212381,
            opponent="history",
            simulations=128,
        )
