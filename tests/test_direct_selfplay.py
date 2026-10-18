import torch
import pytest
from scripts.direct_selfplay import masked_distribution, ppo_objective, rollout, update
from src.history_policy import HistoryPolicy
from src.model import SequenceNet


def test_sampling_never_selects_illegal_position_and_ppo_clips():
    logits = torch.tensor([[100.0, 0.0, -2.0]])
    mask = torch.tensor([[False, True, False]])
    distribution = masked_distribution(logits, mask)
    assert (distribution.sample((100,)) == 1).all()
    old = torch.zeros(2)
    new = torch.tensor([2.0, -2.0], requires_grad=True)
    loss = ppo_objective(new, old, torch.tensor([1.0, -1.0]))
    loss.backward()
    torch.testing.assert_close(new.grad, torch.zeros(2))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_complete_public_history_rollout_and_parameter_update(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = HistoryPolicy(
        SequenceNet(num_filters=8, num_res_blocks=1), "gru", width=16
    ).to(device)
    data, metrics = rollout(model, device, 2, 191)
    assert metrics["games"] == 2 and metrics["positions"] > 0
    assert data["mask"][torch.arange(len(data["action"])), data["action"]].all()
    before = model.condition[-1].weight.detach().clone()
    result = update(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        data,
        device,
        epochs=1,
    )
    assert result["updates"] > 0
    assert model.encoder.training
    assert not model.base.input_conv[1].training
    assert not torch.equal(before, model.condition[-1].weight)


@pytest.mark.parametrize("iterations,games", [(16, 128), (256, 128), (1000, 12000)])
def test_heldout_seed_ranges_stay_outside_all_training_deals(iterations, games):
    from scripts.direct_selfplay import seed_schedule

    stride, evaluation_seeds = seed_schedule(91, iterations, games, 1024)
    last_training_seed = 91 + (iterations - 1) * stride + games - 1
    assert evaluation_seeds[0] > last_training_seed
    assert stride >= games
    assert all(b > a + 1024 for a, b in zip(evaluation_seeds, evaluation_seeds[1:]))
