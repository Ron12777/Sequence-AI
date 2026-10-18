import queue
import numpy as np
import pytest
import torch
from src.model import SequenceNet
from src.train import InferenceServer, train_step


@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
def test_inference_server_preserves_worker_batches(dtype):
    model = SequenceNet(num_filters=8, num_res_blocks=1).eval()
    requests = queue.Queue()
    results = {i: queue.Queue() for i in (0, 1)}
    observations = {}
    for i, n in [(0, 2), (1, 3)]:
        observations[i] = (
            np.random.default_rng(i).integers(0, 2, (n, 11, 10, 10)).astype(dtype)
        )
        requests.put((i, observations[i]))
    server = InferenceServer(
        model, torch.device("cpu"), results, batch_size=5, timeout=0.001
    )
    assert server.step(requests) == 5
    for i, n in [(0, 2), (1, 3)]:
        pi, value = results[i].get_nowait()
        assert pi.shape == (n, 100) and value.shape == (n,)
        np.testing.assert_allclose(pi.sum(axis=1), 1, atol=1e-6)
        with torch.inference_mode():
            expected_pi, expected_value = model(
                torch.from_numpy(observations[i]).float()
            )
        torch.testing.assert_close(torch.from_numpy(pi), expected_pi.exp())
        torch.testing.assert_close(torch.from_numpy(value), expected_value.flatten())


def test_single_sample_training_has_finite_gradients():
    model = SequenceNet(num_filters=8, num_res_blocks=1)
    optimizer = torch.optim.Adam(model.parameters())
    result = train_step(
        model,
        optimizer,
        torch.randn(1, 11, 10, 10),
        torch.full((1, 100), 0.01),
        torch.tensor([1.0]),
        device=torch.device("cpu"),
    )
    assert all(np.isfinite(v) for v in result[:3]) and result[3]


def test_optimizer_state_migration_preserves_moments_and_cpu_step():
    from src.train import move_optimizer_state_to_device

    parameter = torch.nn.Parameter(torch.zeros(2, 11, 3, 3))
    optimizer = torch.optim.Adam([parameter])
    optimizer.state[parameter] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.ones(2, 8, 3, 3),
        "exp_avg_sq": torch.ones(2, 8, 3, 3),
    }
    move_optimizer_state_to_device(optimizer, torch.device("cpu"))
    state = optimizer.state[parameter]
    assert state["step"].device.type == "cpu"
    torch.testing.assert_close(state["exp_avg"][:, :8], torch.ones(2, 8, 3, 3))
    assert state["exp_avg"][:, 8:].count_nonzero() == 0


def test_replay_learner_runs_exact_update_budget():
    from src.train import ReplayLearner

    model = SequenceNet(num_filters=8, num_res_blocks=1)
    optimizer = torch.optim.Adam(model.parameters())
    replay = [
        (np.zeros((11, 10, 10), np.float32), np.full(100, 0.01, np.float32), 0.0)
        for _ in range(4)
    ]
    learner = ReplayLearner(model, optimizer, torch.device("cpu"), replay, 2, 2)
    learner.step()
    learner.step()
    learner.step()
    assert learner.remaining == 0 and len(learner.losses) == 2
    assert np.isfinite(learner.losses).all()
