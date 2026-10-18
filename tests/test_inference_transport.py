"""Validate actual spawned-process slot ownership and inference routing."""

import multiprocessing as mp
import queue
import numpy as np
import pytest
import torch
from src.inference_transport import InferenceSlot
from src.model import SequenceNet
from src.train import InferenceServer


def exchange(worker, slot, requests, responses, completed):
    observations, policies, values = slot.views()
    try:
        for round_index, size in enumerate([3, 1, 2]):
            observations[:size].fill(worker + round_index / 10)
            requests.put((worker, size))
            assert responses.get(timeout=15) == size
            completed.put(
                (worker, round_index, policies[:size].copy(), values[:size].copy())
            )
    except Exception as exc:
        completed.put((worker, "error", repr(exc)))
        raise


def test_shared_slots_cross_spawn_boundary_and_do_not_mix_workers():
    ctx = mp.get_context("spawn")
    slots = {i: InferenceSlot(3) for i in (0, 1)}
    requests, completed = ctx.Queue(), ctx.Queue()
    responses = {i: ctx.Queue() for i in slots}
    torch.manual_seed(11)
    model = SequenceNet(num_filters=8, num_res_blocks=1).eval()
    server = InferenceServer(
        model,
        torch.device("cpu"),
        responses,
        batch_size=6,
        timeout=0.01,
        shared_slots=slots,
    )
    processes = [
        ctx.Process(
            target=exchange, args=(i, slots[i], requests, responses[i], completed)
        )
        for i in slots
    ]
    received = []
    try:
        for process in processes:
            process.start()
        import time

        deadline = time.monotonic() + 20
        while len(received) < 6 and time.monotonic() < deadline:
            server.step(requests)
            try:
                while True:
                    received.append(completed.get_nowait())
            except queue.Empty:
                pass
        assert len(received) == 6
        for worker, round_index, pi, value in received:
            size = [3, 1, 2][round_index]
            with torch.inference_mode():
                expected_pi, expected_value = model(
                    torch.full((size, 11, 10, 10), worker + round_index / 10)
                )
            np.testing.assert_allclose(pi, expected_pi.exp().numpy(), atol=1e-6)
            np.testing.assert_allclose(
                value, expected_value.numpy().reshape(-1), atol=1e-6
            )
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
        for q in [requests, completed, *responses.values()]:
            q.close()


def test_shared_request_rejects_oversized_or_unknown_worker():
    responses = {0: queue.Queue()}
    model = SequenceNet(num_filters=8, num_res_blocks=1)
    server = InferenceServer(
        model, torch.device("cpu"), responses, shared_slots={0: InferenceSlot(2)}
    )
    for item in [(0, 3), (0, 0), (1, 1), (0, np.zeros(1))]:
        requests = queue.Queue()
        requests.put(item)
        with pytest.raises(ValueError):
            server.step(requests)
