"""Focused shared-update routing, whole-batch math, and explicit RNG transition."""
import copy

import pytest
import torch

from scripts import train_exact_policy as trainer
from scripts.benchmark_exact_dataparallel import CpuSplitForward
from tests.test_train_exact_policy import model


def test_parallel_validation_rejects_cpu_and_duplicate_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    assert trainer._parallel_devices('cpu', None) is None
    assert trainer._parallel_devices('cuda:0', [0, 1]) == [0, 1]
    for device, ids in [('cpu', [0, 1]), ('cuda:0', [0, 0]), ('cuda:1', [0, 1]), ('cuda:0', [0, 2])]:
        with pytest.raises(ValueError):
            trainer._parallel_devices(device, ids)


def test_parallel_update_keeps_one_global_loss_and_raw_state(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(7741)
        initial = model()
        data, _ = trainer.rollout(initial, 'cpu', 4, 771)
        # Repeat authentic observations into one full batch plus a small tail.
        ids = torch.arange(1057) % len(data['action'])
        data = {key: value[ids] for key, value in data.items()}
        recipe = {**trainer.RECIPE, 'epochs': 1, 'batch_size': 1024}
        baseline, shared = copy.deepcopy(initial), copy.deepcopy(initial)
        baseline_opt, shared_opt = (trainer.optimizer_for(m, recipe) for m in (baseline, shared))
        torch.manual_seed(990)
        expected = trainer.update(baseline, baseline_opt, data, 'cpu', recipe)
        monkeypatch.setattr(trainer, '_parallel_devices', lambda device, requested: requested)
        calls = []
        def fake_dp(raw, device_ids, output_device):
            assert device_ids == [0, 1] and output_device == 0
            calls.append(raw)
            return CpuSplitForward(raw)
        monkeypatch.setattr(torch.nn, 'DataParallel', fake_dp)
        torch.manual_seed(990)
        actual = trainer.update(shared, shared_opt, data, 'cpu', recipe, parallel_update_devices=[0, 1])
        assert calls == [shared]
        assert actual['parallel_forward_batches'] == 1
        assert actual['updates'] == expected['updates'] == 2
        assert actual['kl_early_stop'] == expected['kl_early_stop']
        assert shared.state_dict().keys() == initial.state_dict().keys()
        for key, value in shared.state_dict().items():
            torch.testing.assert_close(value, baseline.state_dict()[key], atol=2e-7, rtol=2e-6)
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('count', [1, 2, 3])
def test_explicit_parallel_rng_transition(tmp_path, monkeypatch, count):
    actor = model()
    optimizer = trainer.optimizer_for(actor, trainer.RECIPE)
    path = tmp_path / 'resume.pt'
    trainer.checkpoint(actor, optimizer, path, {}, trainer.RECIPE)
    payload = torch.load(path, weights_only=False)
    states = [torch.tensor([index, 7], dtype=torch.uint8) for index in range(count)]
    payload['training_state']['cuda_rng'] = states
    torch.save(payload, path)
    captured = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda values: captured.extend(values))
    if count == 3:
        with pytest.raises(ValueError, match='RNG count'):
            trainer.restore(path, 'cpu', trainer.RECIPE, parallel_update_devices=[0, 1])
    else:
        trainer.restore(path, 'cpu', trainer.RECIPE, parallel_update_devices=[0, 1])
        assert len(captured) == 2
        assert torch.equal(captured[0], states[0])
        assert torch.equal(captured[1], states[-1])
