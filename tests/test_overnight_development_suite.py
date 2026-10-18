"""Final development preserves continuations and never starts before shutdown."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'experiments/2026-10-18/run-final-development-suite.py'
spec = importlib.util.spec_from_file_location('overnight_development', SOURCE)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def setup(tmp_path, monkeypatch):
    base = tmp_path / 'campaign/overnight'
    (base / 'locks').mkdir(parents=True)
    (base / 'milestones').mkdir()
    (base / 'final-development').mkdir()
    (base / 'recipe.json').write_text(json.dumps({'contenders': list(suite.MODELS)}))
    manifest = {}
    for model in suite.MODELS:
        (base / model).mkdir()
        (base / model / 'latest.pt').write_bytes(model.encode())
        (base / model / 'historical_0.pt').write_bytes(b'history')
        for hour in (5, 8):
            key = f'{model}-additional-{hour}h'
            p = base / 'milestones' / (key + '.pt')
            p.write_bytes(key.encode())
            manifest[key] = {'sha256': suite.confirmation.digest(p)}
    (base / 'milestones/manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'incumbent.pt').write_bytes(b'incumbent')
    monkeypatch.setattr(suite.confirmation, 'INCUMBENT', ('incumbent.pt', suite.confirmation.digest(tmp_path / 'incumbent.pt')))
    return base, base / 'final-development'


def test_six_frozen_paired_plans_and_immutable_continuations(tmp_path, monkeypatch):
    base, out = setup(tmp_path, monkeypatch)
    calls = []
    def evaluate(cmd, **kwargs):
        assert len(list(out.glob('*-plan.json'))) == 6
        assert (out / 'snapshot-manifest.json').exists()
        plan = json.loads(Path(cmd[cmd.index('--plan') + 1]).read_text())
        opponent = plan['opponents'][0]
        assert opponent['seed'] == 4200400000001 and opponent['games'] == 4096
        assert kwargs['timeout'] <= 600
        row = {'label': opponent['label'], 'games': 4096, 'outcomes': [0.5] * 4096,
               'score': 0.5, 'paired_score_95ci': [0.5, 0.5]}
        report = {'status': 'complete', 'plan': plan, 'parameters_and_buffers_unchanged': True,
                  'evaluations': [row], 'elapsed_seconds': 1}
        Path(cmd[cmd.index('--output') + 1]).write_text(json.dumps(report))
        calls.append(cmd)
    monkeypatch.setattr(suite.subprocess, 'run', evaluate)
    suite.execute(tmp_path, base, out, {'status': 'stopped'})
    assert len(calls) == 6
    result = json.loads((out / 'suite-complete.json').read_text())
    assert result['selected_for_confirmation']['name'] == 'cnn_ppo_5h'
    assert not result['reserved_final_seeds_used'] and not result['automatic_promotion']
    assert not (base / 'gpu-0.pause').exists()
    # Atomic continuation replacement must not alter the captured final.
    replacement = base / 'cnn_ppo/replacement.pt'
    replacement.write_bytes(b'new')
    os.replace(replacement, base / 'cnn_ppo/latest.pt')
    assert (out / 'checkpoints/cnn_ppo-final.pt').read_bytes() == b'cnn_ppo'


def test_timeout_releases_gpu_before_unpause(tmp_path, monkeypatch):
    base, out = setup(tmp_path, monkeypatch)
    def fail(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs['timeout'])
    monkeypatch.setattr(suite.subprocess, 'run', fail)
    with pytest.raises(subprocess.TimeoutExpired):
        suite.execute(tmp_path, base, out, {'status': 'stopped'})
    with suite.confirmation.lock(base / 'locks/gpu-0.lock'):
        assert not (base / 'gpu-0.pause').exists()
    assert not (out / 'suite-complete.json').exists()


def test_shutdown_timeout_never_snapshots_or_evaluates(tmp_path, monkeypatch):
    base, out = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(suite.time, 'time', lambda: suite.WAIT_UNTIL + 1)
    def forbidden(*args):
        pytest.fail('No evaluation allowed without verified shutdown')
    monkeypatch.setattr(suite, 'execute', forbidden)
    previous = Path.cwd()
    try:
        with pytest.raises(TimeoutError):
            suite.run(tmp_path)
    finally:
        os.chdir(previous)
    assert not (out / 'snapshot-manifest.json').exists()
