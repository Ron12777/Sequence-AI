"""CPU-only fixed final protocol and orchestration checks; no model execution."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'experiments/2026-10-18/run-final-confirmation-suite.py'
spec = importlib.util.spec_from_file_location('final_confirmation_suite', SOURCE)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def report(plan, head=1.0, heldout=0.5):
    return {'status': 'complete', 'plan': plan, 'parameters_and_buffers_unchanged': True,
            'elapsed_seconds': 1.0, 'evaluations': [
                {'label': row['label'], 'games': row['games'],
                 'outcomes': [head if row['label'] == 'validated_oct17_cnn' else heldout] * row['games']}
                for row in plan['opponents']]}


def plans():
    return suite.make_plans(Path('/fake'), {'checkpoint': '/fake/selected.pt', 'sha256': 'a' * 64})


def test_fixed_counts_disjoint_seeds_and_paired_controls():
    p = plans()
    rows = p['candidate']['opponents']
    assert [r['games'] for r in rows] == [8192, 2048, 8192]
    assert rows[1:] == p['incumbent_control']['opponents']
    seeds = [set(range(r['seed'], r['seed'] + r['games'] // 2)) for r in rows]
    assert all(not seeds[i] & seeds[j] for i in range(3) for j in range(i))
    assert rows[0]['seed'] == 9200000000001


@pytest.mark.parametrize('head,candidate_heldout,control_heldout,passed', [
    (1.0, 0.5, 0.5, True), (0.5, 0.5, 0.5, False), (1.0, 0.0, 0.5, False)])
def test_all_gates_and_paired_sample_units(head, candidate_heldout, control_heldout, passed):
    p = plans()
    result = suite.summarize({'candidate': report(p['candidate'], head, candidate_heldout),
                             'incumbent_control': report(p['incumbent_control'], heldout=control_heldout)}, p)
    assert result['all_gates_pass'] is passed
    assert result['head_to_head']['independent_deals'] == 4096
    assert [r['difference']['independent_deals'] for r in result['heldout_retention']] == [1024, 4096]
    assert not result['automatic_deployment']
    assert not result['candidate_reselection_permitted']


@pytest.mark.parametrize('corruption', ['mutated', 'count', 'plan', 'incomplete'])
def test_refuses_invalid_evidence(corruption):
    p = plans()['candidate']
    r = copy.deepcopy(report(p))
    if corruption == 'mutated':
        r['parameters_and_buffers_unchanged'] = False
    elif corruption == 'count':
        r['evaluations'][0]['outcomes'].pop()
    elif corruption == 'plan':
        r['plan']['opponents'][0]['seed'] += 1
    else:
        r['status'] = 'running'
    with pytest.raises(RuntimeError):
        suite.checked_outcomes(r, p)


def prepare(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    base = root / 'campaign/overnight'
    (base / 'locks').mkdir(parents=True)
    names = ['cnn_ppo', 'transformer_ppo', 'cnn_rnad']
    for name in names:
        (base / name).mkdir()
    (base / 'recipe.json').write_text(json.dumps({'contenders': names}))
    (base / 'state.json').write_text(json.dumps({'status': 'stopped', 'workers': {},
        'controller': 'multi_gpu', 'reason': 'training_deadline', 'failure': None}))
    for constant, relative in [('INCUMBENT', 'incumbent.pt'), ('ORIGINAL', 'original.pt'), ('BASELINE', 'baseline.pt')]:
        path = root / relative
        path.write_bytes(constant.encode())
        monkeypatch.setattr(suite, constant, (relative, suite.digest(path)))
    candidate = root / 'selected.pt'
    candidate.write_bytes(b'selected')
    (root / 'scripts').mkdir()
    (root / 'scripts/evaluate_exact_policy.py').write_text('# mocked evaluator\n')
    monkeypatch.setattr(suite, 'trainer_processes', lambda _: [])
    return argparse.Namespace(root=root, candidate=Path('selected.pt'), candidate_sha=suite.digest(candidate),
                              selection_reason='Frozen development comparison'), base


def test_mock_end_to_end_freezes_before_outcomes_and_rejects_reselection(tmp_path, monkeypatch):
    args, base = prepare(tmp_path, monkeypatch)
    out = base / 'final-confirmation'
    calls = []
    def fake_run(command, **kwargs):
        assert (out / 'chosen-candidate.json').exists()
        assert (out / 'candidate-plan.json').exists() and (out / 'incumbent_control-plan.json').exists()
        assert (base / 'gpu-0.pause').exists()
        assert 0 < kwargs['timeout'] <= 360
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0'
        plan = json.loads(Path(command[command.index('--plan') + 1]).read_text())
        Path(command[command.index('--output') + 1]).write_text(json.dumps(report(plan)))
        calls.append(command)
    monkeypatch.setattr(suite.subprocess, 'run', fake_run)
    previous = Path.cwd()
    try:
        suite.run(args)
        assert len(calls) == 2
        assert json.loads((out / 'summary.json').read_text())['all_gates_pass']
        assert not (base / 'gpu-0.pause').exists()
        suite.run(args)
        assert len(calls) == 2
        args.selection_reason = 'Changed after seeing outcomes'
        with pytest.raises(RuntimeError, match='reselection'):
            suite.run(args)
    finally:
        os.chdir(previous)


def test_timeout_releases_pause_and_locks(tmp_path, monkeypatch):
    args, base = prepare(tmp_path, monkeypatch)
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(suite.subprocess, 'run', timeout)
    previous = Path.cwd()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            suite.run(args)
        assert not (base / 'gpu-0.pause').exists()
        with suite.lock(base / 'locks/gpu-0.lock'), suite.lock(base / 'controller.lock'):
            pass
        assert not (base.parent / 'final-confirmation/summary.json').exists()
    finally:
        os.chdir(previous)


def test_active_training_refused(tmp_path, monkeypatch):
    _, base = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(suite, 'trainer_processes', lambda _: [123])
    with pytest.raises(RuntimeError, match='Trainer processes remain'):
        with suite.stopped_training_locks(base):
            pytest.fail('Must not obtain final evaluation permission')
