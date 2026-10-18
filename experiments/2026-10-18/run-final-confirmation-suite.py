"""Fixed final confirmation for one preselected immutable CNN; never launches training."""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time

INCUMBENT = ('artifacts/checkpoints/exact-final/cnn_ppo-final.pt',
             '59d1ed9cc5364e9a67e95fd150286cfdb22f3e0454b987a96858c349ca63dd45')
ORIGINAL = ('artifacts/checkpoints/overnight-round0.pt',
            '884ecab0d77a2274749153b743e1412aa2fbf63401bfd7a45620f84f09114c85')
BASELINE = ('artifacts/checkpoints/small-standard.pt',
            'd82402b1246b1e10641aef7bc17ec470f8557729f6ab7f10f16a4e5acc983cef')
SEED_BASE = 9200000000001
MAX_SECONDS = 360


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, payload):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def immutable_json(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f'Refusing candidate reselection or changed fixed plan: {path}')
    else:
        atomic_json(path, payload)


@contextlib.contextmanager
def lock(path):
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def trainer_processes(base):
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            args = entry.joinpath('cmdline').read_bytes().decode().split('\0')
            if not any(Path(arg).name in ('train_exact_multigpu.py', 'train_exact_policy.py') for arg in args):
                continue
            if '--output-dir' not in args:
                continue
            output = Path(args[args.index('--output-dir') + 1])
            if not output.is_absolute():
                output = entry.joinpath('cwd').resolve() / output
            if output.resolve() == base.resolve():
                found.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


@contextlib.contextmanager
def stopped_training_locks(base):
    with contextlib.ExitStack() as stack:
        for name in ('controller.lock', 'supervisor.lock'):
            stack.enter_context(lock(base / name))
        recipe = json.loads((base / 'recipe.json').read_text())
        names = recipe['contenders']
        if not isinstance(names, list) or not names or any(
            not isinstance(n, str) or not n or Path(n).name != n or n in ('.', '..') for n in names
        ) or len(names) != len(set(names)):
            raise ValueError('Invalid saved contender names')
        for name in names:
            stack.enter_context(lock(base / name / 'worker.lock'))
        state = json.loads((base / 'state.json').read_text())
        if (state.get('status') != 'stopped' or state.get('workers') or state.get('failure')
                or state.get('controller') != 'multi_gpu' or state.get('reason') != 'training_deadline'):
            raise RuntimeError('Training has not completed its clean natural shutdown')
        active = trainer_processes(base)
        if active:
            raise RuntimeError(f'Trainer processes remain: {active}')
        yield state


def make_plans(root, candidate):
    def spec(item):
        return {'checkpoint': str(root / item[0]), 'sha256': item[1]}
    incumbent = spec(INCUMBENT)
    heldout = [
        {'label': 'original_search_128', 'kind': 'champion', **spec(ORIGINAL),
         'games': 2048, 'seed': SEED_BASE + 1000000, 'simulations': 128},
        {'label': 'heldout_baseline_32', 'kind': 'champion', **spec(BASELINE),
         'games': 8192, 'seed': SEED_BASE + 2000000, 'simulations': 32},
    ]
    shared = {'device': 'cuda', 'threads': 1, 'env_threads': 1, 'batch_size': 256,
              'purpose': 'FINAL fixed confirmation of one previously selected candidate. Reserved fresh paired deals, greedy exact policy, no candidate reselection or sample extension. Head-to-head lower95CI>0.5 and both heldout paired difference lower95CI>-0.02 required. No automatic deployment.'}
    return {
        'candidate': {**shared, 'candidate': candidate,
                      'opponents': [{'label': 'validated_oct17_cnn', 'kind': 'exact', **incumbent,
                                     'games': 8192, 'seed': SEED_BASE}, *heldout]},
        'incumbent_control': {**shared, 'candidate': incumbent, 'opponents': heldout},
    }


def interval(values):
    mean = statistics.mean(values)
    half = 1.96 * (statistics.variance(values) / len(values)) ** 0.5
    return {'mean': mean, 'paired95ci': [mean - half, mean + half], 'independent_deals': len(values)}


def checked_outcomes(report, plan):
    if report['status'] != 'complete' or report.get('parameters_and_buffers_unchanged') is not True:
        raise RuntimeError('Incomplete evaluation or model mutation')
    if report['plan'] != plan:
        raise RuntimeError('Evaluation did not use the predeclared fixed plan')
    rows = report['evaluations']
    if len(rows) != len(plan['opponents']) or len({r['label'] for r in rows}) != len(rows):
        raise RuntimeError('Missing or duplicate opponent evaluations')
    result = {}
    for expected in plan['opponents']:
        row = next((r for r in rows if r['label'] == expected['label']), None)
        if row is None or row['games'] != expected['games'] or len(row['outcomes']) != expected['games']:
            raise RuntimeError('Fixed sample count mismatch')
        if any(value not in (0.0, 0.5, 1.0) for value in row['outcomes']):
            raise RuntimeError('Invalid complete-game outcomes')
        result[row['label']] = row['outcomes']
    return result


def summarize(reports, plans):
    candidate = checked_outcomes(reports['candidate'], plans['candidate'])
    control = checked_outcomes(reports['incumbent_control'], plans['incumbent_control'])
    head = candidate['validated_oct17_cnn']
    head_summary = interval([(head[i] + head[i + 1]) / 2 for i in range(0, len(head), 2)])
    head_summary.update(games=len(head), gate_pass=head_summary['paired95ci'][0] > 0.5)
    heldout = []
    for label, b in control.items():
        a = candidate[label]
        pa = next(p for p in plans['candidate']['opponents'] if p['label'] == label)
        pb = next(p for p in plans['incumbent_control']['opponents'] if p['label'] == label)
        if pa != pb:
            raise RuntimeError('Heldout comparisons must use identical opponent settings and paired deals')
        delta = interval([(a[i] - b[i] + a[i + 1] - b[i + 1]) / 2 for i in range(0, len(a), 2)])
        heldout.append({'label': label, 'games_per_model': len(a), 'candidate_score': statistics.mean(a),
                        'incumbent_score': statistics.mean(b), 'difference': delta,
                        'practical_retention_margin': 0.02, 'gate_pass': delta['paired95ci'][0] > -0.02})
    passed = head_summary['gate_pass'] and all(row['gate_pass'] for row in heldout)
    return {'status': 'complete', 'candidate': plans['candidate']['candidate'],
            'incumbent': plans['incumbent_control']['candidate'], 'head_to_head': head_summary,
            'heldout_retention': heldout, 'all_gates_pass': passed,
            'recommendation': 'Eligible for promotion after artifact validation' if passed else 'Retain validated two-hour CNN',
            'automatic_deployment': False, 'candidate_reselection_permitted': False,
            'elapsed_evaluator_seconds': sum(r['elapsed_seconds'] for r in reports.values()),
            'interval_method': 'Normal95% interval over independent deal averages; both swapped seats grouped. Heldout deltas pair candidate/control on the same deal and seat. Fixed samples; no extension after results.'}


def run(args):
    root = args.root.resolve()
    candidate_path = (root / args.candidate).resolve()
    if candidate_path.name == 'latest.pt':
        raise ValueError('Select an immutable milestone/final snapshot, not mutable latest.pt')
    candidate = {'checkpoint': str(candidate_path), 'sha256': args.candidate_sha.lower()}
    if candidate['sha256'] == INCUMBENT[1]:
        raise ValueError('Selected candidate is the retained incumbent; no final test is needed')
    os.chdir(root)
    base, out = root / 'campaign/overnight', root / 'campaign/overnight/final-confirmation'
    out.mkdir(exist_ok=True)
    with lock(out.parent / 'finalization.lock'), lock(out / 'suite.lock'):
        plans = make_plans(root, candidate)
        for plan in plans.values():
            for item in [plan['candidate'], *plan['opponents']]:
                if digest(item['checkpoint']) != item['sha256']:
                    raise RuntimeError(f'Checkpoint checksum mismatch: {item["checkpoint"]}')
        selection = {'candidate': candidate, 'incumbent': plans['incumbent_control']['candidate'],
                     'selection_reason': args.selection_reason,
                     'protocol': {'head_to_head_games': 8192, 'search128_games_each': 2048,
                                  'baseline32_games_each': 8192, 'retention_margin': 0.02,
                                  'reserved_seed_base': SEED_BASE, 'maximum_gpu_seconds': MAX_SECONDS},
                     'no_reselection_or_sample_extension': True}
        if not (out / 'chosen-candidate.json').exists() and any((out / (n + '.json')).exists() for n in plans):
            raise RuntimeError('Existing outcomes lack an immutable candidate-selection record')
        # Freeze the choice and every plan before any subprocess can see outcomes.
        immutable_json(out / 'chosen-candidate.json', selection)
        for name, plan in plans.items():
            immutable_json(out / (name + '-plan.json'), plan)
        if (out / 'summary.json').exists():
            print((out / 'summary.json').read_text(), flush=True)
            return
        with stopped_training_locks(base) as state:
            atomic_json(out / 'execution-provenance.json', {'trainer_state': state,
                'source_sha256': {str(p): digest(p) for p in (Path(__file__), root / 'scripts/evaluate_exact_policy.py')},
                'started_utc': datetime.now(timezone.utc).isoformat(), 'device': 'CUDA_VISIBLE_DEVICES=0'})
            pause = base / 'gpu-0.pause'
            pause.touch(exist_ok=False)
            try:
                with lock(base / 'locks/gpu-0.lock'):
                    started = time.monotonic()
                    reports = {}
                    for name in ('candidate', 'incumbent_control'):
                        remaining = MAX_SECONDS - (time.monotonic() - started)
                        if remaining <= 0:
                            raise TimeoutError('Fixed360second final-confirmation window exceeded')
                        with (out / (name + '.log')).open('a') as log:
                            subprocess.run([sys.executable, '-u', 'scripts/evaluate_exact_policy.py',
                                '--plan', str(out / (name + '-plan.json')), '--output', str(out / (name + '.json'))],
                                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
                        reports[name] = json.loads((out / (name + '.json')).read_text())
                    for plan in plans.values():
                        for item in [plan['candidate'], *plan['opponents']]:
                            if digest(item['checkpoint']) != item['sha256']:
                                raise RuntimeError('Checkpoint changed during final confirmation')
                    summary = summarize(reports, plans)
                    summary['gpu_lease_wall_seconds'] = time.monotonic() - started
                    atomic_json(out / 'summary.json', summary)
                    print(json.dumps(summary), flush=True)
            finally:
                pause.unlink(missing_ok=True)  # GPU lock released before pause removal.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/sequence'))
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-sha', required=True)
    parser.add_argument('--selection-reason', required=True, help='Development evidence used before opening reserved outcomes')
    args = parser.parse_args()
    if len(args.candidate_sha) != 64 or any(c not in '0123456789abcdefABCDEF' for c in args.candidate_sha):
        parser.error('candidate-sha must be a complete SHA256')
    def terminate(*_):
        raise SystemExit('Termination requested; fixed partial evidence retained, no sample extension')
    signal.signal(signal.SIGTERM, terminate)
    run(args)


if __name__ == '__main__':
    main()
