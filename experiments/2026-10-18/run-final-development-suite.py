"""Wait for natural shutdown, preserve continuations, compare six fixed candidates."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

spec = importlib.util.spec_from_file_location('confirmation', Path(__file__).with_name('run-final-confirmation-suite.py'))
confirmation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confirmation)
STOP = datetime.fromisoformat('2026-10-18T14:00:00+00:00').timestamp()
WAIT_UNTIL = STOP + 120
SEED = 4200400000001
MAX_SECONDS = 600
MODELS = ('cnn_ppo', 'transformer_ppo')


def snapshot(source, target):
    sha = confirmation.digest(source)
    if not target.exists():
        os.link(source, target)
    if confirmation.digest(target) != sha:
        raise RuntimeError(f'Immutable snapshot mismatch: {target}')
    return {'checkpoint': str(target), 'sha256': sha, 'bytes': target.stat().st_size}


def make_plan(candidate, incumbent):
    return {'candidate': {k: candidate[k] for k in ('checkpoint', 'sha256')},
            'opponents': [{'label': 'validated_oct17_cnn', 'kind': 'exact', **incumbent,
                           'games': 4096, 'seed': SEED}],
            'device': 'cuda', 'threads': 1, 'env_threads': 1, 'batch_size': 256,
            'purpose': 'Development only: six predeclared checkpoints on shared fresh paired complete games. Reserved final seeds untouched.'}


def execute(root, base, out, state):
    snapshots = out / 'checkpoints'
    snapshots.mkdir(exist_ok=True)
    incumbent = {'checkpoint': str(root / confirmation.INCUMBENT[0]), 'sha256': confirmation.INCUMBENT[1]}
    if confirmation.digest(incumbent['checkpoint']) != incumbent['sha256']:
        raise RuntimeError('Incumbent checksum mismatch')
    milestone_manifest = json.loads((base / 'milestones/manifest.json').read_text())
    selected, history = {}, {}
    for name in MODELS:
        selected[name + '_final'] = snapshot(base / name / 'latest.pt', snapshots / (name + '-final.pt'))
        history[name] = []
        history_dir = snapshots / (name + '-history')
        history_dir.mkdir(exist_ok=True)
        for source in sorted((base / name).glob('historical_*.pt')):
            history[name].append(snapshot(source, history_dir / source.name))
        if not history[name]:
            raise RuntimeError(f'Missing historical pool: {name}')
        for hour in (5, 8):
            key = f'{name}-additional-{hour}h'
            source = base / 'milestones' / (key + '.pt')
            if confirmation.digest(source) != milestone_manifest[key]['sha256']:
                raise RuntimeError('Milestone checksum mismatch')
            selected[f'{name}_{hour}h'] = snapshot(source, snapshots / source.name)
    confirmation.immutable_json(out / 'snapshot-manifest.json', {
        'trainer_state': state, 'snapshots': selected, 'historical_pools': history,
        'recipe': json.loads((base / 'recipe.json').read_text()),
        'source_sha256': confirmation.digest(Path(__file__)),
        'selection_rule': 'Highest score on these shared developmental deals; ties use lexicographic candidate name. One final candidate only; final outcomes never used to reselect.'})
    plans = {name: make_plan(item, incumbent) for name, item in selected.items()}
    for name, plan in plans.items():
        confirmation.immutable_json(out / (name + '-plan.json'), plan)
    pause = base / 'gpu-0.pause'
    pause.touch(exist_ok=False)
    try:
        with confirmation.lock(base / 'locks/gpu-0.lock'):
            start = time.monotonic()
            results = []
            for name, plan in plans.items():
                if (out / (name + '.json')).exists():
                    raise RuntimeError('Partial prior outcomes exist; inspect before continuing, never rerun silently')
                remaining = MAX_SECONDS - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError('Development suite exceeded fixed runtime')
                with (out / (name + '.log')).open('a') as log:
                    subprocess.run([sys.executable, '-u', 'scripts/evaluate_exact_policy.py',
                                    '--plan', str(out / (name + '-plan.json')), '--output', str(out / (name + '.json'))],
                                   check=True, timeout=remaining, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
                report = json.loads((out / (name + '.json')).read_text())
                confirmation.checked_outcomes(report, plan)
                row = report['evaluations'][0]
                results.append({'name': name, 'candidate': plan['candidate'],
                                **{k: row[k] for k in ('games', 'score', 'paired_score_95ci')},
                                'elapsed_seconds': report['elapsed_seconds']})
                print(json.dumps(results[-1]), flush=True)
            for item in selected.values():
                if confirmation.digest(item['checkpoint']) != item['sha256']:
                    raise RuntimeError('Checkpoint changed during development')
            chosen = sorted(results, key=lambda x: (-x['score'], x['name']))[0]
            confirmation.atomic_json(out / 'suite-complete.json', {
                'status': 'complete', 'utc': datetime.now(timezone.utc).isoformat(),
                'results': results, 'selected_for_confirmation': chosen,
                'reserved_final_seeds_used': False, 'automatic_promotion': False,
                'gpu_lease_seconds': time.monotonic() - start})
    finally:
        pause.unlink(missing_ok=True)  # Lease is released before removing the marker.


def run(root):
    root = root.resolve()
    os.chdir(root)
    base = root / 'campaign/overnight'
    out = base / 'final-development'
    out.mkdir(exist_ok=True)
    with confirmation.lock(out / 'suite.lock'):
        if (out / 'suite-complete.json').exists():
            return
        while time.time() < STOP:
            time.sleep(min(5, STOP - time.time()))
        while True:
            if time.time() > WAIT_UNTIL:
                raise TimeoutError('No clean shutdown by14:02UTC; no trainer signals sent')
            state = json.loads((base / 'state.json').read_text())
            if state.get('status') != 'stopped' or state.get('workers'):
                time.sleep(1)
                continue
            with confirmation.lock(base / 'finalization.lock'):
                with confirmation.stopped_training_locks(base) as stopped:
                    execute(root, base, out, stopped)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/sequence'))
    args = parser.parse_args()
    try:
        run(args.root)
    except BaseException as error:
        out = args.root.resolve() / 'campaign/overnight/final-development'
        if out.exists():
            confirmation.atomic_json(out / f'failure-{os.getpid()}.json', {'error': repr(error), 'trainer_signals_sent': False})
        raise
