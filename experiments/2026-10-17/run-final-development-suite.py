"""After natural training shutdown, snapshot finalists and run development only.

Prepared for manual launch by the campaign owner; never sends trainer signals.
The final confirmation seed reservation is deliberately excluded.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

STOP_AT = datetime.fromisoformat('2026-10-17T23:30:00+00:00').timestamp()
WAIT_UNTIL = datetime.fromisoformat('2026-10-17T23:32:00+00:00').timestamp()
INCUMBENT_SHA = 'bfcb0193d48dc0d40449fdbf440df44a7b0400cc33d996e412b1b491b75d6ed8'
GREEDY_SEED = 3100100000001
SAMPLE_SEED = GREEDY_SEED


def utc():
    return datetime.now(timezone.utc).isoformat()


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


def recorded_json(path, payload):
    """Reruns may resume identical plans, never silently replace a frozen plan."""
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f'Existing immutable record differs: {path}')
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
    """Check actual processes, in addition to authoritative leases and state."""
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
                found.append({'pid': int(entry.name), 'command': args})
        except (FileNotFoundError, ProcessLookupError):
            continue  # Normal process teardown race.
    return found


def saved_recipe(base):
    recipe = json.loads((base / 'recipe.json').read_text())
    names = recipe.get('contenders')
    if not isinstance(names, list) or not names or any(
        not isinstance(name, str) or not name or Path(name).name != name
        or name in ('.', '..') or not all(c.isascii() and (c.isalnum() or c == '_') for c in name)
        for name in names
    ):
        raise ValueError('Saved recipe contains invalid contender names')
    if len(set(names)) != len(names) or 'cnn_ppo' not in names:
        raise ValueError('Saved recipe must contain unique contenders including cnn_ppo')
    return recipe


def wait_for_shutdown(base):
    """Yield a stack holding all trainer leases after a clean natural stop."""
    while True:
        if time.time() > WAIT_UNTIL:
            raise TimeoutError('No clean trainer shutdown by23:32UTC; no signals sent')
        if time.time() < STOP_AT:
            time.sleep(min(5, STOP_AT - time.time()))
            continue
        state = json.loads((base / 'state.json').read_text())
        if state.get('status') != 'stopped' or state.get('workers'):
            time.sleep(1)
            continue
        if state.get('failure'):
            raise RuntimeError(f'Training stopped with failure; owner must inspect: {state}')
        if state.get('controller') != 'multi_gpu' or state.get('reason') != 'training_deadline':
            raise RuntimeError(f'Unexpected shutdown state; refusing to call checkpoints final: {state}')
        stack = contextlib.ExitStack()
        try:
            for path in [base / 'controller.lock', base / 'supervisor.lock']:
                stack.enter_context(lock(path))
            # Read the actual recipe only after excluding controller restarts;
            # a proposed replica need not have been created or registered.
            recipe = saved_recipe(base)
            for name in recipe['contenders']:
                stack.enter_context(lock(base / name / 'worker.lock'))
            if trainer_processes(base):
                stack.close()
                time.sleep(1)
                continue
            latest = json.loads((base / 'state.json').read_text())
            if latest != state:
                stack.close()
                continue
            return stack, state, recipe
        except BlockingIOError:
            stack.close()
            time.sleep(1)
        except BaseException:
            stack.close()
            raise


def snapshot(source, destination):
    # Trainer's latest.pt is atomically replaced; hardlinks preserve the exact
    # complete payload/optimizer/RNG without copying intermediate checkpoints.
    source_sha = digest(source)
    if destination.exists():
        if digest(destination) != source_sha:
            raise RuntimeError(f'Snapshot already exists with different weights: {destination}')
    else:
        os.link(source, destination)
    if digest(destination) != source_sha:
        raise RuntimeError('Snapshot checksum mismatch')
    import torch
    payload = torch.load(destination, map_location='cpu', weights_only=True)
    continuation = payload.get('training_state', {})
    rnad = continuation.get('rnad')
    return {'checkpoint': str(destination), 'source': str(source), 'sha256': source_sha,
            'bytes': destination.stat().st_size, 'metadata': payload['metadata'],
            'continuation_keys': sorted(continuation),
            'rnad_config': rnad['config'] if rnad else None,
            'rnad_learner_steps': rnad['learner_steps'] if rnad else None}


def corrected_milestones(milestones, snapshots):
    labels = {'cnn_rnad_threshold8_1h': ('cnn_rnad-threshold8-1h', 9016.89391744716),
              'cnn_rnad_threshold8_2h': ('cnn_rnad-threshold8-2h', 12616.89391744716)}
    while True:
        manifest = json.loads((milestones / 'manifest.json').read_text())
        if all(key in manifest for key, _ in labels.values()) or time.time() >= WAIT_UNTIL:
            break
        time.sleep(max(0, min(5, WAIT_UNTIL - time.time())))
    found, unmet = {}, {}
    for name, (key, minimum) in labels.items():
        if key not in manifest:
            unmet[name] = {'reason': 'Named corrected-duration milestone was not captured by23:32UTC',
                           'minimum_cumulative_training_seconds': minimum}
            continue
        item = manifest[key]
        source = milestones / item['file']
        if digest(source) != item['sha256']:
            raise RuntimeError(f'Corrected milestone checksum mismatch: {key}')
        captured = snapshot(source, snapshots / (key + '.pt'))
        metadata = captured['metadata']
        if metadata['training_seconds'] < minimum or captured['rnad_config']['nerd_threshold'] != 8:
            raise RuntimeError(f'Corrected milestone duration/config verification failed: {key}')
        if not any(change.get('parameter') == 'training_state.rnad.config.nerd_threshold'
                   and change.get('after') == 8
                   for change in metadata.get('recipe_amendments', [])):
            raise RuntimeError(f'Corrected milestone lacks persisted threshold amendment: {key}')
        captured['minimum_cumulative_training_seconds'] = minimum
        found[name] = captured
    return found, unmet


def paired_sample_summary(out, names):
    # Two action draws and both seats share one deal, so average all four
    # outcomes before treating deals as independent observations.
    per_deal = {}
    for name in names:
        replicas = [json.loads((out / f'{name}_sample{i}.json').read_text())['evaluations'][0]['outcomes']
                    for i in (1, 2)]
        if any(len(row) != 1024 for row in replicas):
            raise ValueError('Expected exactly1024 games per fixed action replica')
        per_deal[name] = [sum(row[index] + row[index + 1] for row in replicas) / 4
                          for index in range(0, 1024, 2)]

    def summary(values):
        mean = statistics.mean(values)
        half = 1.96 * (statistics.variance(values) / len(values)) ** 0.5
        return {'mean': mean, 'deal_paired_normal_95ci': [mean - half, mean + half],
                'independent_deals': len(values)}

    result = {'unit': 'One deal, averaged over two seats and two action-seed replicas',
              'candidate_scores': {name: summary(values) for name, values in per_deal.items()},
              'differences_vs_corrected_1h': {}}
    anchor = per_deal.get('cnn_rnad_threshold8_1h')
    if anchor is not None:
        result['differences_vs_corrected_1h'] = {
            name: summary([a - b for a, b in zip(values, anchor)])
            for name, values in per_deal.items() if name != 'cnn_rnad_threshold8_1h'}
    return result


def run(args):
    root = args.root.resolve()
    os.chdir(root)
    base = root / 'campaign/exact/sustained'
    out = root / 'campaign/exact/final-development'
    out.mkdir(exist_ok=True)
    with lock(out / 'suite.lock'):
        if (out / 'suite-complete.json').exists():
            print((out / 'suite-complete.json').read_text(), flush=True)
            return
        started = time.monotonic()
        rnad_samples = args.rnad_samples and not args.cnn_only
        scope = ('User requested CNN delivery and CNN training through23:30UTC. Preserve all continuations actually registered in the saved recipe; '
                 'evaluate mainCNN, any registered CNN/PPO replicas identified by checkpoint metadata, and saved3hCNN. A continuation replica shares parent pre-branch training and is not an independently initialized architecture trial. Corrected RNaD milestone checks and stochastic comparisons '
                 'intentionally omitted, not missing or failed.') if args.cnn_only else 'All-contender developmental comparison'
        trainer_locks, stopped_state, recipe = wait_for_shutdown(base)
        with trainer_locks:
            import torch
            torch.set_num_threads(1)
            snapshots = out / 'checkpoints'
            snapshots.mkdir(exist_ok=True)
            selected = {name + '_final': snapshot(base / name / 'latest.pt', snapshots / (name + '-final.pt'))
                        for name in recipe['contenders']}
            milestones = root / 'campaign/exact/milestones-v2'
            milestone = json.loads((milestones / 'manifest.json').read_text())['cnn_ppo-mixed-3h']
            if digest(milestones / milestone['file']) != milestone['sha256']:
                raise RuntimeError('Saved3hCNN differs from milestone manifest')
            selected['cnn_ppo_3h'] = snapshot(milestones / milestone['file'], snapshots / 'cnn_ppo-mixed-3h.pt')
            corrected, unmet = ({}, {}) if args.cnn_only else corrected_milestones(milestones, snapshots)
            selected.update(corrected)
            incumbent = milestones / 'cnn_ppo-mixed-2h.pt'
            if digest(incumbent) != INCUMBENT_SHA:
                raise RuntimeError('Fixed2hCNN incumbent checksum mismatch')
            ledger = [{'file': p.name, 'sha256': digest(p), 'data': json.loads(p.read_text())}
                      for p in sorted((base / 'hardware_dispatches').glob('*.json'))]
            manifest = {'status': 'snapshotted_after_clean_shutdown', 'trainer_state': stopped_state,
                        'cnn_only': args.cnn_only, 'scope': scope,
                        'snapshots': selected, 'hardware_dispatches': ledger,
                        'unmet_corrected_milestones': unmet,
                        'recipe': recipe,
                        'selection': json.loads((base / 'selection.json').read_text()) if (base / 'selection.json').exists() else None,
                        'source_sha256': {str(p): digest(p) for p in [Path(__file__), root / 'scripts/evaluate_exact_policy.py',
                            root / 'scripts/train_exact_multigpu.py', root / 'scripts/train_exact_policy.py', root / 'src/rnad.py']},
                        'accounting': 'Final continuation budgets may differ after narrowing; all registered contenders preserved. Any CNN replica inherits parent pre-branch training counters and history; do not double-count shared pre-branch compute. Report metadata and hardware ledger. Not a matched-budget architecture ranking, equal FLOPs, or node billing.',
                        'incumbent': {'checkpoint': str(incumbent), 'sha256': INCUMBENT_SHA}}
            recorded_json(out / 'snapshot-manifest.json', manifest)
            jobs = []
            candidates = {name: item for name, item in selected.items()
                          if item['metadata'].get('variant') == 'cnn'
                          and item['metadata'].get('algorithm') == 'ppo'} if args.cnn_only else selected
            if args.cnn_only and not {'cnn_ppo_final', 'cnn_ppo_3h'}.issubset(candidates):
                raise RuntimeError('Main/fixed3h CNN checkpoint metadata does not identify CNN/PPO')
            for name, item in candidates.items():
                jobs.append((name, item, 'greedy', None, 4096, GREEDY_SEED))
            if rnad_samples:
                for name in ('cnn_rnad_final', *corrected):
                    for index, action_seed in enumerate((430071, 430072), 1):
                        jobs.append((f'{name}_sample{index}', selected[name], 'sample', action_seed, 1024, SAMPLE_SEED))
            recorded_json(out / 'suite-plan.json', {
                'cnn_only': args.cnn_only, 'scope': scope,
                'rnad_samples': rnad_samples, 'maximum_gpu_seconds': 600,
                'jobs': [{'name': name, 'checkpoint_sha256': item['sha256'], 'mode': mode,
                          'action_seed': action_seed, 'games': games, 'seed': seed}
                         for name, item, mode, action_seed, games, seed in jobs],
                'reserved_final_seeds_used': False,
            })
            for name, item, mode, action_seed, games, seed in jobs:
                plan = {'candidate': {k: item[k] for k in ('checkpoint', 'sha256')},
                        'opponents': [{'label': 'validated_cnn_2h', 'kind': 'exact', 'checkpoint': str(incumbent),
                                       'sha256': INCUMBENT_SHA, 'games': games, 'seed': seed}],
                        'device': 'cuda', 'threads': 1, 'env_threads': 1, 'batch_size': 256,
                        'purpose': 'Developmental final-checkpoint selection only; common fresh paired deals, fixed2hCNN, reserved9100000000001 final confirmation seeds untouched. No automatic promotion.'}
                if mode == 'sample':
                    plan.update(candidate_action_mode=mode, action_seed=action_seed)
                recorded_json(out / (name + '-plan.json'), plan)
            pause = base / 'gpu-0.pause'
            # Do not replace someone else's reservation; owner may rerun later.
            pause.touch(exist_ok=False)
            try:
                with lock(base / 'locks/gpu-0.lock'):
                    deadline = time.monotonic() + 600
                    results = []
                    for name, *_ in jobs:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError('Fixed600second development suite exceeded')
                        with (out / (name + '.log')).open('a') as log:
                            subprocess.run([sys.executable, '-u', 'scripts/evaluate_exact_policy.py',
                                            '--plan', str(out / (name + '-plan.json')), '--output', str(out / (name + '.json'))],
                                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining,
                                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
                        report = json.loads((out / (name + '.json')).read_text())
                        results.append({'name': name, 'elapsed_seconds': report['elapsed_seconds'],
                                        'evaluations': [{k: row[k] for k in ('label', 'games', 'score', 'paired_score_95ci')}
                                                        for row in report['evaluations']]})
                        print(json.dumps(results[-1]), flush=True)
                    atomic_json(out / 'suite-complete.json', {'status': 'complete', 'utc': utc(),
                        'gpu_suite_seconds': 600 - (deadline - time.monotonic()),
                        'total_wall_seconds_including_wait': time.monotonic() - started,
                        'results': results, 'rnad_samples': rnad_samples,
                        'cnn_only': args.cnn_only, 'scope': scope,
                        'unmet_corrected_milestones': unmet,
                        'rnad_sample_deal_summary': paired_sample_summary(out, ('cnn_rnad_final', *corrected)) if rnad_samples else None,
                        'interpretation': 'Developmental screening only. Sample action replicas share deals and are not independent extra deals. Owner selects one candidate before reserved final confirmation.'})
            finally:
                # The with-lock exits before this pause removal, including on timeout.
                pause.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/sequence'))
    parser.add_argument('--cnn-only', action='store_true',
                        help='Preserve all actual saved contenders; evaluate registered CNN/PPO finals and3hCNN only, without RNaD waits/sampling')
    parser.add_argument('--rnad-samples', action=argparse.BooleanOptionalAction, default=True,
                        help='Run two fixed1024-game T1 replicas for final and available corrected1h/2h RNaD (default enabled)')
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        out = args.root.resolve() / 'campaign/exact/final-development'
        if out.exists():
            # Append distinct failures without overwriting another process's state.
            atomic_json(out / f'failure-{os.getpid()}-{time.time_ns()}.json',
                        {'utc': utc(), 'error': repr(error), 'trainer_signals_sent': False})
        raise


if __name__ == '__main__':
    main()
