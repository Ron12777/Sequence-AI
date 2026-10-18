"""Evaluate immutable three-hour milestones without reserving GPU while waiting."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def interrupted(signum, frame):
    raise InterruptedError(f"Suite interrupted by signal {signum}")


signal.signal(signal.SIGTERM, interrupted)
root = Path('/workspace/sequence')
os.chdir(root)
out = root / 'campaign/exact/threehour-evaluation'
out.mkdir(exist_ok=True)
singleton = (out / 'suite.lock').open('a')
fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (out / 'suite-complete.json').exists():
    raise RuntimeError('Completed suite must not be repeated')
contenders = ['cnn_ppo', 'transformer_ppo', 'cnn_rnad']
started = time.monotonic()
while True:
    manifest = json.loads((root / 'campaign/exact/milestones-v2/manifest.json').read_text())
    if all(name + '-mixed-3h' in manifest for name in contenders):
        break
    if time.monotonic() - started > 2400:
        raise TimeoutError('Three-hour milestones did not arrive within 40 minutes')
    time.sleep(10)

metadata = {name: manifest[name + '-mixed-3h'] for name in contenders}
opponents = [
    dict(label='validated_cnn_2h', kind='exact',
         checkpoint='campaign/exact/milestones-v2/cnn_ppo-mixed-2h.pt',
         sha256='bfcb0193d48dc0d40449fdbf440df44a7b0400cc33d996e412b1b491b75d6ed8',
         games=4096, seed=3000100000001),
    dict(label='previous_cnn_1h_anchor', kind='exact',
         checkpoint='campaign/exact/milestones/cnn_ppo-mixed-1h.pt',
         sha256='f031c72501b1f85b5147b78352febcb2c494ee240505fcb89500fc203fb6c1cb',
         games=2048, seed=3000200000001),
]
names = []
for contender, capture in metadata.items():
    assert capture['actual_training_seconds'] >= 10800
    path = 'campaign/exact/milestones-v2/' + capture['file']
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == capture['sha256']
    modes = [('greedy', None)]
    if contender == 'cnn_rnad':
        modes += [('sample1', 420071), ('sample2', 420072)]
    for mode, action_seed in modes:
        name = contender + '_' + mode
        names.append(name)
        plan = dict(
            candidate=dict(checkpoint=path, sha256=capture['sha256']),
            opponents=[dict(o, games=o['games'] if action_seed is None else 1024)
                       for o in opponents],
            device='cuda', threads=1, env_threads=1, batch_size=256,
            purpose='Developmental mixed-hardware three-hour comparison, common fresh paired deals. '
                    'Shared pretraining separate; these are not equal FLOPs. RNaD includes threshold amendment; '
                    'this is not its two-hour corrected-recipe milestone. No final seeds or automatic promotion.',
        )
        if action_seed is not None:
            plan.update(candidate_action_mode='sample', action_seed=action_seed)
        (out / (name + '-plan.json')).write_text(json.dumps(plan, indent=2) + '\n')
(out / 'checkpoint-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
pause = root / 'campaign/exact/sustained/gpu-0.pause'
pause.touch(exist_ok=False)
lease = None
reserved = time.monotonic()
try:
    lease = (root / 'campaign/exact/sustained/locks/gpu-0.lock').open('a')
    while True:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() - reserved > 120:
                raise TimeoutError('GPU0 did not yield')
            time.sleep(1)
    print('GPU0 acquired; GPU1 continues', flush=True)
    for name in names:
        with (out / (name + '.log')).open('a') as log:
            subprocess.run(
                [sys.executable, '-u', 'scripts/evaluate_exact_policy.py', '--plan',
                 str(out / (name + '-plan.json')), '--output', str(out / (name + '.json'))],
                stdout=log, stderr=subprocess.STDOUT, check=True,
                timeout=max(1, 600 - (time.monotonic() - reserved)),
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0',
                     'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            )
        result = json.loads((out / (name + '.json')).read_text())
        assert result['status'] == 'complete' and result['parameters_and_buffers_unchanged']
        print(json.dumps(dict(name=name, seconds=result['elapsed_seconds'], results=[
            {k: e[k] for k in ['label', 'games', 'score', 'paired_score_95ci']}
            for e in result['evaluations']])), flush=True)
    (out / 'suite-complete.json').write_text(json.dumps(dict(
        status='complete', gpu_lease_wall_seconds=time.monotonic() - reserved)))
finally:
    if lease is not None:
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()
    pause.unlink(missing_ok=True)
    print('GPU0 released; training may resume', flush=True)
