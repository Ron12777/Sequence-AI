"""Sequential local CPU comparison; never reserves or interrupts rental GPUs."""
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/2026-10-17/4090-hour-evaluation'
lock = (OUT / 'suite.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
manifest = json.loads((OUT / 'milestone-manifest.json').read_text())
for contender in ['cnn_ppo', 'transformer_ppo', 'cnn_rnad']:
    entry = manifest[contender + '-4090-1h']
    path = ROOT / 'artifacts/checkpoints/exact-4090-hour' / entry['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    plan = {
        'candidate': {'checkpoint': str(path), 'sha256': entry['sha256']},
        'opponents': [
            {'label': 'validated_exact_cnn_1h', 'kind': 'exact',
             'checkpoint': str(ROOT / 'artifacts/checkpoints/exact-cnn-onehour.pt'),
             'sha256': 'f031c72501b1f85b5147b78352febcb2c494ee240505fcb89500fc203fb6c1cb',
             'games': 4096, 'seed': 2400100000001},
            {'label': 'retained_cnn_7h48', 'kind': 'history',
             'checkpoint': str(ROOT / 'artifacts/checkpoints/board-pre-migration-final.pt'),
             'sha256': 'cd18cad504fa6bc99689152401f7b60fdd7c5f55d7c37e662e5655ab04ed3229',
             'games': 2048, 'seed': 2400200000001}],
        'device': 'cpu', 'threads': 1, 'env_threads': 1, 'batch_size': 256,
        'purpose': 'Developmental matched additional RTX4090-hour controls; inherited RTX5090 budgets differ. Original threshold2 RNaD, not corrected continuation. Fresh paired deals; no final confirmation seeds; evaluation CPU cost separate from training.'}
    plan_path = OUT / (contender + '-plan.json')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_exact_policy.py'),
                    '--plan', str(plan_path), '--output', str(OUT / (contender + '.json'))],
                   cwd=ROOT, check=True)
(OUT / 'suite-complete.json').write_text(json.dumps({'status': 'complete', 'contenders': 3}))
