"""Bounded local CPU evaluation; remote training remains uninterrupted."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/2026-10-18/twohour-development'


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT / 'suite.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        for name in ['cnn_ppo', 'transformer_ppo']:
            output = OUT / (name + '.json')
            if output.exists() and json.loads(output.read_text()).get('status') == 'complete':
                results[name] = {'status': 'already_complete'}
                continue
            with (OUT / (name + '.log')).open('a') as log:
                try:
                    result = subprocess.run(
                        [sys.executable, '-u', 'scripts/evaluate_exact_policy.py',
                         '--plan', str(OUT / (name + '-plan.json')), '--output', str(output)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                        timeout=3600,
                    )
                    results[name] = {'returncode': result.returncode}
                except subprocess.TimeoutExpired:
                    results[name] = {'status': 'timeout', 'partial_results_not_final': True}
        (OUT / 'suite-complete.json').write_text(json.dumps({
            'utc': datetime.now(timezone.utc).isoformat(), 'jobs': results,
            'purpose': 'Development only; no promotion or early training termination.'
        }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    OUT = parser.parse_args().output_dir.resolve()
    main()
