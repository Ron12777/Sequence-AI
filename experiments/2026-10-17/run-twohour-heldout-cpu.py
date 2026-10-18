"""One local CPU-only fixed-size heldout replication; no rental controls."""
import fcntl,subprocess,sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'experiments/2026-10-17/twohour-heldout-replication'
lock=(out/'suite.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for name in ['cnn_ppo','incumbent_control']:
 subprocess.run([sys.executable,'-u',str(root/'scripts/evaluate_exact_policy.py'),'--plan',str(out/f'{name}-plan.json'),'--output',str(out/f'{name}.json')],cwd=root,check=True)
(out/'suite-complete.json').write_text(json.dumps({'status':'complete','games_per_model':8192}))
