"""Read-only compact runtime observation; no training, evaluation, or mutation."""
import json,math,statistics,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/workspace/sequence/campaign/overnight')
r={'utc':datetime.now(timezone.utc).isoformat(),'state':json.loads((root/'state.json').read_text()) if (root/'state.json').exists() else None,'models':{}}
for c in ['cnn_ppo','transformer_ppo']:
 p=root/c/'metrics.jsonl'
 if not p.exists():continue
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-8000000)); lines=f.read().decode().splitlines()
 rows=[]
 for line in lines:
  try: rows.append(json.loads(line))
  except ValueError:pass
 rows=rows[-100:]
 if not rows:continue
 last=rows[-1]; recent=rows[-min(50,len(rows)):]
 d={k:last.get(k) for k in ['training_seconds','games','updates','iterations','policy_entropy','approximate_kl','loss','rollout_value_explained_variance','trunk_learning_rate','new_learning_rate']}
 d['finite']=all(v is None or not isinstance(v,(int,float)) or math.isfinite(v) for v in d.values())
 d['sample_iterations']=len(recent)
 for k in ['iteration_seconds','update_seconds','rollout_seconds']:
  vals=[x[k] for x in recent if isinstance(x.get(k),(float,int))]
  if vals:d[k+'_mean']=statistics.mean(vals)
 d['games_per_second']=2048/d['iteration_seconds_mean']
 d['checkpoint_mtime_utc']=datetime.fromtimestamp((root/c/'latest.pt').stat().st_mtime,timezone.utc).isoformat()
 r['models'][c]=d
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True).strip().splitlines()
r['processes']=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=p.read_bytes().decode().split('\0')
  if any(Path(v).name in ['train_exact_multigpu.py','train_exact_policy.py','sequence-watchdog.py','capture-milestones.py'] for v in a):r['processes'].append({'pid':int(p.parent.name),'args':a})
 except OSError:pass
print(json.dumps(r,indent=2))
