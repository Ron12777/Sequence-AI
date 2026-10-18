"""Read-only remote snapshot; CPU checkpoint loads do not reserve a GPU."""
import datetime,json,math,statistics,subprocess
from pathlib import Path
import torch
base=Path('campaign/exact/sustained')
result={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'state':json.loads((base/'state.json').read_text()),'contenders':{},
        'processes':[l for l in subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines() if any(s in l for s in ['train_exact_multigpu.py','campaign_watchdog.py','capture-sequence-milestones.py'])],
        'pause_markers':[str(p) for p in base.glob('*.pause')]}
keys=['iteration_seconds','rollout_seconds','update_seconds','approximate_kl','policy_entropy','entropy','regularizer_kl','rollout_value_explained_variance','gradient_norm','critic_loss']
for name in ['cnn_ppo','transformer_ppo','cnn_rnad']:
 rows=[json.loads(l) for l in (base/name/'metrics.jsonl').read_text().splitlines() if l.strip()]
 recent=rows[-20:]
 ck=torch.load(base/name/'latest.pt',map_location='cpu',weights_only=False)
 entry={'metadata':ck['metadata'],'latest_metrics':rows[-1], 'last20_means':{k:statistics.mean(r[k] for r in recent if k in r) for k in keys if any(k in r for r in recent)}, 'metrics_finite':all(math.isfinite(v) for r in recent for v in r.values() if isinstance(v,(int,float)))}
 if name=='cnn_rnad':
  cfg=dict(ck['training_state']['rnad']['config'])
  cfg={k:('Infinity' if isinstance(v,float) and math.isinf(v) else v) for k,v in cfg.items()}
  entry['rnad_config']=cfg
 result['contenders'][name]=entry
result['gpu_snapshot']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True)
result['milestones']=json.loads(Path('campaign/exact/milestones/manifest.json').read_text())
print(json.dumps(result,indent=2))
