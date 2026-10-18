"""Matched-state forward latency for trained exact CNN and transformer policies."""
import gzip, hashlib, json, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.exact_policy import load_exact_policy
from scripts.evaluate_exact_policy import freeze,verify_frozen

torch.set_num_threads(1)
paths={n:ROOT/'artifacts/checkpoints/exact-twohour'/f'{n}_ppo-mixed-2h.pt' for n in ['cnn','transformer']}
models={n:load_exact_policy(p,'cpu')[0] for n,p in paths.items()}
snapshots={n:freeze(m) for n,m in models.items()}
fixtures=json.loads(gzip.decompress((ROOT/'experiments/2026-10-17/exact-cnn-onehour-browser-fixtures.json.gz').read_bytes()))
all_steps=[s for g in fixtures for s in g['steps']]
steps=[all_steps[i] for i in np.linspace(0,len(all_steps)-1,128,dtype=int)]
def inputs(rows):
 b=torch.tensor([r['board'] for r in rows],dtype=torch.float32).reshape(-1,11,10,10)
 c=torch.tensor([r['context'] for r in rows],dtype=torch.float32)
 h=torch.tensor([r['history'] for r in rows],dtype=torch.long).reshape(-1,128,4)
 a=torch.full((len(rows),max(len(r['actions']) for r in rows)),-1,dtype=torch.long)
 for i,r in enumerate(rows):a[i,:len(r['actions'])]=torch.tensor(r['actions'])
 return b,c,h,a,a>=0
report={'device':'local CPU','cpu':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'torch':torch.__version__,'threads':1,'scope':'Forward pass only, real public observations and exact legal actions; preprocessing and loading excluded. CPU PyTorch, not browser or CUDA. Alternating model order; same states per model.','state_count':128,'results':[],'checkpoints':{n:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'parameters':sum(v.numel() for v in models[n].parameters())} for n,p in paths.items()}}
with torch.inference_mode():
 for batch in [1,32,128]:
  xs=[inputs(steps[i:i+batch]) for i in range(0,128,batch)]
  for model in models.values():
   for x in xs[:min(8,len(xs))]:model(*x)
  samples={n:[] for n in models}
  repetitions=128 if batch==1 else 24
  for rep in range(repetitions):
   x=xs[rep%len(xs)]
   for n in (['cnn','transformer'] if rep%2 else ['transformer','cnn']):
    start=time.perf_counter(); y=models[n](*x); elapsed=(time.perf_counter()-start)*1000
    assert torch.isfinite(y[0][x[-1]]).all() and torch.isfinite(y[1]).all()
    samples[n].append(elapsed)
  for n,vals in samples.items():report['results'].append({'model':n,'batch':batch,'median_batch_ms':statistics.median(vals),'p95_batch_ms':float(np.percentile(vals,95)),'median_ms_per_position':statistics.median(vals)/batch,'samples':len(vals)})
for n,m in models.items():verify_frozen(m,snapshots[n])
report['models_unchanged']=True
(ROOT/'experiments/2026-10-17/twohour-inference-cost.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
