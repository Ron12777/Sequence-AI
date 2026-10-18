"""Deal-clustered recovery comparison, keeping policy sampling replicas dependent."""
import json,hashlib
from pathlib import Path
import numpy as np
base=Path(__file__).resolve().parent/'rnad-recovery-1h'
if (base/'retrieval-sha256.json').exists():
 for n,h in json.loads((base/'retrieval-sha256.json').read_text()).items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h
reports={n:json.loads((base/(n+'.json')).read_text()) for n in ['corrected_greedy','corrected_sample1','corrected_sample2','prechange_greedy','prechange_sample1','prechange_sample2']}
assert all(r['status']=='complete' and r['parameters_and_buffers_unchanged'] for r in reports.values())
def stats(x):
 half=1.96*x.std(ddof=1)/np.sqrt(len(x));return {'score':float(x.mean()),'paired95ci':[float(x.mean()-half),float(x.mean()+half)]}
summary={'scope':'Corrected RNaD after>=1 additional traininghour versus preserved prechange state. Greedy and learnedT1 evaluated separately on identical fresh paireddeals. Sampling replicas averaged within eachdeal before confidence interval; no replica-as-independent-deal claim. Both opponents fixedgreedy exact policies.','comparisons':[]}
for opponent in ['validated_cnn_2h','previous_cnn_1h_anchor']:
 for mode in ['greedy','sample']:
  suffixes=['greedy'] if mode=='greedy' else ['sample1','sample2']
  scores={}
  for candidate in ['corrected','prechange']:
   arrays=[]
   for suffix in suffixes:
    r=reports[candidate+'_'+suffix];e=next(x for x in r['evaluations'] if x['label']==opponent); arrays.append(np.array(e['outcomes']).reshape(-1,2).mean(1))
   scores[candidate]=np.stack(arrays).mean(0)
  assert len(scores['corrected'])==len(scores['prechange'])
  summary['comparisons'].append({'opponent':opponent,'candidate_mode':mode,'independent_deals':len(scores['corrected']),'action_replicates':len(suffixes),'games_per_candidate':len(scores['corrected'])*2*len(suffixes),'corrected':stats(scores['corrected']),'prechange':stats(scores['prechange']),'corrected_minus_prechange':stats(scores['corrected']-scores['prechange'])})
summary['gpu_evaluation_seconds']=sum(r['elapsed_seconds'] for r in reports.values())
summary['checkpoint_metadata']=json.loads((base/'checkpoint-metadata.json').read_text())
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='checkpoint_metadata'},indent=2))
