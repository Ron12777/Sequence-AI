"""Summarize independent confirmation and deal-paired heldout controls."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
candidate=json.loads((root/'twohour-confirmation/cnn_ppo.json').read_text())
control=json.loads((root/'twohour-confirmation/incumbent_control.json').read_text())
assert candidate['status']==control['status']=='complete'
assert candidate['parameters_and_buffers_unchanged'] and control['parameters_and_buffers_unchanged']
results=[{k:r[k] for k in ['label','games','score','paired_score_95ci']} for r in candidate['evaluations']]
diffs=[]
for b in control['evaluations']:
 a=next(x for x in candidate['evaluations'] if x['label']==b['label'])
 pa=next(x for x in candidate['plan']['opponents'] if x['label']==a['label'])
 pb=next(x for x in control['plan']['opponents'] if x['label']==b['label'])
 assert pa==pb
 diff=(np.array(a['outcomes'])-np.array(b['outcomes'])).reshape(-1,2).mean(1)
 half=1.96*diff.std(ddof=1)/np.sqrt(len(diff))
 diffs.append({'label':a['label'],'games':a['games'],'candidate_score':a['score'],'incumbent_score':b['score'],'candidate_minus_incumbent':float(diff.mean()),'paired95ci':[float(diff.mean()-half),float(diff.mean()+half)]})
report={'candidate':candidate['plan']['candidate'],'confirmation_results':results,'paired_heldout_differences':diffs,'gpu_evaluation_seconds':candidate['elapsed_seconds']+control['elapsed_seconds'],'candidate_training_metadata':candidate['candidate_metadata'],'scope':'Fresh paired complete-game independent confirmation; previousdevelopmentalseedsandfinalcampaignseedsnotused. Heldoutdifferencespairbothcandidateandincumbentbydeal andseat; no independenceclaimperseat.'}
(root/'twohour-confirmation-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='candidate_training_metadata'},indent=2))
