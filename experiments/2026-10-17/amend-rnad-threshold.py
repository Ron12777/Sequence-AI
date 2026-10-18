"""One explicit, audited R-NaD continuation change; no optimizer reset."""
import copy,fcntl,hashlib,json,os,sys,tempfile,time
from datetime import datetime,timezone
from pathlib import Path
import torch
root=Path('/workspace/sequence');os.chdir(root);sys.path.insert(0,str(root));torch.set_num_threads(1)
from src.exact_policy import load_exact_policy
from scripts.train_exact_policy import restore_rnad
output=root/'campaign/exact/recipe-amendments';output.mkdir(exist_ok=True)
lock=(output/'mutation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
record=output/'rnad-threshold8.json'
assert not record.exists(),'Amendment already recorded; do not repeat'
manifest=json.loads((root/'campaign/exact/milestones/manifest.json').read_text());control=manifest['cnn_rnad-4090-1h']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert digest(root/'campaign/exact/milestones'/control['file'])==control['sha256']
assert (root/'campaign/exact/rnad-frozen-diagnostic.json').exists()
assert control['actual_training_seconds'] >= 4982.350604493171
base=root/'campaign/exact/sustained';pauses=[base/f'gpu-{i}.pause' for i in [0,1]]
assert not any(p.exists() for p in pauses),'Another GPU reservation is active'
leases=[];created=[];started=time.monotonic()
def equal(a,b,path=''):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:equal(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(b) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
 else:assert a==b,path
try:
 for p in pauses:p.touch(exist_ok=False);created.append(p)
 for i in [0,1]:
  handle=(base/'locks'/f'gpu-{i}.lock').open('a')
  while True:
   try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:
    if time.monotonic()-started>120:raise TimeoutError('Workers did not yield')
    time.sleep(1)
  leases.append(handle)
 contender=(base/'cnn_rnad/worker.lock').open('a');fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(contender)
 path=base/'cnn_rnad/latest.pt';before=torch.load(path,map_location='cpu',weights_only=True)
 assert before['metadata']['algorithm']=='rnad'
 assert before['training_state']['rnad']['config']['nerd_threshold']==2.0
 backup=output/'rnad-threshold2-continuation.pt';assert not backup.exists();os.link(path,backup)
 payload=copy.deepcopy(before)
 change={'utc':datetime.now(timezone.utc).isoformat(),'parameter':'training_state.rnad.config.nerd_threshold','before':2.0,'after':8.0,'training_seconds_before':before['metadata']['training_seconds'],'learner_steps_before':before['training_state']['rnad']['learner_steps'],'reason':'Frozen64-game diagnostic: beta2 blocks46.1% of positive top-action force; beta8 nearly eliminates this gating. Warm top-centeredlogitmean10.31 collapsedto2.24; learned T1strengthregressed. Hypothesis-driven continuation correction, not a proven strength gain.','diagnostic_sha256':digest(root/'campaign/exact/rnad-frozen-diagnostic.json'),'preserved_control_sha256':control['sha256']}
 payload['training_state']['rnad']['config']['nerd_threshold']=8.0
 payload['metadata']['recipe_amendments']=[*payload['metadata'].get('recipe_amendments',[]),change]
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='threshold8-',suffix='.pt');os.close(fd)
 try:
  torch.save(payload,tmp);after=torch.load(tmp,map_location='cpu',weights_only=True)
  normalized=copy.deepcopy(after);normalized['training_state']['rnad']['config']['nerd_threshold']=2.0
  if 'recipe_amendments' in before['metadata']:normalized['metadata']['recipe_amendments']=before['metadata']['recipe_amendments']
  else:normalized['metadata'].pop('recipe_amendments')
  equal(before,normalized)
  model,_=load_exact_policy(tmp,'cpu');state=restore_rnad(model,tmp,after['metadata']['recipe']);assert state.config.nerd_threshold==8.0
  assert state.learner_steps==before['training_state']['rnad']['learner_steps']
  equal(state.target.state_dict(),before['training_state']['rnad']['target'])
  equal(state.reference.state_dict(),before['training_state']['rnad']['reference'])
  equal(state.previous_reference.state_dict(),before['training_state']['rnad']['previous_reference'])
  os.replace(tmp,path)
 finally:Path(tmp).unlink(missing_ok=True)
 details=change|{'before_sha256':digest(backup),'after_sha256':digest(path),'backup':str(backup),'verification':'All payload values/tensors bitwise equal except explicit threshold and metadata amendment. Restored runtime config8 and identical learnersteps/EMA/references. Model,Adam,RNG,schedulertrainingseconds and rootrecipe unchanged.','global_recipe_note':'nerd_threshold was not present in rootrecipe; serialized RNaD config is authoritative onrestore. Explicit amendment persists in checkpointmetadata.'}
 pending=record.with_suffix('.tmp');pending.write_text(json.dumps(details,indent=2)+'\n');os.replace(pending,record);print(json.dumps(details),flush=True)
finally:
 for handle in reversed(leases):fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
 for p in created:p.unlink(missing_ok=True)
 print('All leases and pauses released; supervisor can resume',flush=True)
