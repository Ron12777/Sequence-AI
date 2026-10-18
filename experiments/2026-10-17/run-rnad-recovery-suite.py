"""Wait for immutable corrected-hour capture, then one bounded GPU reservation."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/workspace/sequence');os.chdir(root)
out=root/'campaign/exact/rnad-recovery-1h';out.mkdir(exist_ok=True)
suite=(out/'suite.lock').open('a');fcntl.flock(suite,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest_path=root/'campaign/exact/milestones-v2/manifest.json'
started=time.monotonic()
while True:
 manifest=json.loads(manifest_path.read_text())
 if 'cnn_rnad-threshold8-1h' in manifest:break
 if time.monotonic()-started>900:raise TimeoutError('Corrected RNaD hour not captured in15minutes')
 time.sleep(10)
m=manifest['cnn_rnad-threshold8-1h']
assert m['actual_training_seconds']>=9016.89391744716
checkpoints={
 'corrected':{'checkpoint':'campaign/exact/milestones-v2/'+m['file'],'sha256':m['sha256']},
 'prechange':{'checkpoint':'campaign/exact/recipe-amendments/rnad-threshold2-continuation.pt','sha256':'d86d2c7130e415c6511b43fa06fc4d546a4f73f83e1735fa1c863d4d104380f6'}}
for spec in checkpoints.values():assert hashlib.sha256(Path(spec['checkpoint']).read_bytes()).hexdigest()==spec['sha256']
opponents=[
 {'label':'validated_cnn_2h','kind':'exact','checkpoint':'campaign/exact/milestones-v2/cnn_ppo-mixed-2h.pt','sha256':'bfcb0193d48dc0d40449fdbf440df44a7b0400cc33d996e412b1b491b75d6ed8','games':4096,'seed':2900100000001},
 {'label':'previous_cnn_1h_anchor','kind':'exact','checkpoint':'campaign/exact/milestones/cnn_ppo-mixed-1h.pt','sha256':'f031c72501b1f85b5147b78352febcb2c494ee240505fcb89500fc203fb6c1cb','games':2048,'seed':2900200000001}]
names=[]
for candidate,spec in checkpoints.items():
 for mode,action_seed in [('greedy',None),('sample1',410071),('sample2',410072)]:
  name=candidate+'_'+mode;names.append(name)
  opps=[dict(o,games=o['games'] if action_seed is None else 1024) for o in opponents]
  plan={'candidate':spec,'opponents':opps,'device':'cuda','threads':1,'env_threads':1,'batch_size':256,'purpose':'Fixed developmental recovery comparison after>=1h threshold8 continuation; identical fresh paired deals, prechange control, same opponent anchors. Greedy andT1policy reported separately; action replicas clustered within deal, not additional independentdeals. No finalseeds or automaticpromotion.'}
  if action_seed is not None:plan.update(candidate_action_mode='sample',action_seed=action_seed)
  (out/(name+'-plan.json')).write_text(json.dumps(plan,indent=2)+'\n')
(out/'checkpoint-metadata.json').write_text(json.dumps({'corrected':m,'prechange_training_seconds':5416.89391744716,'checkpoints':checkpoints},indent=2)+'\n')
pause=root/'campaign/exact/sustained/gpu-0.pause';pause.touch(exist_ok=False)
lease=None;reserved=time.monotonic()
try:
 lease=(root/'campaign/exact/sustained/locks/gpu-0.lock').open('a')
 while True:
  try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   if time.monotonic()-reserved>120:raise TimeoutError('GPU0 did not yield')
   time.sleep(1)
 print('GPU0 acquired; GPU1 continues',flush=True)
 for name in names:
  with (out/(name+'.log')).open('a') as log:
   subprocess.run([sys.executable,'-u','scripts/evaluate_exact_policy.py','--plan',str(out/(name+'-plan.json')),'--output',str(out/(name+'.json'))],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},check=True,timeout=max(1,600-(time.monotonic()-reserved)))
  r=json.loads((out/(name+'.json')).read_text());print(json.dumps({'name':name,'seconds':r['elapsed_seconds'],'results':[{k:e[k] for k in ['label','games','score','paired_score_95ci']} for e in r['evaluations']]}),flush=True)
 (out/'suite-complete.json').write_text(json.dumps({'status':'complete','gpu_lease_wall_seconds':time.monotonic()-reserved}))
finally:
 if lease is not None:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
 pause.unlink(missing_ok=True)
 print('GPU0 released; training may resume',flush=True)
