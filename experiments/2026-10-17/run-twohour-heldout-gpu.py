"""One exclusive GPU reservation; other GPU keeps training; safe lease release."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/workspace/sequence');os.chdir(root)
out=root/'campaign/exact/twohour-heldout-replication';out.mkdir(exist_ok=True)
suite=(out/'suite.lock').open('a');fcntl.flock(suite,fcntl.LOCK_EX|fcntl.LOCK_NB)
for name in ['cnn_ppo','incumbent_control']:
 assert (out/(name+'-plan.json')).is_file()
pause=root/'campaign/exact/sustained/gpu-0.pause'
pause.touch(exist_ok=False)
lease=None;started=time.monotonic()
try:
 lease=(root/'campaign/exact/sustained/locks/gpu-0.lock').open('a')
 while True:
  try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   if time.monotonic()-started>120:raise TimeoutError('GPU0 training did not yield')
   time.sleep(1)
 print('GPU0 acquired; GPU1 continues',flush=True)
 for name in ['cnn_ppo','incumbent_control']:
  with (out/(name+'.log')).open('a') as log:
   subprocess.run([sys.executable,'-u','scripts/evaluate_exact_policy.py','--plan',str(out/(name+'-plan.json')),'--output',str(out/(name+'.json'))],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},check=True,timeout=max(1,600-(time.monotonic()-started)))
  r=json.loads((out/(name+'.json')).read_text())
  print(json.dumps({'candidate':name,'seconds':r['elapsed_seconds'],'results':[{k:e[k] for k in ['label','games','score','paired_score_95ci']} for e in r['evaluations']]}),flush=True)
finally:
 if lease is not None:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
 pause.unlink(missing_ok=True)
 print('GPU0 released; training may resume',flush=True)
