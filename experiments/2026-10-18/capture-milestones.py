"""CPU-only immutable overnight snapshots, at +2/+5/+8 training hours."""
import fcntl,hashlib,json,os,time
from datetime import datetime,timezone
from pathlib import Path
import torch
root=Path('/workspace/sequence/campaign/overnight'); out=root/'milestones';out.mkdir(exist_ok=True)
lock=(out/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
baselines=json.loads((root/'starting-metadata.json').read_text())
manifest_path=out/'manifest.json'; manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
deadline=datetime.fromisoformat('2026-10-18T14:02:00+00:00').timestamp()
torch.set_num_threads(1)
while time.time()<deadline:
 for name,b in baselines.items():
  path=root/name/'latest.pt'
  # Atomic replacement by trainer makes each open read a complete checkpoint.
  data=path.read_bytes()
  import io
  p=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
  for hours in [2,5,8]:
   key=f'{name}-additional-{hours}h'
   if key in manifest or p['metadata']['training_seconds']-b['training_seconds']<hours*3600:continue
   dest=out/(key+'.pt');dest.write_bytes(data)
   manifest[key]={'path':str(dest),'sha256':hashlib.sha256(data).hexdigest(),'training_seconds':p['metadata']['training_seconds'],'additional_training_seconds':p['metadata']['training_seconds']-b['training_seconds'],'games':p['metadata']['games'],'captured_utc':datetime.now(timezone.utc).isoformat()}
   temp=manifest_path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2)+'\n');os.replace(temp,manifest_path)
   print(json.dumps(manifest[key]),flush=True)
 if len(manifest)==6:break
 time.sleep(120)
