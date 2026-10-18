"""Explicit two-GPU CNN continuation to one-GPU execution migration; no recipe/optimizer reset."""
import hashlib,json,torch
from pathlib import Path
root=Path('/workspace/sequence')
source=root/'artifacts/checkpoints/exact-final/cnn_ppo-final.pt'
payload=torch.load(source,map_location='cpu',weights_only=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()=='59d1ed9cc5364e9a67e95fd150286cfdb22f3e0454b987a96858c349ca63dd45'
assert len(payload['training_state']['cuda_rng'])==2
old_states=payload['training_state']['cuda_rng']
payload['training_state']['cuda_rng']=[old_states[0]]
# Preserve the inactive secondary generator for provenance and any later dual-GPU resume.
payload['training_state']['archived_secondary_cuda_rng']=old_states[1:]
dest=root/'campaign/overnight/cnn_ppo/latest.pt'
torch.save(payload,dest)
reloaded=torch.load(dest,map_location='cpu',weights_only=True)
original=torch.load(source,map_location='cpu',weights_only=True)
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
verified={k:equal(original[k],reloaded[k]) for k in original if k!='training_state'}
verified.update({k:equal(v,reloaded['training_state'][k]) for k,v in original['training_state'].items() if k!='cuda_rng'})
assert all(verified.values())
assert torch.equal(old_states[0],reloaded['training_state']['cuda_rng'][0])
assert torch.equal(old_states[1],reloaded['training_state']['archived_secondary_cuda_rng'][0])
selection={'contenders':['cnn_ppo','transformer_ppo'],'reason':'User authorized full overnight run for both contenders until October18 10am Toronto. One dedicated GPU per contender; no replicas or RNaD training.'}
(root/'campaign/overnight/selection.json').write_text(json.dumps(selection,indent=2)+'\n')
audit={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'continuation_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'verified_unchanged':verified,'change':'Active CUDA RNG list uses original GPU0 state; original GPU1 RNG retained in archived_secondary_cuda_rng and immutable source checkpoint. Hardware/execution transition is not bitwise trajectory equivalence.','source_cuda_rng_states':2,'active_cuda_rng_states':1}
(root/'campaign/overnight/independent-worker-transition.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
