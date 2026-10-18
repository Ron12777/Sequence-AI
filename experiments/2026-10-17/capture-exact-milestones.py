"""CPU-only immutable checkpoint capture; never controls training or uses CUDA."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--plan', required=True)
parser.add_argument('--once', action='store_true')
args = parser.parse_args()
plan = json.loads(Path(args.plan).read_text())
root = Path(plan['root'])
output = Path(plan['output'])
output.mkdir(parents=True, exist_ok=True)
lock = (output / 'capture.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
seen = {}
manifest_path = output / 'manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
for key, item in manifest.items():
    assert hashlib.sha256((output / item['file']).read_bytes()).hexdigest() == item['sha256'], key
while time.time() < plan['deadline_epoch']:
    for contender, thresholds in plan['thresholds'].items():
        source = root / contender / 'latest.pt'
        stat = source.stat()
        identity = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if seen.get(contender) == identity:
            continue
        seen[contender] = identity
        temporary = output / (contender + '.inspect.pt')
        temporary.unlink(missing_ok=True)
        os.link(source, temporary)  # Trainer saves by atomic replacement; this inode is immutable.
        try:
            metadata = torch.load(temporary, map_location='cpu', weights_only=True)['metadata']
            for label, threshold in thresholds.items():
                key = contender + '-' + label
                if key in manifest or metadata['training_seconds'] < threshold:
                    continue
                destination = output / (key + '.pt')
                if not destination.exists():
                    os.link(temporary, destination)
                captured = torch.load(destination, map_location='cpu', weights_only=True)['metadata']
                manifest[key] = {'file': destination.name, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'threshold_seconds': threshold, 'actual_training_seconds': captured['training_seconds'], 'overshoot_seconds': captured['training_seconds']-threshold, 'metadata': captured}
                pending = output / 'manifest.tmp'
                pending.write_text(json.dumps(manifest, indent=2)+'\n')
                os.replace(pending, manifest_path)
                print(json.dumps({'captured':key,'training_seconds':metadata['training_seconds'],'overshoot_seconds':metadata['training_seconds']-threshold}),flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    if len(manifest) == sum(len(x) for x in plan['thresholds'].values()):
        print('All requested milestones captured',flush=True)
        break
    if args.once:
        break
    time.sleep(10)
