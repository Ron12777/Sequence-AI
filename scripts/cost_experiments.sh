#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p artifacts experiments
python scripts/benchmark.py --seconds 30 --configs 8:128:64:3,16:128:64:3 --output artifacts/cost_benchmark.jsonl
workers=$(python -c 'import json; rows=[json.loads(l) for l in open("artifacts/cost_benchmark.jsonl")]; print(max(rows,key=lambda r:r["leaf_evaluations_per_second"])["workers"])')
for sims in 8 16; do
    name="small_sims$sims"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1 > "artifacts/gpu_$name.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    timeout --signal=INT --kill-after=30s 1200s python -m src.train train \
      --fresh --no-compile --epochs 24 --games-per-epoch 512 --simulations "$sims" \
      --workers "$workers" --games-per-worker 128 --num-filters 64 --num-res-blocks 3 \
      --batch-size 2048 --train-steps 96 --buffer-size 100000 --lr .001 --seed 37 \
      --output-dir "experiments/$name" > "artifacts/train_$name.log" 2>&1
    kill "$monitor_pid" 2>/dev/null || true
    trap - EXIT
    python scripts/evaluate.py --checkpoint "experiments/$name/latest.pt" \
      --games 512 --seed 15485863 --simulations 0,8,32 --output "artifacts/eval_$name.jsonl"
done
python scripts/evaluate.py --checkpoint models/latest.pt \
  --games 512 --seed 15485863 --simulations 0,8,32 --output artifacts/eval_previous_best.jsonl
printf 'completed\n' > artifacts/cost.status
