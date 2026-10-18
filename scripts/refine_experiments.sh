#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p artifacts experiments
python scripts/benchmark.py --compact-observations --seconds 30 --configs 8:128:64:3,16:128:64:3,16:256:64:3 --output artifacts/packed_benchmark.jsonl
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1 > artifacts/gpu_exploration.csv &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
timeout --signal=INT --kill-after=30s 1200s python -m src.train train \
  --fresh --no-compile --epochs 24 --games-per-epoch 512 --simulations 16 \
  --workers 16 --games-per-worker 128 --num-filters 64 --num-res-blocks 3 \
  --batch-size 2048 --train-steps 96 --buffer-size 100000 --lr .001 --seed 37 \
  --compact-observations --root-noise-fraction .75 --output-dir experiments/small_exploration \
  > artifacts/train_exploration.log 2>&1
kill "$monitor_pid" 2>/dev/null || true
trap - EXIT
python scripts/evaluate.py --checkpoint experiments/small_exploration/latest.pt \
  --games 512 --seed 15485863 --simulations 0,8,32 --output artifacts/eval_exploration.jsonl
printf 'completed\n' > artifacts/refine.status
