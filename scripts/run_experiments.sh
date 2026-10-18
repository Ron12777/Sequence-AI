#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p artifacts experiments
# Bounded experiments, sequential on one GPU. Keep only latest.pt per variant.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 2 > artifacts/training_gpu.csv &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
python scripts/evaluate.py --games 128 --simulations 0,8,32 --output artifacts/untrained_eval.jsonl
for spec in 'small 64 3' 'base 128 6'; do
    read -r name filters blocks <<< "$spec"
    timeout --signal=INT --kill-after=30s 1200s python -m src.train train \
        --fresh --no-compile --epochs 24 --games-per-epoch 512 \
        --simulations 16 --workers 8 --games-per-worker 128 \
        --num-filters "$filters" --num-res-blocks "$blocks" \
        --batch-size 2048 --train-steps 96 --buffer-size 100000 \
        --lr .001 --seed 42 --output-dir "experiments/$name" \
        > "artifacts/train_$name.log" 2>&1
    python scripts/evaluate.py --checkpoint "experiments/$name/latest.pt" \
        --games 128 --simulations 0,8,32 --output "artifacts/eval_$name.jsonl"
done
printf 'completed\n' > artifacts/experiments.status
