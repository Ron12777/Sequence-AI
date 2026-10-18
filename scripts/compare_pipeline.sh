#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
# Run only after the initial experiment suite has finished, on the same GPU.
for ((attempt=0;attempt<360;attempt++)); do
    [[ -f artifacts/experiments.status ]] && break
    sleep 5
done
[[ -f artifacts/experiments.status ]]
python scripts/evaluate.py --checkpoint experiments/original/latest.pt \
  --games 256 --simulations 0,8,32 --output artifacts/eval_original.jsonl
for mode in sequential overlap; do
    extra=()
    [[ "$mode" == overlap ]] && extra+=(--overlap-training)
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1 > "artifacts/gpu_$mode.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    timeout --signal=INT --kill-after=30s 600s python -m src.train train \
      --fresh --no-compile --epochs 6 --games-per-epoch 512 --simulations 16 \
      --workers 8 --games-per-worker 128 --num-filters 128 --num-res-blocks 6 \
      --batch-size 2048 --train-steps 96 --buffer-size 100000 --lr .001 --seed 73 \
      --output-dir "experiments/pipeline_$mode" "${extra[@]}" \
      > "artifacts/pipeline_$mode.log" 2>&1
    kill "$monitor_pid" 2>/dev/null || true
    trap - EXIT
done
for name in small base; do
    python scripts/evaluate.py --checkpoint "experiments/$name/latest.pt" \
      --games 512 --seed 130363 --simulations 0,8,32 \
      --output "artifacts/confirm_$name.jsonl"
done
printf 'completed\n' > artifacts/pipeline.status
