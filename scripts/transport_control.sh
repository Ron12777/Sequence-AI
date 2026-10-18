#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python scripts/benchmark.py --seconds 30 --configs 16:128:64:3 --output artifacts/control_float_before.jsonl
python scripts/benchmark.py --seconds 30 --configs 16:128:64:3 --compact-observations --output artifacts/control_compact.jsonl
python scripts/benchmark.py --seconds 30 --configs 16:128:64:3 --output artifacts/control_float_after.jsonl
printf 'completed\n' > artifacts/control.status
