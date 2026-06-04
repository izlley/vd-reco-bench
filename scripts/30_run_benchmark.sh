#!/usr/bin/env bash
# 평가 + latency profiling.
# 사용법: bash scripts/30_run_benchmark.sh <experiment_yaml>
#
# 설계: reports/03_baseline_methodology.md §4.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_yaml>" >&2
    exit 2
fi

CFG="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

LOG="logs/eval_$(basename "${CFG}" .yaml)_$(date +%Y%m%dT%H%M%S).log"
mkdir -p logs

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -u -m reco_bench.pipelines.evaluate "${CFG}" 2>&1 | tee "${LOG}"
echo "[scripts/30] log → ${LOG}"
