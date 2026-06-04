#!/usr/bin/env bash
# ANN 인덱스 빌드.
# 사용법: bash scripts/20_build_index.sh <experiment_yaml>
#
# 설계: reports/03_baseline_methodology.md §3.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_yaml>" >&2
    exit 2
fi

CFG="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

LOG="logs/build_$(basename "${CFG}" .yaml)_$(date +%Y%m%dT%H%M%S).log"
mkdir -p logs

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -u -m reco_bench.pipelines.build_index "${CFG}" 2>&1 | tee "${LOG}"
echo "[scripts/20] log → ${LOG}"
