#!/usr/bin/env bash
# Two-tower 학습 entry.
# 사용법: bash scripts/10_train_two_tower.sh <experiment_yaml> [--output-dir DIR]
#
# 설계: reports/03_baseline_methodology.md §1, §2.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_yaml> [--output-dir DIR]" >&2
    exit 2
fi

CFG="$1"
shift

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

LOG="logs/train_$(basename "${CFG}" .yaml)_$(date +%Y%m%dT%H%M%S).log"
mkdir -p logs

# 측정 일관성을 위해 GPU 0 만 사용 기본 (다중 GPU 분산은 별도 wrapper).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -u -m reco_bench.pipelines.train "${CFG}" "$@" 2>&1 | tee "${LOG}"
echo "[scripts/10] log → ${LOG}"
