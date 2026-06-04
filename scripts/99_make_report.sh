#!/usr/bin/env bash
# 평가 결과를 reports/baseline_results.md 와 reports/figures/*.png 로 자동 집계.
# 사용법: bash scripts/99_make_report.sh <experiment_yaml>
#
# 설계: reports/03_baseline_methodology.md §6.

set -euo pipefail

if [ $# -lt 1 ]; then
    CFG="configs/experiments/phase1_baseline.yaml"
    echo "[scripts/99] no config given, defaulting to ${CFG}"
else
    CFG="$1"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

python -u -m reco_bench.pipelines.report "${CFG}"
echo "[scripts/99] done. open reports/baseline_results.md to view."
