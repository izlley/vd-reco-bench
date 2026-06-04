#!/usr/bin/env bash
# 데이터셋 다운로드 + 전처리.
# 사용법: bash scripts/00_download_data.sh <dataset_name>
#   <dataset_name> ∈ {ml25m, amazon_beauty, amazon_books, amazon_electronics}
#
# 설계: reports/02_dataset_selection.md §3.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_name>" >&2
    echo "Available: ml25m" >&2
    exit 2
fi

DS="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${ROOT}/configs/datasets/${DS}.yaml"

if [ ! -f "${CFG}" ]; then
    echo "Config not found: ${CFG}" >&2
    exit 2
fi

cd "${ROOT}"
echo "[scripts/00] $(date -Iseconds) starting preprocessing for ${DS}"
echo "[scripts/00] config: ${CFG}"
python -m reco_bench.data.pipeline "${CFG}" --root "${ROOT}"
echo "[scripts/00] $(date -Iseconds) done"
