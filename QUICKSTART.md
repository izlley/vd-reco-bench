# Quick Start — 5분 안에 첫 결과 보기

이 문서는 `reco_bench` 를 처음 접하는 사람이 **단일 명령으로 첫 결과**
까지 도달하도록 돕습니다. 자세한 설계는 [README.md](README.md) 와
[reports/](reports/) 참조.

## 0. 사전 요구

| 항목 | 권장 |
|---|---|
| OS | Linux (Ubuntu 22.04+ 권장) |
| GPU | NVIDIA H100 / A100 / H200 (CUDA 12.x), 1 대 이상 |
| Python | 3.10 ~ 3.12 |
| 디스크 | 30 GB+ 여유 (ML-25M ~ 2 GB, Amazon Beauty ~ 15 GB) |
| 메모리 | 64 GB+ 권장 |

GPU 없이도 CPU baseline (FAISS-CPU HNSW, ScaNN, Milvus Lite, Qdrant
Local) 만 측정 가능.

## 1. 설치 (1 분)

```bash
git clone <repo-url> reco_bench && cd reco_bench

# Python 환경 (이미 PyTorch + CUDA 12.x 가 깔린 환경 가정)
pip install -e .
pip install -r requirements.txt

# Vector DB / ANN 라이브러리
pip install faiss-cpu sentence-transformers scikit-learn nvidia-ml-py \
            "pymilvus[milvus_lite]" qdrant-client scann

# GPU 가속 (cuVS) — NVIDIA pip 채널
pip install --extra-index-url https://pypi.nvidia.com \
    cuvs-cu12 pylibraft-cu12 cupy-cuda12x
```

설치 검증:
```bash
python -c "import torch, faiss, cuvs, pymilvus, qdrant_client, scann; \
print('torch', torch.__version__, '|', torch.cuda.device_count(), 'GPUs')"
```

## 2. 한 번에 끝까지 (5 ~ 30 분 — 데이터셋 크기에 따라)

```bash
# 데이터 → 학습 → 인덱스 → 평가 → 리포트 까지 단일 흐름
bash scripts/00_download_data.sh ml25m
bash scripts/10_train_two_tower.sh configs/experiments/phase1_full.yaml
bash scripts/20_build_index.sh    configs/experiments/phase1_full.yaml
bash scripts/30_run_benchmark.sh  configs/experiments/phase1_full.yaml
bash scripts/99_make_report.sh    configs/experiments/phase1_full.yaml
```

산출물:
- `reports/baseline_results.md` — 자동 생성된 main result.
- `reports/figures/*.png` — 5종 그래프 (Recall-QPS, latency CDF, cost
  bar, throughput vs concurrency, throughput per Watt).
- `results/phase1_full/{metrics.json, aggregate.csv}` — raw 측정값.
- `checkpoints/phase1_full/<dataset>/` — 학습된 모델 + item embedding.
- `indexes/<retriever>/<dataset>/` — retriever 별 ANN 인덱스.

## 3. 결과 보기

```bash
# Markdown 리포트
less reports/baseline_results.md
# 또는 GitHub 에서 그대로 렌더링되는 형식

# 시각화
open reports/figures/recall_vs_qps.png
```

`reports/baseline_results.md` 의 메인 표 컬럼:

| 컬럼 | 의미 |
|---|---|
| **Recall@10 (vs exact)** | ANN 알고리즘의 정확도 (모델과 독립). VDPU 비교의 핵심 지표 |
| **Recall@10 (vs GT)** | 추천 품질 (model + ANN 합) |
| **QPS (max c)** | concurrency 64 에서의 최대 throughput |
| **P99 ms** | single-stream P99 지연 |
| **$/1M** | 1M query 처리 비용 (cloud SKU 기반) |
| **W/QPS** | 전력 효율 |

## 4. 다른 dataset 으로

```bash
# Amazon Reviews 2023 - Beauty/Books/Electronics
bash scripts/00_download_data.sh amazon_beauty
bash scripts/00_download_data.sh amazon_books        # 더 큼 (~30G raw)
bash scripts/00_download_data.sh amazon_electronics  # 가장 큼

# experiment YAML 에 dataset 추가 후 동일 명령으로 재실행
```

## 5. 새 retriever 추가하기 (5 분)

`reco_bench` 의 핵심 설계 원칙은 **모든 retriever 가 한 추상 클래스
뒤에 숨는다** 는 것. Vector DB 든 ANN 라이브러리 든 추가 절차는 동일:

1. `reco_bench/retrievers/<my_retriever>.py` 작성 — `Retriever` 상속,
   `build / search / save / load / device_info` 5개 메서드만 구현.
2. `configs/retrievers/<my_retriever>.yaml` 작성 (기존 yaml 참고).
3. 사용하려는 experiment yaml 의 `retrievers:` 리스트에 추가.

`reco_bench/retrievers/base.py` 의 docstring 이 contract 의 상세.

## 6. VDPU 통합 (Phase 2; 본 phase 의 산출물 아님)

VDPU 가 도착하면 새 retriever 한 개 + YAML 한 개 + cost_model.yaml
한 행만 추가하면 전체 비교에 자동으로 포함된다 (`configs/retrievers/vdpu.yaml`
은 미래 작업으로 비워 둠).

## 7. 자주 묻는 문제

### `pymilvus.exceptions.ConnectionConfigException: milvus-lite is required`

```bash
pip install "pymilvus[milvus_lite]"
```

### `faiss-gpu` cublas symbol error

본 벤치마크는 PyTorch + cublas 12.8 충돌을 피하기 위해 **`faiss-gpu`
를 사용하지 않습니다**. GPU 측은 cuVS 의 IVF-PQ / CAGRA 로 대체.
`pip uninstall faiss-gpu-cu12; pip install --force-reinstall --no-deps faiss-cpu`
로 복구.

### MovieLens 25M sha256 mismatch

GroupLens mirror 의 zip 이 시간에 따라 재패키징되어 sha 가 변할 수
있다. 본 벤치마크는 mismatch 시 경고만 출력하고 진행. raw csv 의
row count 가 동일하면 OK.

### Amazon Reviews 2023 의 datasets 4.x 호환 에러

`datasets` 4.x 가 dataset loading script 지원을 끊었음. 본 벤치마크는
`huggingface_hub.hf_hub_download` 로 raw jsonl 을 직접 받아 처리한다
— 자동으로 동작.

### Recall@10 (vs GT) 가 매우 낮음

이는 ML-25M 의 단순 ID-only Two-tower 의 known 한계 (~0.02 ~ 0.04).
ANN 비교는 **Recall@10 vs exact** 컬럼으로 보세요 (모델 품질과
독립적이며 ANN 알고리즘의 정확도만 측정).

자세한 내용: [`reports/01_metric_design.md §2.4`](reports/01_metric_design.md),
[`reports/history/`](reports/history/).
