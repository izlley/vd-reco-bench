# reco_bench

**Vector-DB 가속기를 위한 Two-tower 추천 검색 benchmark.**

`reco_bench` 는 Two-tower 추천 시스템의 ANN (approximate nearest neighbor)
검색 단계의 비용 효율을 측정한다. ANN 단계는 YouTube, Pinterest, 쿠팡 같은
e-commerce 사이트의 실제 production inference 에서 비용을 지배하는 단계이다.

기존 추천 벤치마크 ([MLPerf DLRM](https://github.com/mlcommons/inference/tree/master/recommendation),
[RecBole](https://github.com/RUCAIBox/RecBole),
[Microsoft recommenders](https://github.com/recommenders-team/recommenders)) 는 주로
**모델 품질**에 집중한다. 기존 ANN 벤치마크
([ann-benchmarks](https://github.com/erikbern/ann-benchmarks),
[VectorDBBench](https://github.com/zilliztech/VectorDBBench)) 는 주로
합성/텍스트 벡터에서의 **시스템 속도**에 집중한다.
`reco_bench` 는 그 중간이다. 실제 Two-tower 모델을 학습시킨 뒤 그
**학습된 item embedding 위에서** ANN 검색 단계를 측정한다. 따라서 측정
비용은 합성 워크로드가 아니라 **추천 워크로드** 를 그대로 반영한다.
어떤 벤치마크에서 무엇을 빌려왔는지는 아래 [참고한 오픈 벤치마크](#참고한-오픈-벤치마크--references) 참조.

본 벤치마크는 등장 중인 가속기들 — CPU, GPU, 그리고 디노티시아의
**VDPU (Vector Data Processing Unit)** 같은 vector-DB 전용 프로세서 — 가
**iso-recall (동일 recall)** 기준으로 하드웨어 종속성 없이 같은 측정 도구로
비교될 수 있도록 설계되었다.

## 진행 상태

| Phase | 범위 | 상태 |
|---|---|---|
| 0 | 방법론 문서 (`reports/`) | ✅ 완료 |
| 1 | Two-tower 학습 + ANN 라이브러리 5종 (FAISS-CPU/GPU, cuVS×2, ScaNN) + Vector DB (Qdrant server) baseline + 5종 자동 시각화 + speedup table | ✅ 완료 (ML-25M + Amazon Beauty) |
| 2 | corpus/dim scaling + sensitivity (`reports/07_scaling.md`) + CPU thread 정책 | ✅ 부분 완료 (VDPU-비의존) |
| 2 | VDPU 통합 (`reco_bench/retrievers/vdpu.py`) | ⏳ 대기 (실 하드웨어 도착 후) |

**핵심 발견** (`reports/06_phase1_findings.md`, `07_scaling.md`):
- 작은 corpus(ML-25M 24k)에선 **FAISS-CPU HNSW 가 cuVS CAGRA 대비 빠름** — GPU dispatch 오버헤드 때문.
- **corpus 가 커질수록 GPU 우위 확대**: synthetic 100k→10M 에서 cuVS CAGRA QPS 유지(14.7k→16.4k), FAISS-CPU 하락(3.6k→3.0k) → 격차 4×→5.5×.
- **고차원일수록 CPU 불리**: dim 128→256 에서 CPU recall·QPS 동시 하락, GPU 는 QPS 유지.
- Vector DB(Qdrant server, gRPC)는 per-query RPC 로 in-process 라이브러리보다 186~447× 느림 (batch API 미사용 lower bound).
- CPU 측정은 cost SKU(128-core)에 맞춰 **thread 128 고정** (측정-비용 정합).
- **이 모든 비교의 frame 위에 VDPU 한 행을 추가하는 것이 Phase 2 의 최종 목표.**

## 측정 항목

- **품질** — Recall@{10,100}, NDCG@{10,100}, HitRate, MRR
- **시스템** — QPS, P50/P95/P99 latency, index build time, host/device peak
  memory, 에너지 (`nvidia-smi power.draw` 적분), throughput-per-Watt
- **Trade-off** — Recall-QPS Pareto 곡선
  ([ann-benchmarks](https://github.com/erikbern/ann-benchmarks) 컨벤션)
- **Cost** — `$/QPS`, `$/1M queries`, `$/Recall-point`
  (`configs/cost_model.yaml` 의 고정된 cloud SKU 스냅샷 기반)

## 데이터셋 (Phase 0+1)

| 데이터셋 | Item 수 | 용도 |
|---|---|---|
| [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) | 약 62 k | Sanity check / 방법론 정착 |
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — Beauty, Books, Electronics | 카테고리당 약 100 k – 1 M | 중간 규모 e-commerce |

두 데이터셋 모두 연구용 라이센스이며, 라이센스/인용 세부 사항은
`reports/02_dataset_selection.md` 참조. 원본 데이터는 본 저장소에 포함하지
않는다.

## 디렉토리 구조

```
reco_bench/
├── reports/        # 외부 reader 가 가장 먼저 볼 deliverable
│   ├── 00_overview.md … 05_reproducibility.md
│   ├── baseline_results.md   # scripts/99_make_report.sh 가 자동 생성
│   ├── planning/   # 설계 제안서 (승인된 plan, RFC 등 중간 산출물)
│   └── history/    # 시간순 개발 history
├── configs/        # dataset / model / retriever / experiment YAML
├── reco_bench/     # Python 패키지
│   ├── data/       # 데이터셋 로더 + splitter
│   ├── models/     # Two-tower 모델 + loss + sampler
│   ├── retrievers/ # base.py + faiss_gpu.py + cuvs.py + scann.py
│   ├── eval/       # metrics, latency profiler, cost model
│   └── pipelines/  # train, build_index, evaluate, report
├── scripts/        # 00_download → 99_make_report shell entrypoint
├── data/           # 원본 + 전처리 데이터셋 (gitignored)
├── checkpoints/    # 학습된 tower + item embedding (gitignored)
├── indexes/        # retriever × dataset 별 직렬화된 ANN index (gitignored)
├── results/        # 실행별 raw JSON / CSV (gitignored)
├── tests/          # pytest: 인터페이스 계약 + 메트릭 sanity
├── logs/           # 실행 로그 (gitignored)
└── papers/         # 참고 논문
```

## 빠른 시작 (Quick Start)

처음 접하는 사람이 **단일 흐름으로 첫 결과**까지 도달하도록 정리했다.
자세한 설계는 [reports/](reports/) 참조.

### 0. 사전 요구

| 항목 | 권장 |
|---|---|
| OS | Linux (Ubuntu 22.04+ 권장) |
| GPU | NVIDIA H100 / A100 / H200 (CUDA 12.x), 1대 이상 |
| Python | 3.10 ~ 3.12 |
| 디스크 | 30 GB+ 여유 (ML-25M ~2 GB, Amazon Beauty ~15 GB) |
| 메모리 | 64 GB+ 권장 |

GPU 없이도 CPU baseline (FAISS-CPU HNSW, ScaNN, Qdrant) 만 측정 가능.

### 1. 설치 (1분)

```bash
git clone <repo-url> reco_bench && cd reco_bench

# Python 환경 (PyTorch + CUDA 12.x 가 깔린 환경 가정)
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
python -c "import torch, faiss, cuvs; \
print('torch', torch.__version__, '|', torch.cuda.device_count(), 'GPUs')"
```

> **FAISS-GPU (선택)**: H100(sm_90) 은 PyPI wheel 미지원 + PyTorch cublas
> ABI 충돌이라, 필요 시 conda-forge `faiss-gpu` 를 격리 env 에 설치해
> worker 로 측정한다 (`reports/05_reproducibility.md §2`).

### 2. 한 번에 끝까지 (5 ~ 30분, 데이터셋 크기에 따라)

```bash
# 데이터 → 학습 → 인덱스 → 평가 → 리포트 까지 단일 흐름
bash scripts/00_download_data.sh ml25m
bash scripts/10_train_two_tower.sh configs/experiments/phase1_full.yaml
bash scripts/20_build_index.sh    configs/experiments/phase1_full.yaml
bash scripts/30_run_benchmark.sh  configs/experiments/phase1_full.yaml
bash scripts/99_make_report.sh    configs/experiments/phase1_full.yaml
```

산출물:
- `reports/baseline_results.md` — 자동 생성된 main result
- `reports/figures/*.png` — 5종 그래프 (Recall-QPS, latency CDF, cost
  bar, throughput vs concurrency, throughput per Watt)
- `results/phase1_full/{metrics.json, aggregate.csv}` — raw 측정값
- `checkpoints/phase1_full/<dataset>/` — 학습된 모델 + item embedding
- `indexes/<retriever>/<dataset>/` — retriever 별 ANN 인덱스

### 3. 결과 보기

```bash
less reports/baseline_results.md          # Markdown 리포트 (GitHub 렌더링)
open reports/figures/recall_vs_qps.png    # 시각화
```

`baseline_results.md` 메인 표 컬럼:

| 컬럼 | 의미 |
|---|---|
| **Recall@10 (vs exact)** | ANN 알고리즘의 정확도 (모델과 독립). VDPU 비교의 핵심 지표 |
| **Recall@10 (vs GT)** | 추천 품질 (model + ANN 합) |
| **QPS (max c)** | 최대 concurrency 에서의 throughput |
| **P99 ms** | single-stream P99 지연 |
| **$/1M** | 1M query 처리 비용 (cloud SKU 기반) |
| **W/QPS** | 전력 효율 |

### 4. 비교 대상 (Phase 1 측정 완료, 한 명령 안에서 측정)

| 카테고리 | Retriever | Device | 측정 환경 |
|---|---|---|---|
| ANN 라이브러리 | FAISS-CPU HNSW | CPU | main |
| ANN 라이브러리 | FAISS-GPU IVF-PQ | GPU (H100) | 격리 conda env (sm_90) |
| ANN 라이브러리 | cuVS IVF-PQ | GPU (H100) | main |
| ANN 라이브러리 | cuVS CAGRA | GPU (H100) | main |
| ANN 라이브러리 | Google ScaNN | CPU (AVX2) | main |
| Vector DB (서버) | Qdrant server (gRPC) | CPU | standalone 바이너리 |
| Vector DB (in-process) | Milvus Lite | CPU | 부분 측정 (lower bound) |
| 가속기 (Phase 2) | Dnotitia VDPU | — | 예정 |

> GPU 측정은 모두 **H100 1장 (`CUDA_VISIBLE_DEVICES=0`)** 기준.

### 5. 다른 dataset 으로

```bash
bash scripts/00_download_data.sh amazon_beauty
bash scripts/00_download_data.sh amazon_books        # 더 큼 (~30G raw)
bash scripts/00_download_data.sh amazon_electronics  # 가장 큼
# experiment YAML 의 datasets: 에 추가 후 동일 명령으로 재실행
```

### 6. 새 retriever 추가하기 (5분)

`reco_bench` 의 핵심 설계 원칙은 **모든 retriever 가 한 추상 클래스
뒤에 숨는다** 는 것. Vector DB 든 ANN 라이브러리든 추가 절차는 동일:

1. `reco_bench/retrievers/<name>.py` — `Retriever` 상속,
   `build / search / save / load / device_info` 5개 메서드만 구현
2. `configs/retrievers/<name>.yaml` 작성 (기존 yaml 참고)
3. experiment yaml 의 `retrievers:` 리스트에 추가

`reco_bench/retrievers/base.py` 의 docstring 이 contract 의 상세.
**VDPU 통합도 동일** — 새 retriever 1개 + YAML 1개 + cost_model 1행.

### 7. 자주 묻는 문제

- **`milvus-lite is required`** → `pip install "pymilvus[milvus_lite]"`
- **`faiss-gpu` cublas symbol error** → main env 엔 faiss-gpu 설치 금지
  (PyTorch cublas 충돌). GPU 는 cuVS 사용, FAISS-GPU 는 격리 conda env.
- **MovieLens sha256 mismatch** → GroupLens mirror 재패키징 탓. 경고만
  출력하고 진행 (raw csv row count 동일하면 OK).
- **Recall@10 (vs GT) 가 매우 낮음** → ML-25M ID-only Two-tower 의 known
  한계 (~0.02). ANN 비교는 **Recall@10 vs exact** 로 볼 것 (모델 품질과
  독립). 근거: `reports/01_metric_design.md §2.4`.

## 이 벤치마크가 필요한 이유

디노티시아는 **VDPU (Vector Data Processing Unit)** 를 개발하고 있다.
Two-tower 검색의 핵심 연산인 벡터-벡터 유사도 및 vector-DB 인덱스 탐색을
가속하는 전용 실리콘이다. 현재까지의 공개 주장 (CPU 대비 10×, GPU 대비
6~7× 가성비) 은 추천 시스템 종사자가 자신의 워크로드로 인식할 수 있는
**현실적인 벤치마크** 에서 검증되어야 의미가 있다.

본 저장소가 그 워크로드이다. Phase 0+1 은 방법론을 정착시키고 GPU baseline
을 구축한다. Phase 2 에서 VDPU 를 끼워 넣고 **iso-recall cost ratio** 를
보고한다:

`Iso-recall cost ratio = ($ / QPS at recall R, GPU) / ($ / QPS at recall R, VDPU)`

헤드라인 "6~7×" 가 사실이라면, 추천 시스템이 실제로 운영하는 recall 구간
(보통 0.90~0.95) 에서 성립해야 한다. Phase 0 의 문서들이 그 의미를 정확히
정의한다.

## 참고한 오픈 벤치마크 / References

`reco_bench` 는 새로 구현되었지만, 측정 방법론과 컨벤션은 다음 공개
프로젝트들을 참고했다 (코드 fork 가 아니라 방법론 참조).

### 측정 방법론

- **[ann-benchmarks](https://github.com/erikbern/ann-benchmarks)** — 벡터 검색
  벤치마크의 사실상 표준. **Recall-QPS Pareto 곡선** 표현, 파라미터 sweep
  후 최적점 선택, **brute-force 정확 top-K 대비 recall** 측정 방식을 채택.
  → `reco_bench/retrievers/exact_topk.py`, `reports/figures/recall_vs_qps.png`,
  메트릭 `recall_vs_exact` (`reports/01_metric_design.md §2.4`).
- **[MLPerf Inference — Recommendation (DLRM)](https://github.com/mlcommons/inference/tree/master/recommendation)** —
  산업 표준 inference 벤치마크. **single-stream (P50/P95/P99 latency) +
  max-throughput (concurrency별 QPS)** 시나리오 분리, warmup→측정→동기화
  프로토콜을 채택. → `reco_bench/eval/profiler.py`.
- **[VectorDBBench](https://github.com/zilliztech/VectorDBBench)** — vector DB
  비교 벤치마크. **여러 vector DB 를 단일 frame 으로 비교** 하는 발상과
  **QPS@recall + $/query 동시 보고** 를 채택. → `Retriever` 추상 클래스,
  `reco_bench/eval/cost.py`.

### 추천 모델 / 데이터 / 학습

- **[Microsoft recommenders](https://github.com/recommenders-team/recommenders)** —
  MovieLens Recall@10 sanity 기준선, implicit feedback cutoff, k-core
  filtering 정책. → `reco_bench/data/preprocessing.py`,
  `reports/02_dataset_selection.md §3`.
- **[RecBole](https://github.com/RUCAIBox/RecBole)** — temporal / leave-last-N
  split 정책의 표준값. → `reco_bench/data/preprocessing.py`.
- **Google Two-tower** (Yi et al., *Sampling-bias-corrected neural modeling
  for large corpus item recommendations*, RecSys 2019) — in-batch
  negatives + **logQ correction** loss, two-tower 아키텍처.
  → `reco_bench/models/{two_tower,losses}.py`,
  `reports/03_baseline_methodology.md §1.4`.

### 데이터셋

- **[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)** (GroupLens) —
  sanity check. 인용: Harper & Konstan, *TOIS* 2015.
- **[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)** (McAuley Lab,
  UCSD) — medium-scale e-commerce. 인용: Hou et al., 2024.

세부 인용 형식(BibTeX)은 `reports/02_dataset_selection.md` 와 `CITATION.cff` 참조.
`reco_bench` 가 이 기존 벤치마크들의 어떤 빈틈을 메우는지는 `reports/00_overview.md §2`.

## 라이센스

- 코드: **Apache-2.0** (`LICENSE` 참조)
- 데이터셋: 각 원본 라이센스 유지 (연구용). `reports/02_dataset_selection.md`
  참조. raw 데이터는 재배포하지 않는다.

## 인용

본 벤치마크를 사용하는 경우 `CITATION.cff` 의 표준 인용 형식 참조.

## 기여

본 저장소는 Phase 1 종료 후 외부 기여를 받기 시작할 예정이다. 그 전까지의
설계 논의는 `reports/planning/` 에, 의사결정 기록은 `reports/history/` 에
누적된다.
