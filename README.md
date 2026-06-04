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
| 1 | Two-tower 학습 + ANN 라이브러리 4종 + Vector DB 2종 baseline + 5종 자동 시각화 + speedup table | ✅ 부분 완료 (ML-25M 측정 완료; Amazon Beauty 측정과 ScaNN 은 stretch) |
| 2 | VDPU 통합 (`reco_bench/retrievers/vdpu.py`) | ⏳ 대기 (실 하드웨어 도착 후) |

**Phase 1 핵심 발견** (`reports/06_phase1_findings.md` 의 narrative):
- ML-25M 의 24k items 같은 작은 corpus 에서 **FAISS-CPU HNSW 가 cuVS CAGRA 대비 74× 빠름** (`$0.0021/1M vs $0.3046/1M`).
- GPU ANN 가속의 진정한 가치는 corpus 가 100만 items 이상으로 커진 후부터 드러남.
- Vector DB (Milvus / Qdrant) 의 in-process 모드는 매우 느림 (1300× 차이) → 서버 모드 측정은 별도 protocol 필요.
- **이 모든 비교의 frame 위에 VDPU 한 행을 추가하는 것이 Phase 2.**

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

## 빠른 시작

> 가장 빠른 첫 결과는 [`QUICKSTART.md`](QUICKSTART.md) 참조. 한 페이지로
> 환경 설치 → 5가지 retriever 비교 결과까지 도달.

```bash
# 단일 흐름 (5~30분, 데이터셋 크기에 따라)
pip install -e . && pip install -r requirements.txt
pip install "pymilvus[milvus_lite]" qdrant-client scann
pip install --extra-index-url https://pypi.nvidia.com \
    cuvs-cu12 pylibraft-cu12 cupy-cuda12x

bash scripts/00_download_data.sh ml25m
bash scripts/10_train_two_tower.sh configs/experiments/phase1_full.yaml
bash scripts/20_build_index.sh    configs/experiments/phase1_full.yaml
bash scripts/30_run_benchmark.sh  configs/experiments/phase1_full.yaml
bash scripts/99_make_report.sh    configs/experiments/phase1_full.yaml
# → reports/baseline_results.md + reports/figures/*.png
```

비교 대상 (Phase 1 측정 완료, 한 명령 안에서 측정):

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
> FAISS-GPU 는 H100(sm_90) + PyTorch cublas ABI 충돌 때문에 격리 conda
> env 의 worker 로 측정 (`reports/05_reproducibility.md §2`).

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
