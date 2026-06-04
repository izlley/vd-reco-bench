# 00 · 개요 (Overview)

> 이 문서는 `reco_bench` 의 진입 문서이다. 처음 보는 reader 가 30분 안에
> 본 벤치마크의 목적, 범위, 비범위, 그리고 핵심 측정 결과의 의미를
> 이해할 수 있도록 작성되었다.

## 1. 한 줄 요약

`reco_bench` 는 **Two-tower 추천 시스템의 ANN 검색 단계** 의 비용 효율을
**iso-recall (동일 recall) 기준** 으로 측정하는 공개 벤치마크이며,
**vector-DB 가속기 (CPU / GPU / 디노티시아 VDPU) 와 오픈소스 vector DB
서버 (Milvus, Qdrant, Weaviate 등)** 를 모두 같은 프레임 안에서 공정하게
비교할 수 있게 한다.

## 2. 왜 만드는가 (Motivation)

추천 시스템 inference 의 비용 구조는 두 단계로 나뉜다:

```
                    online inference
   ┌──────────────────────────────────────────────────────────┐
   │                                                          │
   │    user features ─▶ [user tower] ─▶ user embedding       │
   │                                            │             │
   │                                            ▼             │
   │   (~108 items pre-computed) ──▶  [ANN top-K search] ──▶ K items
   │                                            │             │
   │                                            ▼             │
   │                                    [optional reranker]   │
   └──────────────────────────────────────────────────────────┘
```

대규모 서비스 (수천만~수억 item) 에서는 **ANN top-K search 단계가
inference 비용의 절대 다수를 차지** 한다. user tower forward 는 단일
embedding 한 번이지만, ANN 은 수억 item 벡터를 query 마다 훑어야 한다.

따라서:

- **CPU + FAISS** → 메모리 fit 되어도 batch 처리량/지연이 GPU 대비 열위.
- **GPU + FAISS-GPU / cuVS** → 빠르지만 HBM 가격과 전력 비용이 큼.
- **VDPU (디노티시아)** → 벡터-벡터 유사도와 vector-DB 자료구조 탐색에
  특화된 가속기. 공개 주장: CPU 대비 10×, GPU 대비 6~7× 가성비.

추천 시스템 엔지니어가 VDPU 도입 의사결정을 하려면, 자신의 워크로드와
비슷한 조건에서의 **iso-recall 비용 비교** 가 필요하다. 기존 벤치마크는
다음 중 하나의 한계를 가진다:

- **MLPerf DLRM, RecBole, Microsoft `recommenders`** — 모델 품질 위주,
  ANN 단계가 측정 대상이 아니거나 brute-force 사용.
- **ann-benchmarks, VectorDBBench** — ANN 시스템 자체는 잘 측정하지만,
  벡터가 GloVe, SIFT, OpenAI ada-002 같은 합성/일반 텍스트 임베딩이다.
  추천 임베딩과 분포가 다르고 (특히 popularity skew, dimensionality 선택)
  결과의 일반화가 의심스럽다.
- **VectorDBBench** — vector DB 서버를 black box 로 측정. 가속기 단위가
  아니라 시스템 단위.

`reco_bench` 는 그 사이를 메운다: **실제 Two-tower 모델을 학습 → 그 모델
의 item embedding → ANN 측정** 이라는 end-to-end 파이프라인을 일관된
프레임으로 제공한다.

## 3. 범위 (Scope) — Phase 0+1

이 phase 에서 다루는 것:

- **방법론 문서** (`reports/00_*.md ~ 05_*.md`)
  - 메트릭 정의, 데이터셋 선정 근거, baseline 방법론, VDPU 가치 명제의
    수식화, 재현성 기준.
- **GPU baseline 코드**
  - Two-tower 모델 학습 (in-batch negatives + sampled softmax with logQ).
  - Item embedding 추출.
  - FAISS-GPU (IVF-PQ), cuVS (CAGRA), FAISS-CPU (HNSW) 인덱스 빌드.
  - Query 스트림 생성 및 retrieval 평가.
  - 결과 자동 집계 (`scripts/99_make_report.sh` →
    `reports/baseline_results.md`).
- **데이터셋**
  - MovieLens-25M (sanity).
  - Amazon Reviews 2023 (Beauty, Books, Electronics).

이 phase 에서 다루지 않는 것 (Phase 2 이후):

- **VDPU 실 측정**. 실리콘/시뮬레이터 도착 후. 코드 통합 경로는 이미
  설계되어 있다 (`Retriever` 서브클래스 추가).
- **산업급 데이터셋** (Taobao UB, KuaiRec). 방법론이 정착한 후 sweep
  축으로 추가.
- **Ranker 단계** (cross-encoder, MoE ranker 등). retrieval 단계에서
  ANN 의 영향이 명확히 측정되어야 ranker 와의 분리가 가능.
- **Multi-tenant / multi-stream serving**. MLPerf inference 의
  multi-stream 시나리오는 Phase 2 stretch.

## 4. 핵심 측정 결과 (Phase 0+1 의 최종 산출물)

Phase 1 이 완료되면 다음 결과가 자동 생성된다:

### 4.1 Recall-QPS Pareto 곡선

각 retriever 별, dataset 별, hardware 별로:

```
       Recall@10
         ▲
    1.0  │           ●─●─●  cuVS CAGRA (H100)
         │         ●         ●
         │       ●             ●
    0.9  │     ●─●─●─●─●─●─●─●─●  FAISS-GPU IVF-PQ (H100)
         │   ●
         │  ●
    0.8  │ ●─●─●─●─●─●─●─●─●─●─●  FAISS-CPU HNSW (EPYC 7763)
         │
    0.0  └────────────────────────▶ QPS (log)
```

곡선 위의 모든 점은 (Recall, QPS, hardware, retriever, k) 5-튜플로 라벨링.

### 4.2 비용 효율 표

| Retriever | Hardware | Recall@10 | QPS | $/QPS | $/1M queries | W/QPS |
|---|---|---|---|---|---|---|
| FAISS-GPU IVF-PQ | H100 SXM5 80G | 0.952 | 24 k | $0.000094 | $0.094 | 29 |
| cuVS CAGRA | H100 SXM5 80G | 0.951 | 41 k | $0.000055 | $0.055 | 17 |
| FAISS-CPU HNSW | 2× EPYC 7763 | 0.949 | 6 k | $0.000063 | $0.063 | 90 |
| ScaNN CPU | 2× EPYC 7763 | 0.953 | 5 k | $0.000075 | $0.075 | 96 |
| *VDPU* | *TBD* | *—* | *—* | *—* | *—* | *—* |

(위 숫자는 schema 예시이며 실측이 아니다.)

VDPU 행은 Phase 2 에서 채워진다.

### 4.3 영업용 헤드라인 (Phase 2 산출, 본 phase 는 placeholder)

**Iso-recall cost ratio @ Recall@10 = 0.95**:

`($/QPS, 가장 저렴한 GPU baseline) ÷ ($/QPS, VDPU)` = **목표 6~7×**

이 수치의 정확한 수식과 측정 절차는 `01_metric_design.md` §4 (Cost) 와
`04_vdpu_value_proposition.md` 에서 정의된다.

## 5. Phase 로드맵

| Phase | 기간 | 산출물 | Definition of done |
|---|---|---|---|
| 0 | 1~2일 | `reports/00_*.md ~ 05_*.md` | reader 가 문서만으로 "6~7× 가 무엇을 의미하는가" 답 가능 |
| 1 | 1~2주 | Two-tower 학습 + **ANN 라이브러리 baseline** (FAISS, cuVS, ScaNN) 코드 + 자동 리포트 | MovieLens-25M + Amazon-Beauty 에서 `scripts/00_… → 99_…` end-to-end, `reports/baseline_results.md` 자동 생성 |
| 1.5 | TBD | **오픈소스 vector DB 서버 baseline** (Milvus, Qdrant, Weaviate) 통합 | 같은 단일 명령으로 vector DB 서버까지 측정 결과에 포함 |
| 2 | TBD | VDPU 통합, 산업급 데이터셋, 추가 카테고리 | VDPU 행이 비용 효율 표에 채워지고 iso-recall cost ratio 도출 |
| 2.5 | TBD | 공개 발표 (블로그/논문/오픈소스) | repo `git init` → GitHub publish, citation/leaderboard 운영 |

### 5.1 비교 대상 두 카테고리

`reco_bench` 는 다음 두 종류의 retriever 를 모두 `Retriever` 추상 클래스
하나로 추상화한다:

| 카테고리 | 형태 | Phase | 예시 |
|---|---|---|---|
| **A. ANN 라이브러리** | in-process (Python binding) | 1 | FAISS-CPU HNSW, **FAISS-GPU IVF-PQ**, cuVS IVF-PQ, cuVS CAGRA, ScaNN |
| **B. Vector DB 서버** | client/server (gRPC/HTTP) | 1.5 | **Qdrant (측정 완료)**, Milvus, Weaviate, pgvector |
| **C. VDPU** | 전용 가속기 + driver | 2 | Dnotitia VDPU FPGA / ASIC |

세 카테고리 모두 동일한 (Recall, QPS, $/1M queries) 튜플로 보고되므로
**reader 가 자기 환경에 맞는 비교** 를 즉시 할 수 있다.

## 6. 용어집

| 용어 | 정의 |
|---|---|
| **Two-tower** | User feature 와 item feature 를 각각 독립적인 encoder (tower) 로 매핑한 뒤 dot product (또는 cosine) 로 친화도를 계산하는 추천 아키텍처. YouTube DNN (2019), Google "Sampling-bias-corrected" (2019) 이 대표적. |
| **ANN** | Approximate Nearest Neighbor. 정확한 top-K 대신 빠른 근사 top-K. 대표 알고리즘: HNSW, IVF, IVF-PQ, CAGRA, ScaNN. |
| **Recall@K** | 사용자별로 ground-truth 정답 item 이 retrieval 결과 top-K 안에 들어 있을 확률의 평균. ANN 정확도와 모델 정확도 둘 다에 영향받음. |
| **Iso-recall** | 두 시스템을 비교할 때 둘의 recall 을 같은 값으로 맞추고 (parameter 튜닝으로) 그 조건에서 QPS/비용/지연을 비교하는 방법. "VDPU 가 7× 빠르다" 는 주장이 진짜인지 가짜인지 가르는 기준. |
| **QPS** | Queries Per Second. 1초당 처리한 query (사용자 1명에 대한 top-K retrieval) 의 수. 동시성 (concurrency) 조건이 다르면 의미가 달라진다. |
| **VDPU** | Vector Data Processing Unit. 디노티시아가 개발 중인 벡터 유사도/탐색 가속기. 공개 주장 CPU-10×, GPU-6~7× 가성비. 2025년 FPGA, 2026년 ASIC 양산 예정. |
| **logQ correction** | In-batch negative sampling 시 popular item 이 negative 로 더 자주 등장하는 bias 를 보정하기 위해 logit 에서 log(item 출현 빈도) 를 빼는 기법. Google "Sampling-bias-corrected" 논문의 핵심. |

## 7. 어디서 시작하는가

| 관심사 | 읽을 문서 |
|---|---|
| "이 벤치마크가 측정하는 메트릭이 정확히 뭔가?" | [`01_metric_design.md`](01_metric_design.md) |
| "어떤 데이터셋을 왜 골랐나?" | [`02_dataset_selection.md`](02_dataset_selection.md) |
| "Two-tower 학습부터 ANN 평가까지의 절차는?" | [`03_baseline_methodology.md`](03_baseline_methodology.md) |
| "VDPU 의 6~7× 주장은 어떻게 수치화되나?" | [`04_vdpu_value_proposition.md`](04_vdpu_value_proposition.md) |
| "내 머신에서 재현하려면?" | [`05_reproducibility.md`](05_reproducibility.md) |
| "최종 baseline 숫자는?" | `baseline_results.md` (Phase 1 완료 후 자동 생성) |
| "왜 지금 모습인지의 역사" | [`history/0000_index.md`](history/0000_index.md) |
| "원본 plan 과 RFC" | [`planning/`](planning/) |
