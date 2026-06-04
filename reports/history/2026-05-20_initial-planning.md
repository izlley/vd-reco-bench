---
date: 2026-05-20
phase: 0
topic: planning
status: completed
---

# 초기 기획

## What changed

- `reports/planning/2026-05-20_phase0-plan.md` 를 Phase 0 (문서) 와 Phase 1
  (GPU baseline 코드) 의 source-of-truth 설계 문서로 작성.
- Phase 0+1 의 네 모서리 (scope) 확정:
  1. **하드웨어**: GPU 전용 (VDPU 실리콘/시뮬레이터 아직 없음).
  2. **비교 baseline**: FAISS-GPU (IVF-PQ, HNSW는 CPU), cuVS CAGRA,
     ScaNN (CPU, stretch goal).
  3. **데이터셋**: MovieLens-25M (sanity) + Amazon Reviews 2023 Beauty /
     Books / Electronics (카테고리당 약 100K–1M items, medium 스케일).
  4. **산출물**: `reports/` 의 설계 문서 + 동작하는 baseline 스크립트.
- **VDPU 플러그인 seam** 확립: `Retriever` 추상 클래스의 단일 contract 는
  `build(item_emb) → search(queries, k)` 뿐. VDPU 하드웨어가 도착하면
  통합은 `reco_bench/retrievers/vdpu.py` 단일 파일 + YAML 한 개 + cost
  model 한 행 추가만으로 끝난다.
- **`reports/history/`** 를 시간순 설계 진화의 정식 저장 위치로 지정하고,
  **`reports/planning/`** 를 그 의사결정을 이끈 제안 문서의 저장소로
  분리.

## Why

- **GPU 전용 scope** 는 사용자의 명시적 지시이다. VDPU 실리콘이 도착하기
  전에 baseline 을 먼저 만들어 두는 것이 올바른 risk 순서이다. baseline
  쪽이 더 어렵고 시간이 많이 드는 부분 (모델 학습, ANN 파라미터 sweep,
  latency profiling 정확성) 이며, 이게 갖춰지면 VDPU 통합은 기계적 작업
  이 된다.
- **MovieLens-25M 은 sanity, Amazon Reviews 2023 은 실측** 의 조합은
  산업급 데이터셋 (Taobao UB, KuaiRec) 대신 선택되었다. 벤치마크의
  첫 임무는 가장 큰 corpus 를 다루는 것이 아니라 **방법론을 정착시키는
  것** 이기 때문이다. 대규모 corpus 는 iso-recall 비교 프레임이 정착된
  후의 Phase 2 sweep 차원이다.
- **`Retriever` 만이 유일한 플러그인 seam** 으로 둔 것은 비용 비교
  프레임을 하드웨어 종속성 없이 유지하기 위함이다. 파이프라인과
  `Evaluator` 는 인덱스가 GPU 위에 있든 VDPU 위에 있든 알 필요가 없으며,
  단지 query 벡터의 배치를 넘기고 왕복 시간만 측정한다.
- **`reports/history/`** 는 사용자의 후속 요구사항 (공개 시 작업 history
  보존) 을 해결한다. 이게 없으면 오픈소스 reader 는 최종 상태만 볼 수
  있고, 그 상태에 이르게 한 추론 과정을 볼 수 없다.

## Validation

- 계획된 방법론을 세 가지 공개 컨벤션과 교차 검증:
  - **ann-benchmarks** — Recall-QPS Pareto 곡선 포맷.
  - **MLPerf Inference recommendation (DLRM v2/v3)** —
    single-stream + max-throughput latency profiling 패턴.
  - **Microsoft `recommenders` ML-25M 노트북** — 학습된 Two-tower 가
    sanity check 받을 Recall@10 reference band.
- VDPU 의 공개 주장 (CPU 대비 10×, GPU 대비 6~7×) 의 출처는 hellot.net
  의 디노티시아 CDO 인터뷰 (2025년 3월). 내부 비공개 수치는 사용하지
  않는다. 벤치마크는 공개된 주장만으로 자기 자신을 입증 또는 반증해야
  한다.
- 주변 트리 (`/workspace/izlley/sllm/`) 에 추천 / ANN 관련 기존 코드는
  없음을 확인. `reco_bench` 는 greenfield 이며 따라야 할 legacy 패턴이
  없다.

## Open questions / next

- **GPU 모델**: baseline 을 어느 SKU 에서 돌릴 것인가? cost-model 의
  `$/hr` 칸과 에너지 측정 방법론 (H100 SXM5 vs PCIe vs H200 의 전력
  envelope 차이) 이 여기에 결정된다.
  → Phase 1 진입 시 결정, `reports/05_reproducibility.md` 에 기록.
- **cuVS / RAPIDS pod 설치 경로**: cuVS 는 특정 CUDA/RAPIDS 조합을
  요구한다. pod 의 CUDA 버전과 RAPIDS 24.10 의 floor 와의 호환성이
  필요하다. 호환되지 않으면 0.5~1일 risk.
- **ScaNN 포함 여부**: CPU 전용 + AVX2 gating. 설치가 까다로워지면
  Phase 2 로 미룰 수 있다. 핵심 비교 대상이 아니다.
- **History entry 규율**: "의미 있는 단계마다 entry" 라는 규칙은 commit
  과 함께 entry 가 묶여야만 동작한다. git init 후 pre-commit hook 또는
  PR 템플릿으로 강제하는 방안 필요.
