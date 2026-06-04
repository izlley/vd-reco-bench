---
date: 2026-05-20
phase: 0
topic: phase0-complete
status: completed
---

# Phase 0 문서 완료 + 코드 seam 안착

## What changed

- `reports/` 의 Phase 0 deliverable 6개 모두 작성 완료:
  - `00_overview.md` — 외부 reader 용 진입점, 동기와 scope, 용어집.
  - `01_metric_design.md` — Quality / System / Trade-off / Cost 의 4
    카테고리 메트릭을 수식 수준에서 정의. **iso-recall cost ratio** 와
    **Recall@K vs exact** (모델 오차와 ANN 오차의 분리) 를 명문화.
  - `02_dataset_selection.md` — ML-25M, Amazon Reviews 2023 의 선정
    근거, 라이센스, citation, 전처리 절차, sanity check 기준.
  - `03_baseline_methodology.md` — Two-tower 아키텍처, 학습 프로토콜
    (sampled softmax + in-batch neg + logQ), retriever 별 sweep grid,
    평가 절차, hardware 캡처.
  - `04_vdpu_value_proposition.md` — "GPU 대비 6~7×" 영업 narrative 의
    수식화. TCO 모델, sensitivity sweep schema, anti-narrative.
  - `05_reproducibility.md` — 재현성 3 층위 (L1/L2/L3), 환경 핀, 시드
    정책, 비교 절차, 알려진 비결정성.
- `configs/` 디렉토리에 baseline 실험에 필요한 YAML 모두 작성:
  `cost_model.yaml`, 4개 dataset, 1개 model, 4개 retriever, 1개
  experiment 통합 (`phase1_baseline.yaml`).
- `reco_bench/` Python 패키지의 핵심 seam 구현:
  - `retrievers/base.py` — `Retriever` 추상 클래스. **VDPU 통합의 단일
    플러그인 포인트**. CPU-side numpy 반환 contract, warmup hook,
    device sync 의 기대를 docstring 에 박아둠.
  - `eval/metrics.py` — Recall@K, HitRate@K, NDCG@K, MRR, 그리고
    ANN-isolation 의 `recall_at_k_vs_exact`.
  - `data/`, `models/`, `pipelines/`, `utils/` 의 빈 `__init__.py`
    (다음 단계에서 채워질 placeholder).
- 단위 테스트 스켈레톤 (`tests/test_metrics.py`,
  `tests/test_retriever_contract.py`).

## Why

- 사용자가 한글 작성과 reports/ 누적을 명시했으므로 Phase 0 의 모든
  설계 문서는 한글로 작성되었다. 영어 (Apache 라이센스, BibTeX,
  CITATION.cff) 만 표준 포맷 유지.
- 메트릭 정의를 코드보다 먼저 못 박아 두는 것은 의도된 risk 순서이다.
  "6~7× 가 무엇을 의미하는가" 가 합의된 후에야 그것을 측정하는 코드를
  쓸 의미가 있다.
- `Retriever` 추상은 4개 메서드만 가지며, 그 외의 모든 측정/비교 로직은
  `eval/` 과 `pipelines/` 가 담당. 이 분리가 깨지면 VDPU 통합이 다시
  파이프라인 전체를 건드리게 된다.
- `metrics.py` 와 `Retriever.base` 만으로 인라인 sanity check 가
  돌아가도록 의존성을 numpy 로 한정. pytest/torch/faiss 가 없어도
  CI 의 가벼운 lint 단계에서 import 검증 가능.

## Validation

- 모든 reports 문서가 각 §1 ~ §N 의 헤더 구조로 self-contained 함을
  검증 (각 문서의 reader 가 다른 문서 없이도 그 문서의 주제를 이해
  가능).
- `reco_bench.eval.metrics.compute_all` 을 numpy-only 환경에서 실행해
  기대값 일치 확인:
  - recall@1 = 0.75 (계산: (1/1 + 1/2) / 2)
  - recall@3 = 1.0
  - recall@2 partial = 0.5
- `Retriever` 추상 클래스를 인라인 DummyRetriever 로 인스턴스화해 build
  → search → warmup round-trip 동작. 점수 행별 내림차순 정렬, ID
  remap 정확성 모두 확인.

## Open questions / next

- **다음 단계 = Phase 1 코드 구현**:
  - `reco_bench/data/loaders.py` — ML-25M 다운로드/전처리.
  - `reco_bench/models/two_tower.py` — 아키텍처 + sampled softmax loss.
  - `reco_bench/retrievers/{faiss_cpu, faiss_gpu, cuvs, scann_cpu}.py` —
    각각 `Retriever` 의 구체 구현.
  - `reco_bench/pipelines/{train, build_index, evaluate, report}.py`.
  - `scripts/00_~99_*.sh`.
- **GPU 환경 확인 필요**: 본 Pod 의 CUDA 버전과 H100 SXM5 / PCIe 여부
  를 `nvidia-smi` 로 확인한 뒤 `reports/05_reproducibility.md §2` 의
  환경 핀을 tighten.
- **데이터 다운로드 위치**: 본 Pod 의 `/workspace/data` 는 비영구이고
  CephFS 만 영구라는 사용자 메모리가 있다. 큰 raw 데이터는 어디에
  둘지 결정 필요.
- **공개 시점**: 영업 narrative 의 결론 (VDPU 측정) 이 없는 상태에서
  Phase 0 의 방법론 문서만 먼저 공개할지, Phase 2 측정 후 통합 공개할지
  의사결정 필요.
