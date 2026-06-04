---
date: 2026-06-04
phase: 1
topic: faiss-gpu-eval
status: completed
---

# FAISS-GPU IVF-PQ 평가 (격리 conda env)

## What changed

- 사용자 요청으로 **FAISS-GPU IVF-PQ** 를 baseline 에 추가 측정.
- env-setup (`2026-06-04_env-setup.md`) 에서 PyTorch cublas ABI 충돌로
  제외했던 것을, **격리 conda env + PyTorch 미사용 worker** 로 우회 측정.
- 추가 파일:
  - `scripts/dump_eval_arrays.py` — main python(PyTorch)으로 query_emb /
    exact top-K / ground truth 를 `results/_eval_arrays/<ds>.npz` 로 dump.
  - `scripts/faiss_gpu_worker.py` — conda faiss env 에서 npz 읽어 FAISS-GPU
    IVF-PQ build + nprobe grid search + recall/latency/QPS 측정.
  - `configs/retrievers/faiss_ivfpq_gpu.yaml` — 재생성 (conda worker 명시).
- `scripts/combine_results.py` 기본 목록에 `phase1_faiss_gpu` 추가.
- 측정 결과 통합 → `baseline_results.md` 가 **6 retriever × 2 dataset =
  57 rows**.

## 측정 결과

| Dataset | recall@10 vs exact (max) | QPS (max) | $/1M (min) |
|---|---|---|---|
| ML-25M | 0.4181 | **356,097** | $0.0027 |
| Amazon Beauty | 0.5102 | **56,810** | $0.0028 |

- **GPU 진영 최고 throughput**. 같은 IVF-PQ 인 cuVS IVF-PQ (1,055 /
  11,599 QPS) 보다 수십~수백 배 빠름 — FAISS 의 batch GPU search 가
  효율적.
- 단 **recall vs exact 0.42~0.51** (PQ 압축 손실, nprobe 올려도 saturate)
  → iso-recall 0.95 speedup 표엔 미포함. Recall-QPS Pareto 의 "초고속·
  저recall" 영역.

## Why

- **conda-forge faiss-gpu 1.10** 이 H100(sm_90) 커널을 포함. PyPI
  `faiss-gpu-cu12` 는 sm_90 미지원(`CUDA error 209`)이라 불가.
- **격리 env** 인 이유: faiss-gpu 가 cublas 12.9 를 끌어와 main env 의
  PyTorch(cublas 12.8 핀)를 깨뜨림. worker 를 별도 conda python 으로
  돌려 ABI 격리. query/exact/gt 를 npz 로 주고받아 PyTorch 의존 제거.
- **cost 후처리**: worker 는 power/cost 를 측정 못해 0 으로 남겼고,
  H100 SKU 단가($4.90/hr)와 max QPS 로 `$/1M` 를 후처리 계산.
  (`reports/01_metric_design.md §4.2` 의 공식.)

## Validation

- conda faiss 1.10 H100 smoke: IVF-PQ build+search 정상 (sm_90 커널 OK).
- recall vs exact 가 nprobe 증가에 따라 단조 상승 (0.17→0.51) — 측정
  타당성 확인.
- 통합 후 report 재생성: faiss_ivfpq_gpu row 가 메인 표 + Recall-QPS
  Pareto 에 반영.
- GPU 측정 중 dummy/keepalive 정지 → 측정 후 복구 (4 GPU 100% 유지).

## Open questions / next

- **cuVS IVF-PQ vs FAISS-GPU IVF-PQ 의 QPS 격차(수백 배)** 는 측정 환경
  차이(main 파이프라인 vs 격리 worker)도 일부 기여 가능 → 동일 harness
  로 재측정 시 격차 재확인 필요.
- IVF-PQ 의 low recall 은 PQ 코드북 손실. `pq_bits` 상향 또는 IVF-Flat
  (무압축) 변형으로 high-recall 영역 측정 가능 (Phase 2).
- FAISS-GPU 를 main 파이프라인의 `Retriever` 로 통합하려면 subprocess
  bridge 가 필요 (현재는 standalone worker). 향후 정식 통합 검토.
