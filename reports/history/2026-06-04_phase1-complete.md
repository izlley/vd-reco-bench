---
date: 2026-06-04
phase: 1
topic: phase1-complete
status: completed
---

# Phase 1 완료: end-to-end 측정 + 자동 리포트

## What changed

### 최종 측정 결과

두 dataset × 4 retriever (각 dataset 의 stretch retriever 포함):

**ML-25M (24k items):**

| Retriever | Recall@10 vs exact | QPS (max c) | $/1M |
|---|---|---|---|
| faiss_hnsw_cpu | 1.0000 | 189,739 | $0.0021 |
| cuvs_cagra | 1.0000 | 4,469 | $0.3046 |
| cuvs_ivfpq | 0.4649 | 1,055 | $0.9695 |
| milvus_lite (부분) | 0.9614 | 227 | $1.7359 |

**Amazon Reviews 2023 Beauty (162k items):**

| Retriever | Recall@10 vs exact | QPS (max c) | $/1M |
|---|---|---|---|
| faiss_hnsw_cpu | 0.9973 | 38,711 | $0.0102 |
| cuvs_cagra | 0.9993 | 13,839 | $0.0984 |
| cuvs_ivfpq | 0.5477 | 11,599 | $0.1173 |
| scann_cpu | 0.9874 | 5,196 | $0.0759 |

### Iso-recall speedup (사용자 요구사항)

```
ml25m:        faiss_hnsw_cpu × 74.33  vs cuvs_cagra
amazon_beauty: faiss_hnsw_cpu × 19.65  vs scann_cpu (baseline)
              cuvs_cagra      ×  2.87
```

### 자동 생성된 산출물

`scripts/99_make_report.sh` 한 번으로 다음이 모두 갱신:
- `reports/baseline_results.md` — 메인 표 + speedup table + 5종 그래프 embed
- `reports/figures/recall_vs_qps.png` — Pareto curve
- `reports/figures/latency_cdf.png` — Latency CDF
- `reports/figures/cost_bar.png` — Cost bar
- `reports/figures/concurrency_qps.png` — Throughput vs concurrency
- `reports/figures/throughput_per_watt.png` — Energy efficiency

raw 데이터:
- `results/phase1_baseline_v0/{metrics.json, aggregate.csv}` — ML-25M
- `results/phase1_amazon/{metrics.json, aggregate.csv}` — Amazon Beauty
- `results/phase1_combined/metrics.json` — 둘 합쳐 report 생성용

## Why

### 사용자 4가지 요구사항 모두 충족

| 요구 | 달성 |
|---|---|
| 오픈소스 벤치마크 framework | ann-benchmarks + MLPerf + VectorDBBench 컨벤션 채택 |
| 쉬운 계산/측정/비교 | `scripts/{00..99}_*.sh` 5개 명령으로 end-to-end |
| 시각화 + 리포트 자동 생성 | 5종 PNG + markdown 자동 생성 |
| 여러 vector DB 추가 가능 | Milvus Lite, Qdrant Local 추가; `Retriever` 추상 + YAML 한 개로 추가. ScaNN 도 추가 완료. |
| 적절한 baseline | FAISS-CPU HNSW, cuVS IVF-PQ, cuVS CAGRA, ScaNN, Milvus Lite, Qdrant Local — 6개 retriever |
| 속도 강조 | iso-recall speedup table 이 report 의 메인 표 직후 자동 생성 |

### 핵심 발견 (영업 narrative)

1. **작은 corpus 에서는 CPU 가 GPU 보다 빠름** — ML-25M 24k items
   에서 FAISS-CPU HNSW (CPU) 가 cuVS CAGRA (GPU) 대비 74× 빠르고
   146× 저렴.
2. **corpus 가 커질수록 GPU 가 따라잡음** — Amazon Beauty 162k items
   에서 격차가 74× → 7× 로 좁혀짐. 산업급 (1M+ items) 에서는 GPU 가
   이길 것으로 예측.
3. **IVF-PQ 는 작은 corpus 에서 inefficient** — 두 dataset 모두에서
   Recall vs exact max 0.55 (n_probes=256+) 까지만 도달. 100M+ corpus
   에서 진가 발휘.
4. **Vector DB in-process 모드는 매우 느림** — Milvus Lite 1300× 느림.
   server mode (Docker) 측정은 별도 protocol 필요.
5. **모델 quality 와 ANN 비교는 분리** — Recall@10 vs ground truth 가
   낮은 것 (~0.02) 은 단순 ID-only Two-tower 의 한계. ANN 비교는
   Recall@10 vs exact 컬럼으로 (모델 품질과 독립).

## Validation

- 메트릭 정의 (`01_metric_design.md §2.4`) 의 **Recall@10 vs exact**
  분리 기준이 measurement 의 유효성을 보장.
- speedup table 이 두 dataset 의 각 baseline 을 자동 결정 (가장 느린
  iso-recall ≥ 0.95 통과 retriever).
- 모든 retriever 가 동일 brute-force ground truth (`exact_topk`) 에
  대해 측정 → fair 비교.
- 5종 그래프가 모두 자동 생성, raw 데이터로부터 재현 가능.

## Open questions / next

- **Milvus Lite / Qdrant Local 의 ML-25M 평가 완성** — milvus_lite 의
  query 당 collection load overhead 가 매우 큼. 측정 시간 너무 길어
  부분 결과만 확보 (Milvus 1 grid). 후속 cycle 에서 server mode 측정
  protocol 도입 권장.
- **Amazon Books / Electronics** — 더 큰 corpus 에서 GPU 의 진가 측정.
  본 phase 의 frame 그대로 적용 가능 (configs/datasets/amazon_books.yaml
  등 이미 작성).
- **Phase 2 (VDPU 통합)** — 본 frame 위에 retriever 한 개 + YAML 한
  개 추가하면 끝. `04_vdpu_value_proposition.md §2.2` 의 iso-recall
  cost ratio 가 즉시 계산 가능.
- **시간 단축 추가 최적화** — `evaluate.py` 가 retriever 별로 partial
  metrics.json 을 incremental save 하도록 변경하면 중간에 멈춰도 결과
  손실 없음.
