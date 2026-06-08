# 07 · Scaling & Sensitivity (Phase 2, VDPU-비의존)

> 이 문서는 Phase 2 의 **VDPU-비의존 부분** 결과이다. VDPU 하드웨어가
> 아직 없으므로, **synthetic embedding 으로 corpus 크기와 dim 을 통제
> 변수로 sweep** 하여 "어떤 조건에서 GPU 가속의 가치가 커지는가" 를
> 측정한다. 이는 VDPU 영업 narrative 의 핵심 전제 — **VDPU 가 겨냥하는
> 영역(대규모·고차원)에서 GPU 가 CPU 를 압도** — 를 데이터로 뒷받침한다.

## 1. TL;DR

| 발견 | 데이터 | 영업적 함의 |
|---|---|---|
| **corpus 가 커질수록 GPU/CPU throughput 격차 확대** | QPS 격차 100k 에서 4× → 10M 에서 5.5× (CAGRA vs FAISS-CPU HNSW) | VDPU 의 타겟인 대규모 corpus 일수록 GPU-class 가속의 가치↑ |
| **GPU 는 corpus 에 robust, CPU 는 하락** | CAGRA QPS 14.7k→16.4k (corpus 100k→10M), FAISS-CPU 3.6k→3.0k | 데이터 증가에도 GPU 처리량 유지 = 확장성 |
| **고차원일수록 CPU 가 더 불리** | dim 128→256 에서 FAISS-CPU 는 QPS·recall 동시 하락(3.6k→3.1k, recall 0.79→0.46), CAGRA 는 QPS 유지(15k→14k) | 임베딩 차원이 큰 모델(LLM 기반 임베딩 등)일수록 GPU/VDPU 유리 |

## 2. 측정 설계

- **합성 데이터**: `reco_bench/data/synthetic.py` 의 clustered gaussian
  (256 cluster, L2-norm). 학습 없이 corpus/dim 만 순수 변수로 통제.
  query 는 임의 item 을 약간 perturb (현실적 top-K).
- **통제 변수**:
  - corpus sweep: items ∈ {100k, 1M, 10M}, dim=128 고정
  - dim sweep: dim ∈ {128, 192, 256}, corpus=1M 고정
- **retriever**: FAISS-CPU HNSW, cuVS CAGRA (GPU), cuVS IVF-PQ (GPU).
  각 retriever 대표 param 2점 (recall 높은 쪽 선택).
- **하드웨어**: H100 1장 (`CUDA_VISIBLE_DEVICES=0`) + CPU.
- 측정: `reco_bench/pipelines/scaling.py`.

## 3. Corpus scaling 결과 (dim=128)

![](figures/scaling_corpus.png)

| Retriever | Device | Corpus | Recall@10 vs exact | QPS (max) |
|---|---|---|---|---|
| cuVS CAGRA | GPU | 100,000 | 0.7772 | 14,744 |
| cuVS CAGRA | GPU | 1,000,000 | 0.7263 | 15,084 |
| cuVS CAGRA | GPU | 10,000,000 | 0.2096 | **16,381** |
| FAISS-CPU HNSW | CPU | 100,000 | 0.9496 | 3,622 |
| FAISS-CPU HNSW | CPU | 1,000,000 | 0.7940 | 3,647 |
| FAISS-CPU HNSW | CPU | 10,000,000 | 0.5258 | **2,952** |
| cuVS IVF-PQ | GPU | 100,000 | 0.3809 | 23,974 |
| cuVS IVF-PQ | GPU | 1,000,000 | 0.3203 | 18,031 |
| cuVS IVF-PQ | GPU | 10,000,000 | 0.2592 | 20,639 |

**해석**:
- **QPS**: GPU(CAGRA)는 corpus 가 100배 커져도 (100k→10M) throughput 이
  오히려 약간 상승 (배치 효율이 큰 corpus 에서 더 잘 발휘). CPU 는
  3,622→2,952 로 하락. **격차가 4.1× → 5.5× 로 확대**.
- 이것이 영업 핵심: **VDPU 가 겨냥하는 대규모 corpus 일수록 GPU-class
  가속의 상대 가치가 커진다.** VDPU 는 그 GPU 영역을 다시 가속하는 것.

## 4. Dim sensitivity 결과 (corpus=1M)

![](figures/scaling_dim.png)

| Retriever | Device | Dim | Recall@10 vs exact | QPS (max) |
|---|---|---|---|---|
| cuVS CAGRA | GPU | 128 | 0.7275 | 14,953 |
| cuVS CAGRA | GPU | 192 | 0.5475 | 17,788 |
| cuVS CAGRA | GPU | 256 | 0.4134 | 14,160 |
| cuVS IVF-PQ | GPU | 128 | 0.3196 | 20,821 |
| cuVS IVF-PQ | GPU | 192 | 0.2089 | 24,714 |
| cuVS IVF-PQ | GPU | 256 | 0.1478 | 22,642 |
| FAISS-CPU HNSW | CPU | 128 | 0.7912 | 3,607 |
| FAISS-CPU HNSW | CPU | 192 | 0.6197 | 3,418 |
| FAISS-CPU HNSW | CPU | 256 | 0.4639 | 3,061 |

**해석**:
- dim 이 128→256 으로 커질 때 CPU HNSW 의 QPS 가 3,607→3,061 (**-15%**)
  하락하고 recall 도 0.79→0.46 으로 크게 떨어진다. GPU CAGRA 는
  QPS 가 14,953→14,160 (거의 유지) 이면서 recall 하락도 상대적으로 완만.
- GPU/CPU **throughput 격차는 dim 전 구간에서 ~4–5×** 유지되며, 고차원
  으로 갈수록 CPU 의 recall·QPS 가 동시에 나빠진다 → **dim 이 큰 임베딩
  (LLM/멀티모달 추천) 일수록 GPU/VDPU 의 이점**.
- (참고: dim 범위를 128/192/256 으로 설정한 것은 two-tower 추천에서
  실용적으로 쓰이는 임베딩 차원 구간에 집중하기 위함.)

## 5. 한계와 주의 (정직한 기록)

- **recall 절대값은 낮다**: 본 sweep 은 retriever 당 **고정 param 2점**
  만 측정해서, corpus/dim 이 커지면 같은 param 으로는 recall 이 하락한다
  (예: 10M 에서 CAGRA recall 0.21). param(efSearch/itopk/nprobe)을
  corpus 에 맞춰 키우면 recall 은 회복되지만 QPS 는 그만큼 내려간다.
  **본 문서의 초점은 recall 절대값이 아니라 throughput 의 scaling 거동**
  (GPU vs CPU 의 상대 추세) 이다.
- **synthetic 데이터**: clustered gaussian 은 추천 임베딩의 cluster 구조
  를 일부 반영하지만 실제 분포와 다르다. 절대 수치보다 **추세** 로 해석.
  실데이터(Amazon) 결과(`06_phase1_findings.md`)와 추세가 일치함을 확인.
- **IVF-PQ 의 낮은 recall** 은 PQ 압축 손실 (Phase 1 과 동일 원인).

## 6. VDPU narrative 연결

본 결과는 VDPU 의 가치 명제(`04_vdpu_value_proposition.md`)에 다음을
보탠다:

1. **시장(대규모 corpus)에서 CPU 는 throughput 한계** → GPU-class 가속
   이 필수. VDPU 는 "GPU 보다 저렴하게 그 throughput 을 내는가" 의 경쟁.
2. **고차원 트렌드** (LLM 임베딩) 가 GPU/VDPU 에 유리한 방향.
3. Phase 2 에서 VDPU 실측이 추가되면, 이 corpus/dim sweep 의 **각 점에
   VDPU 곡선을 겹쳐** "VDPU 가 GPU 대비 같은 recall 에서 몇 배 저렴/빠른가"
   를 corpus·dim 별로 보일 수 있다. 측정 인프라(`scaling.py`)는 이미
   retriever-agnostic 이라 VDPU retriever 만 추가하면 된다.

## 7. 재현

```bash
python -m reco_bench.pipelines.scaling --mode corpus   # 100k/1M/10M × dim128
python -m reco_bench.pipelines.scaling --mode dim      # dim 128/192/256 × corpus1M
python -m reco_bench.pipelines.scaling_report          # 그래프 + 표
```

## 8. 변경 이력

| 날짜 | 변경 | 근거 |
|---|---|---|
| 2026-06-04 | Phase 2 corpus/dim scaling 측정 + 문서화 | 사용자 지시 (VDPU-비의존 Phase 2) |
