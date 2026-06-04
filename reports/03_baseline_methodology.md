# 03 · Baseline 방법론 (Methodology)

> 이 문서는 Phase 1 에서 구현할 GPU baseline 의 **모델 아키텍처**,
> **학습 절차**, **인덱스 빌드 파라미터**, **평가 절차**, **하드웨어
> 매트릭스** 를 정의한다. `pipelines/*.py` 의 구현은 본 문서를 그대로
> 따른다.

---

## 1. Two-tower 아키텍처

### 1.1 전체 그림

```
              user features                       item features
                   │                                    │
                   ▼                                    ▼
        ┌─────────────────────┐               ┌────────────────────┐
        │  User Tower         │               │  Item Tower        │
        │  (학습 가능)         │               │  (학습 가능)        │
        │                     │               │                    │
        │  ID embedding +     │               │  ID embedding +    │
        │  recent-N mean-pool │               │  (text encoder,    │
        │  + MLP              │               │   Amazon only,     │
        │                     │               │   frozen) + MLP    │
        └─────────────────────┘               └────────────────────┘
                   │                                    │
                   ▼                                    ▼
            user emb u (d-dim)                  item emb v (d-dim)
                            \                   /
                             \   dot product   /
                              \─────●─────────/
                                     │
                                  score(u, v)
```

- **공유 차원** $d = 128$ (Phase 1 고정). dim sweep 은 Phase 2.
- **정규화**: 두 tower 의 출력에 L2 normalize → score = cosine.
- **유사도**: cosine. (FAISS HNSW/IVF, cuVS CAGRA 모두 inner product
  지원하며, 정규화 후에는 inner product = cosine.)

### 1.2 User tower 상세

```
input:
  user_id           int
  recent_item_ids   List[int] (최근 N 개의 positive 상호작용, N=50)

forward:
  u_id_emb = UserEmbedding(user_id)                    # (d_id,)
  recent_embs = ItemEmbedding(recent_item_ids)         # (N, d_id)
  recent_mean = mean(recent_embs, axis=0)              # (d_id,)
  concat = [u_id_emb, recent_mean]                     # (2*d_id,)
  u = MLP(concat) → ReLU → MLP → L2_normalize          # (d,)
```

- `d_id = 64`, hidden MLP = 256, output `d = 128`.

### 1.3 Item tower 상세

**ML-25M (ID-only):**
```
v_id_emb = ItemEmbedding(item_id)                     # (d_id,)
v = MLP(v_id_emb) → ReLU → MLP → L2_normalize          # (d,)
```

**Amazon (ID + text):**
```
v_id_emb = ItemEmbedding(item_id)                     # (d_id,)
v_text_emb = SentenceTransformer(item.text)           # (384,), frozen
concat = [v_id_emb, v_text_emb]                       # (d_id + 384,)
v = MLP(concat) → ReLU → MLP → L2_normalize           # (d,)
```

ItemEmbedding 은 user tower 의 ItemEmbedding 과 **공유** (sub-linear 시
contextual bandit / dual tower 와의 통합 용이).

### 1.4 Loss

**Sampled softmax with in-batch negatives + logQ correction.** Google 의
"Sampling-bias-corrected" (Yi et al., 2019) 공식:

배치 안의 $(u_i, v_i)$ pair 에 대해:

$$
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B}
\log \frac{\exp(s_{ii} - \log Q(v_i))}{\sum_{j=1}^{B} \exp(s_{ij} - \log Q(v_j))}
$$

- $s_{ij} = \langle u_i, v_j \rangle / \tau$.
- $\tau = 0.07$ (CLIP / DCG 표준).
- $Q(v)$: streaming frequency estimate of item $v$ (count-min sketch
  size = 2^20).
- logQ 보정은 popular item 이 negative 로 더 자주 등장하는 bias 를
  완화한다.

추가: **hard negative mining** 은 Phase 2 (구현 단순성 우선).

---

## 2. 학습 프로토콜

| 항목 | ML-25M | Amazon-Beauty | Amazon-Books | Amazon-Electronics |
|---|---|---|---|---|
| Batch size | 4096 | 4096 | 2048 | 2048 |
| Epoch (max) | 10 | 5 | 5 | 5 |
| Optimizer | AdamW lr=1e-3, wd=1e-4 | 동일 | 동일 | 동일 |
| LR scheduler | cosine, warmup 1k steps | 동일 | 동일 | 동일 |
| Gradient clip | 1.0 | 동일 | 동일 | 동일 |
| Early stop | val Recall@100 plateau 3 epoch | 동일 | 동일 | 동일 |
| Precision | bfloat16 (matmul), fp32 (norm/loss) | 동일 | 동일 | 동일 |
| GPU | 1× H100 80G (Phase 1 가정) | 동일 | 동일 | 동일 |
| Seed | 42 | 42 | 42 | 42 |
| Random sampler | RandomSampler with `seed=42` | 동일 | 동일 | 동일 |

학습 로그:
- TensorBoard event → `logs/tb_<exp_id>/`
- 텍스트 로그 → `logs/train_<exp_id>.log`
- val Recall@10, Recall@100, loss epoch 별 기록.

산출:
- `checkpoints/<exp_id>/user_tower.pt`
- `checkpoints/<exp_id>/item_tower.pt`
- `checkpoints/<exp_id>/item_embeddings.npy` (shape: `(num_items, 128)`,
  dtype `float32`)
- `checkpoints/<exp_id>/item_ids.npy` (shape: `(num_items,)`, dtype
  `int64`, item embedding 행과 동순서)
- `checkpoints/<exp_id>/train_meta.json` (config snapshot, val 메트릭,
  학습 시간)

### 2.1 Sanity threshold (Phase 1 DoD 의 일부)

ML-25M val Recall@100 ≥ **0.30** 이 되어야 학습 success. (Microsoft
`recommenders` NCF/SAR baseline 의 ML-25M Recall@10 = 0.10~0.15 / Recall@100 ≈ 0.35
근방 기준.)

미달 시 학습 hyperparameter 조정 후 재시도. baseline benchmark 가 약한
모델 위에서 측정되면 cost ratio 의 의미가 사라진다.

---

## 3. ANN 인덱스 빌드

### 3.1 Retriever 비교 매트릭스 (Phase 1)

| Retriever | 구현체 | Index | 디바이스 | 비고 |
|---|---|---|---|---|
| FAISS-CPU HNSW | `faiss-cpu` 1.14 | HNSW | CPU | CPU baseline 표준. GPU 미지원. |
| cuVS IVF-PQ | `cuvs.neighbors.ivf_pq` (cuVS 26.04) | IVF + PQ | GPU | GPU baseline 의 메모리 효율 측. (원래 FAISS-GPU IVF-PQ 를 쓸 계획이었으나 PyTorch 와 cublas ABI 충돌해 cuVS 의 동등 구현체로 변경 — `reports/history/2026-06-04_env-setup.md` 참조.) |
| cuVS CAGRA | `cuvs.neighbors.cagra` (cuVS 26.04) | CAGRA (graph) | GPU | GPU baseline 의 속도 측 (VDPU 의 진짜 경쟁) |
| ScaNN | `scann` (pip) | scann_partitioned | CPU | Google 의 CPU 강자, AVX2 gated, stretch |

### 3.2 파라미터 grid (Recall-QPS 곡선용)

**FAISS-CPU HNSW:**
```yaml
build:
  M: [16, 32, 64]
  efConstruction: 200
search:
  efSearch: [16, 32, 64, 128, 256, 512]
```

**cuVS IVF-PQ:**
```yaml
build:
  n_lists: [1024, 4096, 16384]
  pq_dim: 16     # PQ subquantizer 수 (dim=128 → 8-dim subvector)
  pq_bits: 8     # PQ code 비트
search:
  n_probes: [1, 4, 16, 64, 256]
```

**cuVS CAGRA:**
```yaml
build:
  graph_degree: [32, 64, 128]
  intermediate_graph_degree: 96
search:
  itopk_size: [32, 64, 128, 256, 512]
```

**ScaNN:**
```yaml
build:
  num_leaves: [1024, 4096]
  num_leaves_to_search: [50, 200, 500]
  reorder_size: [100, 500, 2000]
search:
  pre_reorder_num_neighbors: 100
```

각 grid 점에서 (Recall, QPS) 한 쌍 → Recall-QPS plot 의 한 점.

### 3.3 빌드 시점 측정

빌드는 1회만 실행하고 결과를 `indexes/<retriever>/<dataset>/<param>/` 에
직렬화. 다음을 `results/<exp_id>/build.json` 에 기록:

```json
{
  "retriever": "cuvs_cagra",
  "dataset": "ml25m",
  "params": {"graph_degree": 64, "intermediate_graph_degree": 96},
  "wall_seconds": 12.3,
  "peak_host_mb": 1450,
  "peak_device_mb": 880,
  "index_disk_bytes": 31_457_280,
  "num_items": 62423,
  "dim": 128,
  "build_started": "2026-05-21T13:02:18+09:00",
  "hardware": "H100_SXM5_80G"
}
```

---

## 4. 평가 프로토콜

### 4.1 Query set 생성

1. `checkpoints/<exp_id>/user_tower.pt` 로 test split 의 user 들에 대해
   user embedding 일괄 계산 → `queries.npy` (shape: `(num_test_users, 128)`).
2. 같은 user_id 의 ground-truth item 집합 (test split 의 미래 interaction)
   → `gt.pkl`.

### 4.2 Quality 평가 (offline)

전체 query 를 한 배치로 retriever 에 dispatch:
```python
ids, scores = retriever.search(queries, k=100)
```

- Recall@10, Recall@100, NDCG@10, HitRate@10, MRR 계산.
- 동시에 `exact_topk.npy` 와의 일치율로 **Recall@K vs exact** 도 계산
  (ANN-isolation 메트릭).

### 4.3 Latency profiling

`01_metric_design.md §3.2` 의 프로토콜 그대로:

- **Single-stream**: warmup 1000, measure 5000, 직렬, `cudaStreamSynchronize`.
  → P50/P95/P99 latency.
- **Max-throughput**: warmup 1000, measure ≥30초 또는 ≥10000 queries.
  concurrency ∈ {1, 4, 16, 64}.
  → QPS, mean latency.

```python
# 의사 코드
for concurrency in [1, 4, 16, 64]:
    warmup(retriever, queries, n=1000, batch=concurrency)
    start_power_sampler(retriever.device)
    t0 = time.monotonic()
    n_done = 0
    while time.monotonic() - t0 < 30 or n_done < 10_000:
        ids, scores = retriever.search(batch_of_queries(concurrency), k=10)
        n_done += concurrency
    elapsed = time.monotonic() - t0
    power_trace = stop_power_sampler()
    record(concurrency, n_done / elapsed, p50, p95, p99, power_trace)
```

### 4.4 Power sampling

- `pynvml.nvmlDeviceGetPowerUsage`, sample interval 100 ms.
- baseline idle: 측정 시작 5초 전, GPU idle 시 평균.
- 결과: 평균 net power, peak power, integration energy.

### 4.5 Hardware 캡처 (재현용)

각 실행 시 다음을 `results/<exp_id>/hardware.json` 에 기록:

```json
{
  "gpu": {
    "name": "NVIDIA H100 80GB HBM3",
    "uuid": "GPU-XXXXX",
    "driver": "550.54.15",
    "cuda": "12.4",
    "vbios": "96.00.74.00.1f",
    "smi_dump": "<output of nvidia-smi -q>"
  },
  "cpu": {
    "model": "AMD EPYC 7763 64-Core",
    "cores": 128,
    "threads": 256,
    "cpuinfo_dump": "<output of lscpu>"
  },
  "os": {"kernel": "5.15.0-1083-nvidia", "distro": "Ubuntu 22.04"},
  "captured_at": "2026-05-21T13:02:18+09:00"
}
```

---

## 5. 실험 변수와 그 의미

| 변수 | 값 (Phase 1) | 변경 시 가설 |
|---|---|---|
| Dataset | ML-25M, Amazon-Beauty | corpus size 가 작으면 retriever 간 차이가 흐려질 것 |
| Retriever | FAISS-CPU HNSW, cuVS IVF-PQ, cuVS CAGRA | retriever 별로 Pareto frontier 위치 다를 것 |
| Hardware | 1× H100 SXM5 80G + 2× EPYC 7763 | (sweep 제외) |
| k | 10, 100 | k 가 클수록 모든 retriever 의 recall 상승, QPS 감소 |
| Concurrency | 1, 4, 16, 64 | GPU 는 high concurrency 에서 유리 |
| Dim | 128 | (Phase 1 고정) |

---

## 6. 결과 산출물 매핑

| 측정 | 파일 | 작성자 |
|---|---|---|
| Per-run raw | `results/<exp_id>/{config.yaml, metrics.json, latency.csv, build.json, hardware.json, power_trace.csv}` | `pipelines/evaluate.py` |
| Aggregate | `results/aggregate.csv` | `pipelines/report.py` |
| Markdown | `reports/baseline_results.md` | `pipelines/report.py` |
| Recall-QPS plot | `reports/figures/recall_vs_qps.png` | `pipelines/report.py` |
| Latency CDF | `reports/figures/latency_cdf_<dataset>.png` | `pipelines/report.py` |
| Cost bar | `reports/figures/cost_bar.png` | `pipelines/report.py` |

---

## 7. VDPU 통합 (Phase 2) 예고

본 문서의 모든 절차는 `Retriever` 의 인터페이스 계약 (build / search /
save / load / device_info) 위에서만 동작한다. VDPU 를 위해 추가될 코드는:

1. `reco_bench/retrievers/vdpu.py` — `Retriever` 의 한 서브클래스.
2. `configs/retrievers/vdpu.yaml` — VDPU 별 param grid.
3. `configs/cost_model.yaml` — `vdpu_*` SKU 행 추가.

본 문서의 §3.2 의 grid 와 평행하는 grid 가 VDPU 에서도 정의되며, 동일
파이프라인이 그대로 실행된다.

---

## 8. 변경 이력

| 날짜 | 변경 | 근거 |
|---|---|---|
| 2026-05-20 | 초안 작성 | Phase 0 초기 설계 |
