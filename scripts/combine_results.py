"""여러 results/<exp>/metrics.json 을 합쳐 results/phase1_combined/metrics.json 생성.

(dataset, retriever) 가 여러 src 에 중복되면 **나중 src 가 override** 한다.
→ CPU 128-thread 재측정(phase1_cpu_rethread)이 기존 CPU 값을 대체.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
srcs = sys.argv[1:] or [
    "results/phase1_baseline_v0/metrics.json",   # ml25m: faiss_hnsw_cpu, cuvs_ivfpq, cuvs_cagra
    "results/phase1_amazon/metrics.json",        # amazon: + scann_cpu
    "results/phase1_qdrant_server/metrics.json", # both: qdrant_server (gRPC)
    "results/phase1_faiss_gpu/metrics.json",     # both: faiss_ivfpq_gpu (conda env, H100 sm_90)
    "results/phase1_cpu_rethread/metrics.json",  # CPU 128-thread 재측정 (override, 마지막)
]

collected = []
for s in srcs:
    p = ROOT / s
    if not p.exists():
        print(f"  - {s}: (없음, skip)")
        continue
    src_rows = json.loads(p.read_text()).get("rows", [])
    collected.append(src_rows)
    print(f"  + {s}: {len(src_rows)} rows")

# (dataset, retriever) 별 마지막 등장 src 만 채택 (override)
last_idx = {}
for i, src_rows in enumerate(collected):
    for r in src_rows:
        last_idx[(r["dataset"], r["retriever"])] = i
rows = []
for i, src_rows in enumerate(collected):
    for r in src_rows:
        if last_idx[(r["dataset"], r["retriever"])] == i:
            rows.append(r)

out = ROOT / "results/phase1_combined"
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps({"experiment_id": "phase1_combined", "rows": rows}, indent=2, default=str))
print(f"\ncombined {len(rows)} rows → results/phase1_combined/metrics.json")
