"""Phase 2 scaling / sensitivity sweep.

synthetic embedding 으로 corpus 크기와 dim 을 통제 변수로 sweep 하여
각 retriever 의 Recall(vs exact)-QPS 거동을 측정한다. VDPU 없이도
"corpus 가 커질수록 GPU 가속의 가치가 커진다" 는 영업 곡선을 입증.

CLI:
  python -m reco_bench.pipelines.scaling --mode corpus
  python -m reco_bench.pipelines.scaling --mode dim

설계: reports/07_scaling.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..data.synthetic import make_clustered_embeddings, make_queries
from ..eval.metrics import recall_at_k_vs_exact
from ..eval.profiler import measure_max_throughput
from ..retrievers.cuvs_cagra import CuvsCagra
from ..retrievers.cuvs_ivfpq import CuvsIvfPq
from ..retrievers.exact_topk import ExactTopK
from ..retrievers.faiss_cpu import FaissHnswCpu
from ..utils.io import dump_json, ensure_dir

K = 10

# retriever 별 대표 grid (recall-QPS Pareto 의 몇 점). 측정 시간 절약 위해 축소.
RETRIEVERS = {
    "faiss_hnsw_cpu": {
        "cls": FaissHnswCpu,
        "build": {"M": 32, "efConstruction": 200, "metric": "inner_product"},
        "param_setter": "set_search_param",
        "grid": [64, 256],          # efSearch
    },
    "cuvs_cagra": {
        "cls": CuvsCagra,
        "build": {"graph_degree": 64, "intermediate_graph_degree": 96, "metric": "inner_product"},
        "param_setter": "set_search_param",
        "grid": [64, 256],          # itopk
    },
    "cuvs_ivfpq": {
        "cls": CuvsIvfPq,
        "build": {"n_lists": 4096, "pq_dim": 16, "pq_bits": 8, "metric": "inner_product"},
        "param_setter": "set_search_param",
        "grid": [16, 64],           # n_probes
    },
}

CONCURRENCIES = [1, 16, 64]


def _measure_one(name: str, spec: dict, item_emb, item_ids, queries, exact_ids) -> list[dict]:
    rows = []
    retr = spec["cls"]()
    t0 = time.monotonic()
    stats = retr.build(item_emb, item_ids, spec["build"])
    build_s = time.monotonic() - t0
    for p in spec["grid"]:
        getattr(retr, spec["param_setter"])(p)
        ann_ids, _ = retr.search(queries, k=K)
        recall = recall_at_k_vs_exact(ann_ids, exact_ids, K)
        # max-throughput
        best_qps = 0.0
        for c in CONCURRENCIES:
            rep = measure_max_throughput(retr, queries, k=K, concurrency=c,
                                         warmup_queries=100, min_seconds=1.5, min_queries=1000)
            best_qps = max(best_qps, rep.qps)
        rows.append({
            "retriever": name, "device": retr.device, "param": p,
            "recall_vs_exact@10": recall, "qps_max": best_qps,
            "build_seconds": build_s,
            "peak_device_mb": stats.peak_device_mb, "peak_host_mb": stats.peak_host_mb,
            "num_items": int(item_emb.shape[0]), "dim": int(item_emb.shape[1]),
        })
        print(f"      {name} p={p}: recall={recall:.4f} qps={best_qps:.0f}")
    return rows


def run(mode: str, out_dir: Path) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(out_dir)

    if mode == "corpus":
        settings = [(n, 128) for n in (100_000, 1_000_000, 10_000_000)]
    elif mode == "dim":
        settings = [(1_000_000, d) for d in (64, 128, 256, 512)]
    else:
        raise ValueError(mode)

    all_rows: list[dict] = []
    for n_items, dim in settings:
        print(f"[scaling:{mode}] corpus={n_items:,} dim={dim} 생성 중...")
        item_emb = make_clustered_embeddings(n_items, dim, seed=42)
        item_ids = np.arange(n_items, dtype=np.int64)
        queries = make_queries(item_emb, n_queries=1000, seed=123)

        ex = ExactTopK(device=device)
        ex.build(item_emb, item_ids, {})
        t0 = time.monotonic()
        exact_ids, _ = ex.search(queries, k=K)
        print(f"    exact top-{K}: {time.monotonic()-t0:.1f}s")
        del ex

        for name, spec in RETRIEVERS.items():
            try:
                rows = _measure_one(name, spec, item_emb, item_ids, queries, exact_ids)
                for r in rows:
                    r["mode"] = mode
                all_rows.extend(rows)
            except Exception as e:  # noqa: BLE001
                print(f"      [skip] {name}: {e}")
        del item_emb, queries

    dump_json({"mode": mode, "rows": all_rows}, out_dir / f"scaling_{mode}.json")
    print(f"\n[scaling:{mode}] {len(all_rows)} rows → {out_dir}/scaling_{mode}.json")
    return {"rows": all_rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["corpus", "dim"], required=True)
    p.add_argument("--out-dir", default="results/phase2_scaling")
    args = p.parse_args()
    run(args.mode, Path(args.out_dir))


if __name__ == "__main__":
    main()
