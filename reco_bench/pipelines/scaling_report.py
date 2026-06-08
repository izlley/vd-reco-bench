"""Phase 2 scaling/sensitivity 결과 → 그래프 + markdown.

results/phase2_scaling/scaling_{corpus,dim}.json 을 읽어
reports/figures/ 에 그래프를, reports/07_scaling.md 에 표를 생성.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
FIGS = ROOT / "reports" / "figures"
RES = ROOT / "results" / "phase2_scaling"

COLORS = {
    "faiss_hnsw_cpu": "#d62728",
    "cuvs_cagra": "#2ca02c",
    "cuvs_ivfpq": "#1f77b4",
}
LABEL = {
    "faiss_hnsw_cpu": "FAISS-CPU HNSW",
    "cuvs_cagra": "cuVS CAGRA (GPU)",
    "cuvs_ivfpq": "cuVS IVF-PQ (GPU)",
}


def _best_per_x(rows, xkey):
    """retriever × x 별로 recall 가장 높은 param 의 (x, qps, recall) 선택."""
    best = {}
    for r in rows:
        key = (r["retriever"], r[xkey])
        if key not in best or r["recall_vs_exact@10"] > best[key]["recall_vs_exact@10"]:
            best[key] = r
    series = defaultdict(list)
    for (rtr, x), r in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        series[rtr].append((x, r["qps_max"], r["recall_vs_exact@10"]))
    return series


def plot_corpus(rows: list[dict]) -> None:
    series = _best_per_x(rows, "num_items")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for rtr, pts in series.items():
        xs = [p[0] for p in pts]
        qps = [p[1] for p in pts]
        rec = [p[2] for p in pts]
        ax1.plot(xs, qps, "o-", color=COLORS.get(rtr), label=LABEL.get(rtr, rtr))
        ax2.plot(xs, rec, "o-", color=COLORS.get(rtr), label=LABEL.get(rtr, rtr))
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("Corpus size (items, log)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    ax1.set_yscale("log")
    ax1.set_ylabel("QPS (max concurrency, log)")
    ax1.set_title("Throughput vs corpus size")
    ax2.set_ylabel("Recall@10 vs exact")
    ax2.set_title("Recall vs corpus size (best param)")
    fig.suptitle("Phase 2 — Corpus scaling (synthetic, dim=128)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "scaling_corpus.png", dpi=110)
    plt.close(fig)


def plot_dim(rows: list[dict]) -> None:
    series = _best_per_x(rows, "dim")
    fig, ax = plt.subplots(figsize=(7, 5))
    for rtr, pts in series.items():
        xs = [p[0] for p in pts]
        qps = [p[1] for p in pts]
        ax.plot(xs, qps, "o-", color=COLORS.get(rtr), label=LABEL.get(rtr, rtr))
    ax.set_xlabel("Embedding dim")
    ax.set_ylabel("QPS (max concurrency, log)")
    ax.set_yscale("log")
    ax.set_title("Phase 2 — Throughput vs embedding dim\n(synthetic, corpus=1M)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "scaling_dim.png", dpi=110)
    plt.close(fig)


def _table(rows, xkey, xlabel):
    series = _best_per_x(rows, xkey)
    lines = [f"| Retriever | Device | {xlabel} | Recall@10 vs exact | QPS (max) |",
             "|---|---|---|---|---|"]
    dev = {r["retriever"]: r["device"] for r in rows}
    for rtr, pts in series.items():
        for x, qps, rec in pts:
            xs = f"{x:,}" if xkey == "num_items" else str(x)
            lines.append(f"| {LABEL.get(rtr,rtr)} | {dev.get(rtr)} | {xs} | {rec:.4f} | {qps:,.0f} |")
    return "\n".join(lines)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((RES / "scaling_corpus.json").read_text())["rows"] if (RES / "scaling_corpus.json").exists() else []
    dim = json.loads((RES / "scaling_dim.json").read_text())["rows"] if (RES / "scaling_dim.json").exists() else []
    if corpus:
        plot_corpus(corpus)
        print("wrote scaling_corpus.png")
    if dim:
        plot_dim(dim)
        print("wrote scaling_dim.png")
    print("\n--- corpus table ---")
    if corpus:
        print(_table(corpus, "num_items", "Corpus"))
    print("\n--- dim table ---")
    if dim:
        print(_table(dim, "dim", "Dim"))


if __name__ == "__main__":
    main()
