"""results/<exp_id>/metrics.json 을 집계해 reports/baseline_results.md 와
reports/figures/*.png 을 자동 생성.

설계: reports/03_baseline_methodology.md §6, reports/00_overview.md §4.
사용자 요구사항 (5종 그래프): reports/history/2026-06-04_user-requirements-clarified.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..utils.io import ensure_dir, load_yaml


def _load_rows(metrics_path: Path) -> list[dict]:
    with open(metrics_path) as f:
        data = json.load(f)
    return data.get("rows", [])


def _pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """점 (recall, qps) 의 Pareto frontier — recall 큰 + qps 큰 우상단."""
    sorted_pts = sorted(points, key=lambda p: (-p[0], -p[1]))
    out: list[tuple[float, float]] = []
    best_qps = -1
    for r, q in sorted(sorted_pts, key=lambda p: p[0]):
        if q >= best_qps:
            out.append((r, q))
            best_qps = q
    return out


def plot_recall_vs_qps(rows: list[dict], outpath: Path) -> None:
    """Recall@K vs exact 기준 (ANN-isolation) vs QPS."""
    fig, ax = plt.subplots(figsize=(8, 6))
    by_combo: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        ds = r["dataset"]
        rtr = r["retriever"]
        key = f"{rtr} [{r['device']}] · {ds}"
        recall = r["metrics"].get("recall_vs_exact@10", r["metrics"].get("recall@10", 0.0))
        max_c = max(int(c) for c in r["latency_max_throughput"])
        qps = r["latency_max_throughput"][str(max_c)]["qps"] if str(max_c) in r["latency_max_throughput"] else r["latency_max_throughput"][max_c]["qps"]
        by_combo.setdefault(key, []).append((recall, qps))

    cmap = plt.get_cmap("tab10")
    for i, (label, pts) in enumerate(sorted(by_combo.items())):
        xs = [p[1] for p in pts]
        ys = [p[0] for p in pts]
        ax.scatter(xs, ys, label=label, color=cmap(i % 10), s=42, alpha=0.8)
        front = _pareto_frontier(pts)
        if len(front) > 1:
            front.sort(key=lambda p: p[1])
            ax.plot([p[1] for p in front], [p[0] for p in front], color=cmap(i % 10), linestyle="--", linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("QPS (log scale)")
    ax.set_ylabel("Recall@10 vs exact (ANN-isolation)")
    ax.set_title("Recall–QPS Pareto curve")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.axhline(y=0.95, color="r", linestyle=":", linewidth=1, alpha=0.6, label="iso-recall=0.95")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_latency_cdf(rows: list[dict], outpath: Path) -> None:
    """Latency CDF (single-stream samples). retriever 별 색."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    plotted = set()
    for i, r in enumerate(rows):
        key = f"{r['retriever']} [{r['device']}]"
        if key in plotted:
            continue
        samples = r["latency_single_stream"].get("samples_ms", [])
        if not samples:
            continue
        s = np.sort(np.array(samples))
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, cdf, label=key, color=cmap(len(plotted) % 10))
        plotted.add(key)
    ax.set_xscale("log")
    ax.set_xlabel("single-stream latency (ms, log)")
    ax.set_ylabel("CDF")
    ax.set_title("Single-stream latency distribution")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_cost_bar(rows: list[dict], outpath: Path) -> None:
    """retriever 별 최저 $/1M (가장 효율적인 grid 점) 막대."""
    by_rtr: dict[str, float] = {}
    for r in rows:
        # iso-recall 의 ideal 은 0.95+ 의 grid 점 중 최저 cost
        if r["metrics"].get("recall_vs_exact@10", 0) < 0.90:
            continue
        cost = r["cost"]["usd_per_1m_queries"]
        if r["retriever"] not in by_rtr or by_rtr[r["retriever"]] > cost:
            by_rtr[r["retriever"]] = cost

    fig, ax = plt.subplots(figsize=(7, 5))
    if by_rtr:
        keys = sorted(by_rtr.keys(), key=lambda k: by_rtr[k])
        vals = [by_rtr[k] for k in keys]
        ax.barh(keys, vals, color="steelblue")
        for i, v in enumerate(vals):
            ax.text(v, i, f"  ${v:.4f}", va="center", fontsize=9)
        ax.set_xlabel("$ / 1M queries  (at Recall@10 vs exact ≥ 0.90)")
        ax.set_title("Cost efficiency at iso-recall (lower is better)")
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No retriever reached Recall@10 vs exact ≥ 0.90", ha="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_concurrency_qps(rows: list[dict], outpath: Path) -> None:
    """concurrency vs QPS — retriever 의 batch sweet-spot 시각화."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    plotted = set()
    for r in rows:
        key = f"{r['retriever']} [{r['device']}]"
        if key in plotted:
            continue
        cc = r["latency_max_throughput"]
        # JSON 에서 key 가 str
        xs = sorted(int(c) for c in cc.keys())
        ys = [cc[str(c) if str(c) in cc else c]["qps"] for c in xs]
        ax.plot(xs, ys, marker="o", label=key, color=cmap(len(plotted) % 10))
        plotted.add(key)
    ax.set_xscale("log")
    ax.set_xlabel("Concurrency (batch size)")
    ax.set_ylabel("QPS")
    ax.set_title("Throughput vs concurrency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_throughput_per_watt(rows: list[dict], outpath: Path) -> None:
    """retriever × dataset 별 QPS/W."""
    fig, ax = plt.subplots(figsize=(7, 5))
    labels: list[str] = []
    vals: list[float] = []
    for r in rows:
        if r["metrics"].get("recall_vs_exact@10", 0) < 0.90:
            continue
        watts_per_qps = r["cost"].get("watts_per_qps", 0)
        if watts_per_qps <= 0 or not np.isfinite(watts_per_qps):
            continue
        qps_per_w = 1.0 / watts_per_qps
        labels.append(f"{r['retriever']} · {r['dataset']}")
        vals.append(qps_per_w)
    if not labels:
        ax.text(0.5, 0.5, "No power data (sampler unavailable or all CPU)", ha="center")
        ax.set_axis_off()
    else:
        order = np.argsort(vals)[::-1][:12]
        labels = [labels[i] for i in order]
        vals = [vals[i] for i in order]
        ax.barh(labels[::-1], vals[::-1], color="darkgreen")
        ax.set_xlabel("QPS / Watt  (higher is better)")
        ax.set_title("Energy efficiency at iso-recall")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def _speedup_table(rows: list[dict], target_recall: float = 0.95) -> list[str]:
    """iso-recall (>=target) 에서 retriever 별 최고 QPS 와 그 기준 speedup ratio.

    사용자 요구: "vector DB 별 retrieval 속도 향상" 의 직접 비교 지표.
    """
    by_combo: dict[tuple[str, str], float] = {}  # (dataset, retriever) → best QPS
    for r in rows:
        if r["metrics"].get("recall_vs_exact@10", 0) < target_recall:
            continue
        cc = r["latency_max_throughput"]
        max_c = max(int(c) for c in cc.keys())
        qps = cc[str(max_c) if str(max_c) in cc else max_c]["qps"]
        key = (r["dataset"], r["retriever"])
        if key not in by_combo or by_combo[key] < qps:
            by_combo[key] = qps

    # dataset 별 baseline = 가장 느린 retriever (FAISS-CPU HNSW 가 일반적으로 강력하므로
    # speedup base 가 1 이 되도록 가장 낮은 QPS 사용)
    out_lines = [
        f"## Iso-recall speedup (Recall@10 vs exact ≥ {target_recall:.2f})",
        "",
        "각 dataset 안에서 가장 느린 retriever 를 1× 로 두고, 다른 retriever 의 QPS 배수.",
        "",
        "| Dataset | Retriever | Device | QPS | Speedup vs slowest |",
        "|---|---|---|---|---|",
    ]
    by_ds: dict[str, list[tuple[str, str, float]]] = {}
    for r in rows:
        if r["metrics"].get("recall_vs_exact@10", 0) < target_recall:
            continue
        cc = r["latency_max_throughput"]
        max_c = max(int(c) for c in cc.keys())
        qps = cc[str(max_c) if str(max_c) in cc else max_c]["qps"]
        key = (r["dataset"], r["retriever"])
        if key in by_combo and by_combo[key] == qps:
            by_ds.setdefault(r["dataset"], []).append((r["retriever"], r["device"], qps))

    for ds, items in sorted(by_ds.items()):
        items.sort(key=lambda x: x[2])  # 느린 순
        baseline = items[0][2] if items else 1.0
        for rtr, dev, qps in items:
            mul = qps / baseline if baseline > 0 else 0
            out_lines.append(
                f"| {ds} | {rtr} | {dev} | {qps:,.0f} | {mul:.2f}× |"
            )
    return out_lines


def generate_markdown(rows: list[dict], outpath: Path, figs_dir: Path) -> None:
    """baseline_results.md 자동 생성."""
    # 가장 좋은 (recall_vs_exact@10 가장 큰, QPS 가장 큰) 점 per retriever × dataset
    table_lines = [
        "| Dataset | Retriever | Device | Recall@10 (vs exact) | Recall@10 (vs GT) | NDCG@10 | QPS (max c) | P99 ms | $/1M | W/QPS |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    by_combo: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["dataset"], r["retriever"])
        if key not in by_combo:
            by_combo[key] = r
        else:
            prev = by_combo[key]
            if r["metrics"].get("recall_vs_exact@10", 0) > prev["metrics"].get("recall_vs_exact@10", 0):
                by_combo[key] = r
            elif r["metrics"].get("recall_vs_exact@10", 0) == prev["metrics"].get("recall_vs_exact@10", 0):
                # 동일 recall 이면 더 빠른 것
                cc_keys_r = [int(c) for c in r["latency_max_throughput"].keys()]
                cc_keys_p = [int(c) for c in prev["latency_max_throughput"].keys()]
                qps_r = max(r["latency_max_throughput"][str(c) if str(c) in r["latency_max_throughput"] else c]["qps"] for c in cc_keys_r)
                qps_p = max(prev["latency_max_throughput"][str(c) if str(c) in prev["latency_max_throughput"] else c]["qps"] for c in cc_keys_p)
                if qps_r > qps_p:
                    by_combo[key] = r

    for (ds, rtr), r in sorted(by_combo.items()):
        m = r["metrics"]
        cc = r["latency_max_throughput"]
        max_c = max(int(c) for c in cc.keys())
        qps = cc[str(max_c) if str(max_c) in cc else max_c]["qps"]
        p99 = r["latency_single_stream"]["p99_ms"]
        cost = r["cost"]["usd_per_1m_queries"]
        w_per_qps = r["cost"].get("watts_per_qps", float("inf"))
        w_str = f"{w_per_qps:.2f}" if np.isfinite(w_per_qps) else "—"
        table_lines.append(
            f"| {ds} | {rtr} | {r['device']} | "
            f"{m.get('recall_vs_exact@10', 0):.4f} | "
            f"{m.get('recall@10', 0):.4f} | "
            f"{m.get('ndcg@10', 0):.4f} | "
            f"{qps:,.0f} | "
            f"{p99:.2f} | "
            f"${cost:.4f} | "
            f"{w_str} |"
        )

    speedup_lines = _speedup_table(rows, target_recall=0.95)
    speedup_block = "\n".join(speedup_lines)
    md = f"""# baseline_results.md — 자동 생성

> 이 파일은 `scripts/99_make_report.sh` 가 자동 생성합니다. 직접 편집
> 하지 마세요. 메트릭 정의는 [`01_metric_design.md`](01_metric_design.md),
> 방법론은 [`03_baseline_methodology.md`](03_baseline_methodology.md) 참조.

## 메인 결과 표

각 (dataset, retriever) 조합에서 **Recall@10 vs exact 가 가장 높은 grid
점** 의 결과 (동일 recall 이면 QPS 가 더 큰 쪽 선택).

{chr(10).join(table_lines)}

{speedup_block}

## 시각화

### Recall-QPS Pareto 곡선

![](figures/recall_vs_qps.png)

곡선의 점선은 retriever × dataset 별 Pareto frontier. 빨간 점선은
iso-recall = 0.95 비교 기준선 (`reports/01_metric_design.md §4.1`).

### Single-stream latency 분포

![](figures/latency_cdf.png)

### Cost @ iso-recall (≥0.90)

![](figures/cost_bar.png)

### Throughput vs concurrency

![](figures/concurrency_qps.png)

### Throughput per Watt @ iso-recall (≥0.90)

![](figures/throughput_per_watt.png)

전력 측정이 가능한 경우에만 표시. CPU 만 사용한 retriever 는 보통
NVML 샘플링 결과가 0 이라 제외된다.

## 데이터 소스

- 메트릭: `results/<exp_id>/metrics.json`
- 집계 CSV: `results/<exp_id>/aggregate.csv`
- 하드웨어: `results/<exp_id>/hardware.json`
- 비용 모델: `configs/cost_model.yaml`

## 주의 (현재 단계 한계)

본 자동 생성 결과는 ML-25M sanity check 단계의 측정이다. 본 표의 절대
값보다 **retriever × hardware 간 상대 비교** 에 의미를 둔다. 그래도
`Recall@10 vs ground truth` 가 낮다면 모델 학습이 부족한 신호이며,
`Recall@10 vs exact` 는 ANN 알고리즘 자체의 정확도라 모델 품질과 독립.
"""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(md, encoding="utf-8")


def make_report(config_path: str | Path) -> None:
    cfg = load_yaml(config_path)
    exp_id = cfg["experiment_id"]
    res_dir = Path(cfg.get("reporting", {}).get("output_dir", f"results/{exp_id}"))
    md_path = Path(cfg.get("reporting", {}).get("generate_markdown", "reports/baseline_results.md"))
    figs_dir = ensure_dir(cfg.get("reporting", {}).get("generate_figures", "reports/figures/"))

    metrics_path = res_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"[report] {metrics_path} not found. Run scripts/30_run_benchmark.sh first.")
        return
    rows = _load_rows(metrics_path)
    if not rows:
        print(f"[report] no rows in {metrics_path}")
        return

    print(f"[report] generating from {metrics_path} ({len(rows)} rows)")
    plot_recall_vs_qps(rows, figs_dir / "recall_vs_qps.png")
    plot_latency_cdf(rows, figs_dir / "latency_cdf.png")
    plot_cost_bar(rows, figs_dir / "cost_bar.png")
    plot_concurrency_qps(rows, figs_dir / "concurrency_qps.png")
    plot_throughput_per_watt(rows, figs_dir / "throughput_per_watt.png")
    generate_markdown(rows, md_path, figs_dir)
    print(f"[report] wrote {md_path} + 5 figures in {figs_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    args = p.parse_args()
    make_report(args.config)


if __name__ == "__main__":
    main()
