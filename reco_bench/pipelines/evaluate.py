"""평가 파이프라인 — Quality + System + Cost.

CLI: ``python -m reco_bench.pipelines.evaluate <experiment_yaml>``
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.base import build_ground_truth
from ..data.pipeline import load_train_positives
from ..eval.cost import cost_for_run
from ..eval.metrics import compute_all
from ..eval.profiler import (
    PowerSampler,
    measure_max_throughput,
    measure_single_stream,
)
from ..models.two_tower import TwoTower
from ..retrievers.exact_topk import ExactTopK
from ..retrievers.registry import load_retriever
from ..utils.io import dump_json, ensure_dir, load_yaml
from ..utils.seed import set_global_seed


def _load_model(num_users: int, num_items: int, model_cfg: dict, ckpt_path: Path, device: str):
    arch = model_cfg["architecture"]
    model = TwoTower(
        num_users=num_users,
        num_items=num_items,
        embed_dim_id=arch["embed_dim_id"],
        embed_dim_out=arch["embed_dim_out"],
        mlp_hidden=arch["mlp_hidden"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def _compute_query_set(
    model: TwoTower,
    test_df: pd.DataFrame,
    train_positives: dict,
    recent_n: int,
    device: str,
    max_users: int = 5000,
) -> tuple[np.ndarray, np.ndarray, dict[int, set[int]]]:
    """test user embedding + ground truth 생성."""
    test_users = test_df["user_id"].unique()
    rng = np.random.RandomState(42)
    if len(test_users) > max_users:
        sel = rng.choice(test_users, size=max_users, replace=False)
    else:
        sel = test_users

    user_ids = torch.tensor(sel, dtype=torch.long, device=device)
    recent = np.full((len(sel), recent_n), -1, dtype=np.int64)
    lengths = np.zeros(len(sel), dtype=np.int64)
    for i, u in enumerate(sel.tolist()):
        pos = train_positives.get(int(u))
        if pos is not None and len(pos) > 0:
            n = min(len(pos), recent_n)
            recent[i, :n] = pos[-n:]
            lengths[i] = n
    recent_t = torch.tensor(recent, device=device)
    lens_t = torch.tensor(lengths, device=device)
    u_emb = model.encode_user(user_ids, recent_t, lens_t).cpu().numpy().astype(np.float32)

    gt = build_ground_truth(test_df, users=sel)
    return sel.astype(np.int64), u_emb, gt


def evaluate_all(config_path: str | Path) -> dict:
    cfg = load_yaml(config_path)
    set_global_seed()
    exp_id = cfg["experiment_id"]
    out_root = ensure_dir(Path(cfg.get("reporting", {}).get("output_dir", f"results/{exp_id}")))
    cost_model = load_yaml(cfg["cost_model"])
    model_cfg = load_yaml(cfg["model"])
    recent_n = int(model_cfg["architecture"]["recent_n"])
    ks = list(cfg["evaluation"]["ks"])
    max_k = max(ks)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: list[dict] = []
    for ds_cfg_path in cfg["datasets"]:
        ds_cfg = load_yaml(ds_cfg_path)
        ds_name = ds_cfg["name"]
        ckpt_dir = Path("checkpoints") / exp_id / ds_name
        if not (ckpt_dir / "item_embeddings.npy").exists():
            print(f"[eval] skip {ds_name}: no checkpoint.")
            continue
        proc_dir = Path(ds_cfg["processed_dir"])
        test_df = pd.read_parquet(proc_dir / "test.parquet")
        train_positives = load_train_positives(proc_dir / "train_positives.npz")
        import json

        stats = json.loads((proc_dir / "stats.json").read_text())

        model = _load_model(stats["num_users"], stats["num_items"], model_cfg, ckpt_dir / "model.pt", device)
        user_ids, query_emb, gt = _compute_query_set(
            model, test_df, train_positives, recent_n, device
        )
        print(f"[eval] {ds_name}: queries={len(user_ids):,}, ground_truth_users={len(gt):,}")

        # exact top-K (ANN isolation 기준)
        item_emb = np.load(ckpt_dir / "item_embeddings.npy")
        item_ids = np.load(ckpt_dir / "item_ids.npy")
        ex = ExactTopK(device="cuda" if device == "cuda" else "cpu")
        ex.build(item_emb, item_ids, {})
        exact_ids, _ = ex.search(query_emb, k=max_k)

        # 각 retriever × param grid
        for r_cfg_path in cfg["retrievers"]:
            r_cfg = load_yaml(r_cfg_path)
            if r_cfg.get("status") == "not_implemented_yet":
                continue
            try:
                retr = load_retriever(r_cfg)
            except ImportError as e:
                print(f"  [skip] {r_cfg['name']}: {e}")
                continue
            print(f"  retriever={r_cfg['name']}")

            # 인덱스 load
            idx_dir = Path("indexes") / r_cfg["name"] / ds_name
            try:
                retr.load(idx_dir)
            except Exception as e:
                print(f"    [warn] load failed ({e}); rebuilding")
                retr.build(item_emb, item_ids, r_cfg["build"])

            # search grid 파라미터 추출
            grid_axes = _extract_search_grid(r_cfg)
            # power baseline 은 retriever 당 한 번만 (전체 grid 공유) — 성능 최적화
            _baseline_w_for_retriever = None
            for grid_point in grid_axes:
                _set_search_param(retr, grid_point)
                # quality
                ann_ids, _ = retr.search(query_emb, k=max_k)
                metrics = compute_all(user_ids, ann_ids, gt, exact_topk=exact_ids, ks=tuple(ks))

                # latency (single-stream)
                lat_ss = measure_single_stream(
                    retr, query_emb, k=max(ks), warmup_queries=200, measure_queries=min(2000, len(query_emb))
                )
                # latency (max throughput at concurrency=64)
                concurrencies = list(cfg["evaluation"]["latency"].get("concurrencies", [1, 4, 16, 64]))
                # power sampling
                power_sampler = PowerSampler(
                    sample_interval_ms=int(cfg["evaluation"]["power"]["sample_interval_ms"]),
                    device_index=0,
                )
                if _baseline_w_for_retriever is None:
                    _baseline_w_for_retriever = (
                        power_sampler.baseline(
                            seconds=float(cfg["evaluation"]["power"]["baseline_idle_seconds"])
                        )
                        if retr.device == "cuda"
                        else 0.0
                    )
                baseline_w = _baseline_w_for_retriever
                power_sampler.start()
                lat_mt: dict[int, dict] = {}
                for c in concurrencies:
                    rep = measure_max_throughput(
                        retr,
                        query_emb,
                        k=max(ks),
                        concurrency=c,
                        warmup_queries=200,
                        min_seconds=2.0,           # Phase 1 첫 검증은 짧게
                        min_queries=2000,
                    )
                    lat_mt[c] = asdict(rep)
                _, _ = power_sampler.stop()
                power = power_sampler.report(baseline_w=baseline_w)

                # cost (concurrency=max 기준의 QPS 사용)
                max_qps = max(lat_mt[c]["qps"] for c in concurrencies)
                cost = cost_for_run(
                    r_cfg["name"],
                    retr.device,
                    qps=max_qps,
                    mean_power_w=max(power.mean_power_w - power.baseline_idle_w, 0.0),
                    cost_model=cost_model,
                )

                row = {
                    "experiment_id": exp_id,
                    "dataset": ds_name,
                    "retriever": r_cfg["name"],
                    "device": retr.device,
                    "grid": grid_point,
                    "metrics": metrics,
                    "latency_single_stream": asdict(lat_ss),
                    "latency_max_throughput": lat_mt,
                    "power": asdict(power),
                    "cost": asdict(cost),
                }
                rows.append(row)
                print(
                    f"    grid={grid_point} recall@10={metrics['recall@10']:.4f} "
                    f"recall_vs_exact@10={metrics.get('recall_vs_exact@10', 0):.4f} "
                    f"qps_max={max_qps:.0f} $/1M=${cost.usd_per_1m_queries:.4f}"
                )

    dump_json({"experiment_id": exp_id, "rows": rows}, out_root / "metrics.json")
    # 단순 CSV 도 생성
    _dump_csv(rows, out_root / "aggregate.csv")
    return {"rows": rows}


def _extract_search_grid(r_cfg: dict) -> list[dict]:
    """retriever YAML 에서 search grid 추출.

    Phase 1 의 첫 검증은 sweep 차원 1개만 사용 (efSearch / n_probes / itopk).
    """
    s = r_cfg["search"]
    if "efSearch_grid" in s:
        return [{"efSearch": v} for v in s["efSearch_grid"]]
    if "n_probes_grid" in s:
        return [{"n_probes": v} for v in s["n_probes_grid"]]
    if "itopk_grid" in s:
        return [{"itopk_size": v} for v in s["itopk_grid"]]
    if "hnsw_ef_grid" in s:
        return [{"hnsw_ef": v} for v in s["hnsw_ef_grid"]]
    if "ef_grid" in s:
        return [{"ef": v} for v in s["ef_grid"]]
    if "leaves_to_search_grid" in s:
        return [{"leaves_to_search": v} for v in s["leaves_to_search_grid"]]
    return [{}]


def _set_search_param(retriever, grid_point: dict) -> None:
    if not grid_point:
        return
    if hasattr(retriever, "set_search_param"):
        # 첫 값을 사용
        v = next(iter(grid_point.values()))
        retriever.set_search_param(v)


def _dump_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    import csv

    flat: list[dict] = []
    for r in rows:
        for k_val in r["metrics"]:
            pass
        grid = r["grid"]
        grid_str = ",".join(f"{k}={v}" for k, v in grid.items())
        max_c = max(r["latency_max_throughput"].keys())
        rec = {
            "dataset": r["dataset"],
            "retriever": r["retriever"],
            "device": r["device"],
            "grid": grid_str,
        }
        for k, v in r["metrics"].items():
            rec[k] = v
        rec["p50_ms"] = r["latency_single_stream"]["p50_ms"]
        rec["p95_ms"] = r["latency_single_stream"]["p95_ms"]
        rec["p99_ms"] = r["latency_single_stream"]["p99_ms"]
        rec[f"qps_c{max_c}"] = r["latency_max_throughput"][max_c]["qps"]
        rec["mean_power_w"] = r["power"]["mean_power_w"]
        rec["baseline_idle_w"] = r["power"]["baseline_idle_w"]
        rec["usd_per_1m"] = r["cost"]["usd_per_1m_queries"]
        flat.append(rec)
    keys = sorted({k for r in flat for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in flat:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    args = p.parse_args()
    evaluate_all(args.config)


if __name__ == "__main__":
    main()
