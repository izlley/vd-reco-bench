"""ANN index 빌드 파이프라인.

체크포인트의 ``item_embeddings.npy`` 를 읽어 각 retriever 의 인덱스를
빌드하고 ``indexes/<retriever>/<dataset>/`` 에 직렬화.

CLI: ``python -m reco_bench.pipelines.build_index <experiment_yaml>``
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..retrievers.registry import load_retriever
from ..utils.io import dump_json, ensure_dir, load_yaml
from ..utils.seed import set_global_seed


def build_all(config_path: str | Path) -> dict:
    cfg = load_yaml(config_path)
    set_global_seed()
    exp_id = cfg["experiment_id"]
    out_root = Path(cfg.get("reporting", {}).get("output_dir", f"results/{exp_id}"))
    ensure_dir(out_root)

    summary: dict = {"experiment_id": exp_id, "builds": []}

    for ds_cfg_path in cfg["datasets"]:
        ds_cfg = load_yaml(ds_cfg_path)
        ds_name = ds_cfg["name"]
        ckpt_dir = Path("checkpoints") / exp_id / ds_name
        emb_path = ckpt_dir / "item_embeddings.npy"
        ids_path = ckpt_dir / "item_ids.npy"
        if not emb_path.exists():
            print(f"[build] skip {ds_name}: {emb_path} not found.")
            continue
        item_emb = np.load(emb_path)
        item_ids = np.load(ids_path)
        print(f"[build] {ds_name}: items={item_emb.shape[0]:,}, dim={item_emb.shape[1]}")

        for r_cfg_path in cfg["retrievers"]:
            r_cfg = load_yaml(r_cfg_path)
            # vector DB stub (status: not_implemented_yet) 는 skip
            if r_cfg.get("status") == "not_implemented_yet":
                print(f"  [skip] {r_cfg['name']}: not implemented yet (Phase 1.5+).")
                continue
            print(f"  [build] retriever={r_cfg['name']}")
            try:
                retr = load_retriever(r_cfg)
            except ImportError as e:
                print(f"    [skip] {r_cfg['name']}: import failed ({e}).")
                continue
            # client/server retriever (Milvus, Qdrant server) 는 단일 서버에
            # collection 으로 공존하므로 dataset 별로 고유 이름이 필요하다.
            # (in-process 파일 기반 retriever 는 별도 디렉토리라 무관하지만
            #  collection_name 에 dataset 을 붙여도 안전하다.)
            build_cfg = dict(r_cfg["build"])
            if "collection_name" in build_cfg:
                build_cfg["collection_name"] = f"{build_cfg['collection_name']}_{ds_name}"
            stats = retr.build(item_emb, item_ids, build_cfg)
            idx_dir = ensure_dir(Path("indexes") / r_cfg["name"] / ds_name)
            retr.save(idx_dir)
            # 디스크 크기 갱신
            disk_bytes = sum(p.stat().st_size for p in idx_dir.rglob("*") if p.is_file())
            row = {
                "retriever": r_cfg["name"],
                "dataset": ds_name,
                "build": {**asdict(stats), "index_disk_bytes": disk_bytes},
                "device": retr.device_info(),
            }
            summary["builds"].append(row)
            print(
                f"    wall={stats.wall_seconds:.2f}s, device_peak={stats.peak_device_mb:.1f}MB, "
                f"disk={disk_bytes/1024**2:.1f}MB"
            )

    dump_json(summary, out_root / "build_summary.json")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    args = p.parse_args()
    build_all(args.config)


if __name__ == "__main__":
    main()
