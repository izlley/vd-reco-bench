"""데이터 다운로드 → 전처리 → 분할 → 디스크 직렬화 의 단일 entry.

``scripts/00_download_data.sh`` 가 본 모듈의 ``run_preprocessing`` 을 호출.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..utils.io import dump_json, load_yaml
from ..utils.seed import set_global_seed
from .base import DatasetStats, RecoDataset
from .preprocessing import (
    implicit_filter,
    kcore_filter,
    positives_by_user,
    remap_ids,
    stats_of,
    temporal_split,
    validate_split,
)

# 데이터셋 이름 → 클래스 매핑. 새 데이터셋 추가 시 여기와 import 만 갱신.
_DATASETS = {}


def _register(name: str, cls) -> None:
    _DATASETS[name] = cls


def get_dataset_cls(name: str):
    if not _DATASETS:
        from .ml25m import MovieLens25M
        from .amazon2023 import AmazonReviews2023

        _register("ml25m", MovieLens25M)
        # Amazon Reviews 2023 의 모든 카테고리는 같은 클래스를 공유 — 이름만 다름
        for cat_name in ("amazon_beauty", "amazon_books", "amazon_electronics"):
            _register(cat_name, AmazonReviews2023)
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(_DATASETS)}")
    return _DATASETS[name]


def run_preprocessing(config_path: str | Path, root: str | Path = ".") -> DatasetStats:
    """YAML 한 개를 받아 download → 전처리 → 분할 → 직렬화."""
    cfg = load_yaml(config_path)
    set_global_seed()

    cls = get_dataset_cls(cfg["name"])
    ds: RecoDataset = cls(cfg, root=root)

    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg['name']}] download → {ds.raw_dir}")
    ds.download()

    print(f"[{cfg['name']}] loading raw")
    inters, items = ds._load_raw()
    print(f"  raw interactions: {len(inters):,}, raw items: {len(items):,}")

    pp = cfg["preprocessing"]
    print(f"[{cfg['name']}] implicit filter @ rating >= {pp['implicit_positive_rating_threshold']}")
    inters = implicit_filter(inters, pp["implicit_positive_rating_threshold"])
    print(f"  remaining: {len(inters):,}")

    print(f"[{cfg['name']}] k-core filter @ k={pp['kcore_min_interactions']}")
    inters = kcore_filter(inters, k=pp["kcore_min_interactions"])
    print(f"  remaining: {len(inters):,}")

    print(f"[{cfg['name']}] ID remap")
    inters, id_map = remap_ids(inters)
    # item_meta 도 같은 map 으로 정렬 (학습에 등장하지 않는 item 은 제외)
    item_keep = set(id_map["items"].keys())
    items = items[items["item_id"].isin(item_keep)].copy()
    items["item_id"] = items["item_id"].map(id_map["items"]).astype("int64")
    items = items.sort_values("item_id").reset_index(drop=True)

    print(f"[{cfg['name']}] temporal split @ ratios={pp['split_ratios']}")
    splits = temporal_split(inters, ratios=tuple(pp["split_ratios"]))
    print(
        f"  train={len(splits['train']):,}, val={len(splits['val']):,}, test={len(splits['test']):,}"
    )
    validate_split(splits)

    # 직렬화
    print(f"[{cfg['name']}] writing parquet → {out_dir}")
    inters.to_parquet(out_dir / "interactions.parquet", index=False)
    items.to_parquet(out_dir / "item_meta.parquet", index=False)
    for s, df in splits.items():
        df.to_parquet(out_dir / f"{s}.parquet", index=False)

    # ID map 은 JSON 으로 (재현용)
    with open(out_dir / "id_map.json", "w") as f:
        json.dump(
            {
                "users": {str(k): int(v) for k, v in id_map["users"].items()},
                "items": {str(k): int(v) for k, v in id_map["items"].items()},
            },
            f,
        )

    # user 별 train positive 캐시 (학습 sampler / user-tower 입력)
    print(f"[{cfg['name']}] computing per-user train positives")
    pos = positives_by_user(splits["train"], recent_n=None)
    # 직렬화: numpy 압축
    import numpy as np

    np.savez_compressed(
        out_dir / "train_positives.npz",
        user_ids=np.array(sorted(pos.keys()), dtype=np.int64),
        lengths=np.array([len(pos[u]) for u in sorted(pos.keys())], dtype=np.int64),
        items=np.concatenate([pos[u] for u in sorted(pos.keys())]) if pos else np.array([], dtype=np.int64),
    )

    # stats
    s_all = stats_of(inters)
    stats = DatasetStats(
        name=cfg["name"],
        num_users=s_all["num_users"],
        num_items=s_all["num_items"],
        num_interactions=s_all["num_interactions"],
        num_train=len(splits["train"]),
        num_val=len(splits["val"]),
        num_test=len(splits["test"]),
        split_policy=pp["split_policy"],
        positive_threshold=pp["implicit_positive_rating_threshold"],
        kcore=pp["kcore_min_interactions"],
    )
    dump_json(
        {
            "name": stats.name,
            "num_users": stats.num_users,
            "num_items": stats.num_items,
            "num_interactions": stats.num_interactions,
            "num_train": stats.num_train,
            "num_val": stats.num_val,
            "num_test": stats.num_test,
            "split_policy": stats.split_policy,
            "positive_threshold": stats.positive_threshold,
            "kcore": stats.kcore,
        },
        out_dir / "stats.json",
    )
    # sanity check: 기대값 floor
    expected = cfg.get("stats_expected", {})
    for k, v in expected.items():
        actual = getattr(stats, k.replace("_min", ""), None)
        if actual is None or actual < v:
            print(f"  [warn] {k}: expected min {v:,}, got {actual:,}")

    print(
        f"[{cfg['name']}] done — users={stats.num_users:,}, items={stats.num_items:,}, "
        f"train={stats.num_train:,}, val={stats.num_val:,}, test={stats.num_test:,}"
    )
    return stats


def load_train_positives(path: str | Path) -> dict[int, "np.ndarray"]:  # noqa: F821
    """``train_positives.npz`` 를 dict 로 복원."""
    import numpy as np

    z = np.load(path, allow_pickle=False)
    users = z["user_ids"]
    lengths = z["lengths"]
    items = z["items"]
    out: dict[int, np.ndarray] = {}
    cur = 0
    for u, ln in zip(users.tolist(), lengths.tolist()):
        out[int(u)] = items[cur : cur + ln]
        cur += ln
    return out


def __main__() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to dataset YAML (configs/datasets/*.yaml)")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    run_preprocessing(args.config, root=args.root)


if __name__ == "__main__":
    __main__()
