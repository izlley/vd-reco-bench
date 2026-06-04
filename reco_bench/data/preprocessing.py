"""전처리 공통 단계: implicit conversion, k-core, ID remap, split.

설계: reports/02_dataset_selection.md §3.3, §3.4, §3.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implicit_filter(
    df: pd.DataFrame, threshold: float, rating_col: str = "rating"
) -> pd.DataFrame:
    """rating >= threshold 인 row 만 남긴다 (implicit positive 변환).

    ``reports/02_dataset_selection.md §3.3`` 의 cutoff 정책.
    """
    if rating_col not in df.columns:
        return df.reset_index(drop=True)
    mask = df[rating_col].astype(float) >= float(threshold)
    return df[mask].reset_index(drop=True)


def kcore_filter(
    df: pd.DataFrame,
    k: int = 5,
    user_col: str = "user_id",
    item_col: str = "item_id",
    max_iters: int = 30,
) -> pd.DataFrame:
    """k-core filtering. 수렴 시 종료.

    ``reports/02_dataset_selection.md §3.4``.
    """
    cur = df
    for _ in range(max_iters):
        prev_len = len(cur)
        u_counts = cur.groupby(user_col).size()
        keep_users = set(u_counts[u_counts >= k].index)
        cur = cur[cur[user_col].isin(keep_users)]

        i_counts = cur.groupby(item_col).size()
        keep_items = set(i_counts[i_counts >= k].index)
        cur = cur[cur[item_col].isin(keep_items)]

        if len(cur) == prev_len:
            break
    return cur.reset_index(drop=True)


def remap_ids(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """user/item ID 를 [0, N) 의 dense int 로 remap.

    Returns:
        (remapped_df, {"users": {raw_id: dense_id}, "items": {raw_id: dense_id}})
    """
    users = sorted(df[user_col].unique())
    items = sorted(df[item_col].unique())
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {it: i for i, it in enumerate(items)}

    out = df.copy()
    out[user_col] = out[user_col].map(u_map).astype("int64")
    out[item_col] = out[item_col].map(i_map).astype("int64")
    return out, {"users": u_map, "items": i_map}


def temporal_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.90, 0.05, 0.05),
    ts_col: str = "ts",
) -> dict[str, pd.DataFrame]:
    """timestamp 정렬 후 비율로 잘라 train/val/test 분할.

    reports/02_dataset_selection.md §3.5 의 기본 split policy.
    """
    s = float(sum(ratios))
    ratios = tuple(r / s for r in ratios)  # 정규화

    df_sorted = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    return {
        "train": df_sorted.iloc[:n_train].reset_index(drop=True),
        "val": df_sorted.iloc[n_train : n_train + n_val].reset_index(drop=True),
        "test": df_sorted.iloc[n_train + n_val :].reset_index(drop=True),
    }


def stats_of(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> dict[str, int]:
    return {
        "num_users": int(df[user_col].nunique()),
        "num_items": int(df[item_col].nunique()),
        "num_interactions": int(len(df)),
    }


def validate_split(splits: dict[str, pd.DataFrame]) -> None:
    """split sanity check (no time leak, no row leak).

    ``reports/02_dataset_selection.md §4`` 의 검증 항목.
    """
    train_max = splits["train"]["ts"].max()
    val_min = splits["val"]["ts"].min() if len(splits["val"]) else float("inf")
    val_max = splits["val"]["ts"].max() if len(splits["val"]) else float("-inf")
    test_min = splits["test"]["ts"].min() if len(splits["test"]) else float("inf")

    if val_min != float("inf") and train_max > val_min:
        raise AssertionError(
            f"Temporal split leak: train_max={train_max} > val_min={val_min}"
        )
    if test_min != float("inf") and val_max > test_min:
        raise AssertionError(
            f"Temporal split leak: val_max={val_max} > test_min={test_min}"
        )

    # row 단위 중복 없음 확인 (그룹별 cutoff 라 자연스러우나 안전 차원)
    pair = lambda d: set(zip(d["user_id"].tolist(), d["item_id"].tolist(), d["ts"].tolist()))
    train_pairs = pair(splits["train"])
    test_pairs = pair(splits["test"])
    if train_pairs & test_pairs:
        raise AssertionError("train/test set has overlapping (user, item, ts) triples")


def positives_by_user(
    train_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    ts_col: str = "ts",
    recent_n: int | None = None,
) -> dict[int, np.ndarray]:
    """user 별 train positive item id 리스트 (최근 N 개로 truncate 가능).

    user tower 의 recent-N mean-pool 입력에 사용 (reports/03_baseline_methodology.md §1.2).
    """
    out: dict[int, np.ndarray] = {}
    for u, sub in train_df.sort_values(ts_col).groupby(user_col):
        arr = sub[item_col].to_numpy(dtype=np.int64)
        if recent_n is not None and len(arr) > recent_n:
            arr = arr[-recent_n:]
        out[int(u)] = arr
    return out
