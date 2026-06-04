"""Dataset abstract base + canonical schema.

설계: reports/02_dataset_selection.md §3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetStats:
    """전처리 후 데이터셋 statistics. ``tests/test_data.py`` 의 sanity
    check 와 ``reports/02_dataset_selection.md §4`` 가 본 dataclass 와
    매칭된다."""

    name: str
    num_users: int
    num_items: int
    num_interactions: int
    num_train: int
    num_val: int
    num_test: int
    split_policy: str
    positive_threshold: float
    kcore: int


class RecoDataset(ABC):
    """추천 데이터셋의 추상 기본 클래스.

    canonical schema (모든 데이터셋이 산출해야 하는 형태):

    - ``interactions.parquet``: ``[user_id int64, item_id int64,
      ts int64, rating float32]``. user_id / item_id 는 [0, N) 로
      remap. ts 는 unix epoch seconds.
    - ``item_meta.parquet``: ``[item_id int64, title str, category str,
      text str]``.

    구체 클래스는 ``download()`` 와 ``_load_raw()`` 만 구현하면 된다.
    """

    name: str = "abstract"

    def __init__(self, cfg: dict, root: str | Path = "data") -> None:
        self.cfg = cfg
        self.root = Path(root)
        self.raw_dir = Path(cfg["source"]["raw_dir"])
        self.processed_dir = Path(cfg["processed_dir"])

    @abstractmethod
    def download(self) -> None:
        """원본 데이터를 ``self.raw_dir`` 로 가져온다."""

    @abstractmethod
    def _load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """원본 파일을 ``(interactions_df, item_meta_df)`` 로 정규화.

        반환 시 컬럼은 canonical 이름이어야 한다:
        - interactions: ``user_id, item_id, ts, rating`` (raw IDs, no remap)
        - item_meta: ``item_id, title, category, text`` (raw IDs)
        """

    # ---- Helpers (서브클래스가 override 할 필요 없음) ----

    def load_processed(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """이미 전처리된 parquet 을 로드."""
        inters = pd.read_parquet(self.processed_dir / "interactions.parquet")
        items = pd.read_parquet(self.processed_dir / "item_meta.parquet")
        return inters, items

    def is_processed(self) -> bool:
        return (self.processed_dir / "interactions.parquet").exists() and (
            self.processed_dir / "item_meta.parquet"
        ).exists()

    def load_split(self) -> dict[str, pd.DataFrame]:
        """전처리된 train/val/test split 로드."""
        out = {}
        for s in ("train", "val", "test"):
            p = self.processed_dir / f"{s}.parquet"
            if not p.exists():
                raise FileNotFoundError(
                    f"split {s} not found at {p}. Run preprocessing first."
                )
            out[s] = pd.read_parquet(p)
        return out


def build_ground_truth(
    test_df: pd.DataFrame, users: np.ndarray | None = None
) -> dict[int, set[int]]:
    """test split 의 정답 item 집합 만들기.

    ``reports/01_metric_design.md §2.1`` 의 ``G_u``.
    """
    gt: dict[int, set[int]] = {}
    if users is not None:
        users_set = set(int(u) for u in users)
    else:
        users_set = None
    for u, g in test_df.groupby("user_id")["item_id"]:
        ui = int(u)
        if users_set is not None and ui not in users_set:
            continue
        gt[ui] = set(int(x) for x in g.values)
    return gt
