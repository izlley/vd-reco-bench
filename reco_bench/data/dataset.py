"""학습용 PyTorch Dataset / Sampler.

설계: reports/03_baseline_methodology.md §2.
- 한 row = (user_id, item_id, recent_item_ids[max recent_n], recent_length)
- recent_items 는 train 시 해당 row 의 ts 이전의 positive 만 사용해야
  하지만, 본 구현은 단순화를 위해 전체 train positive 의 최근 N 개를
  사용한다 (학습 stability 우선; eval 은 별도로 leak-free).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionsDataset(Dataset):
    """``(user_id, item_id, recent_items, recent_length)`` 를 산출.

    recent_items 는 ``train_positives.npz`` 에 미리 계산된 user 별
    positive 리스트의 최근 N 개 (padding 은 ``-1``).
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        train_positives: dict[int, np.ndarray],
        recent_n: int = 50,
    ) -> None:
        self.users = interactions["user_id"].to_numpy(dtype=np.int64)
        self.items = interactions["item_id"].to_numpy(dtype=np.int64)
        self.recent_n = recent_n
        # padded numpy lookup: user → (N,) array of recent item ids (or -1 pad)
        # 메모리 절약을 위해 dict 그대로 두고 __getitem__ 에서 변환
        self.train_positives = train_positives

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[int, int, np.ndarray, int]:
        u = int(self.users[idx])
        v = int(self.items[idx])
        pos = self.train_positives.get(u, np.array([], dtype=np.int64))
        n = min(len(pos), self.recent_n)
        recent = np.full(self.recent_n, -1, dtype=np.int64)
        if n > 0:
            recent[:n] = pos[-n:]
        return u, v, recent, n


def collate_batch(batch):
    """DataLoader collate_fn."""
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    items = torch.tensor([b[1] for b in batch], dtype=torch.long)
    recent = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.long)
    lengths = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return users, items, recent, lengths
