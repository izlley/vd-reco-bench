"""Exact (brute-force) top-K retriever.

ANN-isolation 메트릭 (``recall_at_k_vs_exact``) 의 ground truth 생성에
사용. ``reports/01_metric_design.md §2.4`` 참조.

CPU 또는 GPU 모두 동작 가능. dim 128 / item 24k 면 GPU 한 번에 충분.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BuildStats, Retriever


class ExactTopK(Retriever):
    name = "exact_topk"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._item_emb: np.ndarray | None = None
        self._item_ids: np.ndarray | None = None

    def build(self, item_emb: np.ndarray, item_ids: np.ndarray, cfg: dict[str, Any]) -> BuildStats:
        t0 = time.monotonic()
        self._item_emb = item_emb.astype(np.float32)
        self._item_ids = item_ids.astype(np.int64)
        return BuildStats(
            wall_seconds=time.monotonic() - t0,
            peak_host_mb=self._item_emb.nbytes / 1024**2,
            peak_device_mb=0.0,
            index_disk_bytes=self._item_emb.nbytes + self._item_ids.nbytes,
            num_items=int(self._item_emb.shape[0]),
            dim=int(self._item_emb.shape[1]),
        )

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._item_emb is None:
            raise RuntimeError("build() not called")
        if self.device == "cuda":
            import torch

            q = torch.from_numpy(queries.astype(np.float32)).cuda()
            x = torch.from_numpy(self._item_emb).cuda()
            scores = q @ x.T
            top_scores, top_idx = scores.topk(k, dim=1)
            ids = self._item_ids[top_idx.cpu().numpy()]
            torch.cuda.synchronize()
            return ids.astype(np.int64), top_scores.cpu().numpy().astype(np.float32)
        else:
            scores = queries.astype(np.float32) @ self._item_emb.T
            top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            row = np.arange(top_idx.shape[0])[:, None]
            order = np.argsort(-scores[row, top_idx], axis=1)
            top_idx = top_idx[row, order]
            return (
                self._item_ids[top_idx].astype(np.int64),
                scores[row, top_idx].astype(np.float32),
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "item_emb.npy", self._item_emb)
        np.save(path / "item_ids.npy", self._item_ids)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._item_emb = np.load(path / "item_emb.npy")
        self._item_ids = np.load(path / "item_ids.npy")

    def device_info(self) -> dict[str, Any]:
        return {"device": self.device, "name": "exact_topk"}
