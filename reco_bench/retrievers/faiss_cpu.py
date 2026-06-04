"""FAISS-CPU HNSW retriever.

설계: reports/03_baseline_methodology.md §3, configs/retrievers/faiss_hnsw_cpu.yaml.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BuildStats, Retriever

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]


class FaissHnswCpu(Retriever):
    name = "faiss_hnsw_cpu"
    device = "cpu"

    def __init__(self) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu not installed. `pip install faiss-cpu`")
        self._index = None
        self._item_ids: np.ndarray | None = None
        self._ef_search: int = 64

    def build(self, item_emb: np.ndarray, item_ids: np.ndarray, cfg: dict[str, Any]) -> BuildStats:
        import psutil

        proc = psutil.Process()
        m_before = proc.memory_info().rss

        n, d = item_emb.shape
        M = int(cfg.get("M", 32))
        ef_construction = int(cfg.get("efConstruction", 200))
        metric = cfg.get("metric", "inner_product")
        faiss_metric = faiss.METRIC_INNER_PRODUCT if metric == "inner_product" else faiss.METRIC_L2

        t0 = time.monotonic()
        index = faiss.IndexHNSWFlat(d, M, faiss_metric)
        index.hnsw.efConstruction = ef_construction
        index.add(item_emb.astype(np.float32))
        elapsed = time.monotonic() - t0

        self._index = index
        self._item_ids = item_ids.astype(np.int64)

        peak_host = (proc.memory_info().rss - m_before) / 1024**2
        return BuildStats(
            wall_seconds=elapsed,
            peak_host_mb=max(peak_host, 0.0),
            peak_device_mb=0.0,
            index_disk_bytes=0,             # save() 후에 갱신
            num_items=n,
            dim=d,
            extra={"M": M, "efConstruction": ef_construction},
        )

    def set_search_param(self, ef_search: int) -> None:
        self._ef_search = ef_search
        if self._index is not None:
            self._index.hnsw.efSearch = ef_search

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._index is not None
        self._index.hnsw.efSearch = self._ef_search
        D, I = self._index.search(queries.astype(np.float32), k)
        # I 는 row index 형태이므로 item_ids 로 매핑
        ids = self._item_ids[I.clip(min=0)]
        # FAISS 가 -1 을 padding 으로 줄 때 (k > n) 처리
        mask = I < 0
        ids[mask] = -1
        return ids.astype(np.int64), D.astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        np.save(path / "item_ids.npy", self._item_ids)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._index = faiss.read_index(str(path / "index.faiss"))
        self._item_ids = np.load(path / "item_ids.npy")

    def device_info(self) -> dict[str, Any]:
        return {"device": "cpu", "name": "faiss_hnsw_cpu", "faiss": "1.14.x"}
