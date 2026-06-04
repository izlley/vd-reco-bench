"""cuVS CAGRA retriever (GPU, graph-based).

설계: reports/03_baseline_methodology.md §3, configs/retrievers/cuvs_cagra.yaml.
VDPU 의 가장 강한 GPU 경쟁자.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BuildStats, Retriever


class CuvsCagra(Retriever):
    name = "cuvs_cagra"
    device = "cuda"

    def __init__(self) -> None:
        import cupy as cp
        from cuvs.neighbors import cagra

        self._cp = cp
        self._cagra = cagra
        self._index = None
        self._item_ids: np.ndarray | None = None
        self._search_params: dict[str, Any] = {}

    def build(self, item_emb: np.ndarray, item_ids: np.ndarray, cfg: dict[str, Any]) -> BuildStats:
        cp = self._cp
        cagra = self._cagra

        data = cp.asarray(item_emb.astype(np.float32))

        graph_degree = int(cfg.get("graph_degree", 64))
        intermediate = int(cfg.get("intermediate_graph_degree", 96))
        metric = cfg.get("metric", "inner_product")

        build_params = cagra.IndexParams(
            graph_degree=graph_degree,
            intermediate_graph_degree=intermediate,
            metric=metric,
        )

        mempool_before = cp.get_default_memory_pool().used_bytes()
        t0 = time.monotonic()
        index = cagra.build(build_params, data)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.monotonic() - t0
        peak_device = (cp.get_default_memory_pool().used_bytes() - mempool_before) / 1024**2

        self._index = index
        self._item_ids = item_ids.astype(np.int64)
        return BuildStats(
            wall_seconds=elapsed,
            peak_host_mb=0.0,
            peak_device_mb=max(peak_device, 0.0),
            index_disk_bytes=0,
            num_items=int(item_emb.shape[0]),
            dim=int(item_emb.shape[1]),
            extra={"graph_degree": graph_degree, "intermediate_graph_degree": intermediate},
        )

    def set_search_param(self, itopk_size: int) -> None:
        self._search_params = {"itopk_size": int(itopk_size)}

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        cp = self._cp
        cagra = self._cagra

        q = cp.asarray(queries.astype(np.float32))
        search_params = cagra.SearchParams(
            itopk_size=int(self._search_params.get("itopk_size", 64))
        )
        distances, indices = cagra.search(search_params, self._index, q, k)
        cp.cuda.runtime.deviceSynchronize()
        idx_np = cp.asnumpy(indices)
        dist_np = cp.asnumpy(distances)

        ids = self._item_ids[np.clip(idx_np, 0, len(self._item_ids) - 1)]
        ids = ids.astype(np.int64)
        ids[idx_np < 0] = -1
        return ids, dist_np.astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        try:
            from cuvs.neighbors import cagra

            cagra.save(str(path / "index.cagra"), self._index)
        except Exception:
            with open(path / "index.pkl", "wb") as f:
                pickle.dump(self._index, f)
        np.save(path / "item_ids.npy", self._item_ids)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        try:
            from cuvs.neighbors import cagra

            self._index = cagra.load(str(path / "index.cagra"))
        except Exception:
            with open(path / "index.pkl", "rb") as f:
                self._index = pickle.load(f)
        self._item_ids = np.load(path / "item_ids.npy")

    def device_info(self) -> dict[str, Any]:
        import cuvs

        return {"device": "cuda", "name": "cuvs_cagra", "cuvs": cuvs.__version__}
