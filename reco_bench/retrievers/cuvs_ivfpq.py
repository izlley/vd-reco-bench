"""cuVS IVF-PQ retriever (GPU).

설계: reports/03_baseline_methodology.md §3, configs/retrievers/cuvs_ivfpq.yaml.
FAISS-GPU 대신 cuVS 의 IVF-PQ 를 사용 (cublas ABI 충돌 회피).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BuildStats, Retriever


class CuvsIvfPq(Retriever):
    name = "cuvs_ivfpq"
    device = "cuda"

    def __init__(self) -> None:
        import cupy as cp
        from cuvs.neighbors import ivf_pq

        self._cp = cp
        self._ivf_pq = ivf_pq
        self._index = None
        self._item_ids: np.ndarray | None = None
        self._search_params: dict[str, Any] = {}

    def build(self, item_emb: np.ndarray, item_ids: np.ndarray, cfg: dict[str, Any]) -> BuildStats:
        cp = self._cp
        ivf_pq = self._ivf_pq

        data = cp.asarray(item_emb.astype(np.float32))

        n_lists = int(cfg.get("n_lists", 4096))
        pq_dim = int(cfg.get("pq_dim", 16))
        pq_bits = int(cfg.get("pq_bits", 8))
        metric = cfg.get("metric", "inner_product")

        build_params = ivf_pq.IndexParams(
            n_lists=n_lists,
            metric=metric,
            pq_dim=pq_dim,
            pq_bits=pq_bits,
            kmeans_n_iters=int(cfg.get("kmeans_n_iters", 20)),
            kmeans_trainset_fraction=float(cfg.get("kmeans_trainset_fraction", 0.5)),
            add_data_on_build=bool(cfg.get("add_data_on_build", True)),
        )

        mempool_before = cp.get_default_memory_pool().used_bytes()
        t0 = time.monotonic()
        index = ivf_pq.build(build_params, data)
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
            extra={"n_lists": n_lists, "pq_dim": pq_dim, "pq_bits": pq_bits},
        )

    def set_search_param(self, n_probes: int) -> None:
        self._search_params = {"n_probes": int(n_probes)}

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        cp = self._cp
        ivf_pq = self._ivf_pq

        q = cp.asarray(queries.astype(np.float32))
        search_params = ivf_pq.SearchParams(
            n_probes=int(self._search_params.get("n_probes", 16))
        )
        distances, indices = ivf_pq.search(search_params, self._index, q, k)
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
        # cuVS index 는 직접 save API 가 24.10+ 부터 있으나 호환성 위해 generic pickle
        # 시도가 실패하면 빌드를 다시 하는 것을 권장 (in-RAM only mode).
        try:
            from cuvs.neighbors import ivf_pq

            ivf_pq.save(str(path / "index.cuvs"), self._index)
        except Exception:
            with open(path / "index.pkl", "wb") as f:
                pickle.dump(self._index, f)
        np.save(path / "item_ids.npy", self._item_ids)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        try:
            from cuvs.neighbors import ivf_pq

            self._index = ivf_pq.load(str(path / "index.cuvs"))
        except Exception:
            with open(path / "index.pkl", "rb") as f:
                self._index = pickle.load(f)
        self._item_ids = np.load(path / "item_ids.npy")

    def device_info(self) -> dict[str, Any]:
        import cuvs

        return {"device": "cuda", "name": "cuvs_ivfpq", "cuvs": cuvs.__version__}
