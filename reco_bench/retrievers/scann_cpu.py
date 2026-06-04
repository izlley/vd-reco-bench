"""Google ScaNN (CPU) retriever.

설계: reports/03_baseline_methodology.md §3, configs/retrievers/scann_cpu.yaml.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BuildStats, Retriever


class ScannCpu(Retriever):
    name = "scann_cpu"
    device = "cpu"

    def __init__(self) -> None:
        try:
            import scann
        except ImportError as e:
            raise ImportError(
                "scann not installed or not importable on this platform "
                "(requires x86_64 + AVX2). `pip install scann`"
            ) from e
        self._scann = scann
        self._searcher = None
        self._item_ids: np.ndarray | None = None
        self._tmpdir: Path | None = None
        self._search_param: dict[str, int] = {}

    def build(self, item_emb: np.ndarray, item_ids: np.ndarray, cfg: dict[str, Any]) -> BuildStats:
        import tempfile

        scann = self._scann
        x = item_emb.astype(np.float32)

        num_leaves = int(cfg.get("num_leaves", 2000))
        num_leaves_search_default = int(cfg.get("num_leaves_to_search_default", 100))
        reorder = int(cfg.get("reordering_num_neighbors", 500))
        anisotropic = float(cfg.get("anisotropic_quantization_threshold", 0.2))

        builder = (
            scann.scann_ops_pybind.builder(x, 10, "dot_product")
            .tree(
                num_leaves=min(num_leaves, max(1, x.shape[0] // 4)),
                num_leaves_to_search=num_leaves_search_default,
                training_sample_size=min(x.shape[0], 100_000),
            )
            .score_ah(2, anisotropic_quantization_threshold=anisotropic)
            .reorder(reorder)
        )
        t0 = time.monotonic()
        self._searcher = builder.build()
        elapsed = time.monotonic() - t0
        self._item_ids = item_ids.astype(np.int64)

        # ScaNN searcher 는 디스크 직렬화 가능
        self._tmpdir = Path(tempfile.mkdtemp(prefix="scann_"))
        self._searcher.serialize(str(self._tmpdir))
        disk_b = sum(p.stat().st_size for p in self._tmpdir.rglob("*") if p.is_file())

        return BuildStats(
            wall_seconds=elapsed,
            peak_host_mb=0.0,
            peak_device_mb=0.0,
            index_disk_bytes=disk_b,
            num_items=int(x.shape[0]),
            dim=int(x.shape[1]),
            extra={"num_leaves": num_leaves, "reorder": reorder},
        )

    def set_search_param(self, leaves_to_search: int) -> None:
        self._search_param = {"leaves_to_search": int(leaves_to_search)}

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._searcher is not None
        leaves = int(self._search_param.get("leaves_to_search", 100))
        # search_batched: (queries, final_num_neighbors, leaves_to_search, pre_reorder_num_neighbors)
        idx, dist = self._searcher.search_batched(
            queries.astype(np.float32),
            final_num_neighbors=k,
            leaves_to_search=leaves,
        )
        # idx: (B, k) of int32
        ids = self._item_ids[idx.astype(np.int64)]
        return ids, dist.astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._tmpdir and self._tmpdir.exists():
            target = path / "scann"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(self._tmpdir, target)
        np.save(path / "item_ids.npy", self._item_ids)

    def load(self, path: str | Path) -> None:
        import scann

        path = Path(path)
        self._searcher = scann.scann_ops_pybind.load_searcher(str(path / "scann"))
        self._item_ids = np.load(path / "item_ids.npy")

    def device_info(self) -> dict[str, Any]:
        return {"device": "cpu", "name": "scann_cpu"}
