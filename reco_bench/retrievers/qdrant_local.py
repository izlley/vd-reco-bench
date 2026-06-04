"""Qdrant — 오픈소스 vector DB 의 in-process (local file) 구현체.

qdrant-client 의 ``QdrantClient(path="<dir>")`` 로 in-process 모드 사용.
서버 없이 동작 — fair 한 in-process 비교.
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .base import BuildStats, Retriever


class QdrantLocal(Retriever):
    name = "qdrant_local"
    device = "cpu"

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self._QdrantClient = QdrantClient
        self._qm = qmodels
        self._client = None
        self._collection_name = "reco_bench"
        self._item_ids: np.ndarray | None = None
        self._db_dir: Path | None = None
        self._search_hnsw_ef: int = 64

    def build(self, item_emb: np.ndarray, item_ids: np.ndarray, cfg: dict[str, Any]) -> BuildStats:
        import tempfile

        self._collection_name = cfg.get("collection_name", "reco_bench")
        vc = cfg.get("vectors_config", {})
        distance_name = vc.get("distance", "Cosine")
        hnsw_cfg = vc.get("hnsw_config", {})
        m = int(hnsw_cfg.get("m", 32))
        ef_construct = int(hnsw_cfg.get("ef_construct", 200))

        self._db_dir = Path(tempfile.mkdtemp(prefix="qdrant_local_"))
        self._client = self._QdrantClient(path=str(self._db_dir))

        # recreate
        if self._collection_name in [c.name for c in self._client.get_collections().collections]:
            self._client.delete_collection(self._collection_name)

        distance = {
            "Cosine": self._qm.Distance.COSINE,
            "Dot": self._qm.Distance.DOT,
            "Euclid": self._qm.Distance.EUCLID,
        }.get(distance_name, self._qm.Distance.DOT)

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=self._qm.VectorParams(
                size=int(item_emb.shape[1]),
                distance=distance,
            ),
            hnsw_config=self._qm.HnswConfigDiff(m=m, ef_construct=ef_construct),
        )

        t0 = time.monotonic()
        bs = 4096
        for s in range(0, len(item_ids), bs):
            e = min(s + bs, len(item_ids))
            self._client.upsert(
                collection_name=self._collection_name,
                points=self._qm.Batch(
                    ids=[int(item_ids[i]) for i in range(s, e)],
                    vectors=[item_emb[i].astype(np.float32).tolist() for i in range(s, e)],
                ),
            )
        elapsed = time.monotonic() - t0

        self._item_ids = item_ids.astype(np.int64)
        return BuildStats(
            wall_seconds=elapsed,
            peak_host_mb=0.0,
            peak_device_mb=0.0,
            index_disk_bytes=sum(
                p.stat().st_size for p in self._db_dir.rglob("*") if p.is_file()
            ),
            num_items=int(item_emb.shape[0]),
            dim=int(item_emb.shape[1]),
            extra={"m": m, "ef_construct": ef_construct},
        )

    def set_search_param(self, ef: int) -> None:
        self._search_hnsw_ef = int(ef)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._client is not None
        b = len(queries)
        ids = np.full((b, k), -1, dtype=np.int64)
        scores = np.zeros((b, k), dtype=np.float32)
        sp = self._qm.SearchParams(hnsw_ef=self._search_hnsw_ef)
        # 새 qdrant-client API: query_points 한 줄씩. (이전 search_batch 는 deprecated.)
        for i in range(b):
            try:
                res = self._client.query_points(
                    collection_name=self._collection_name,
                    query=queries[i].astype(np.float32).tolist(),
                    limit=k,
                    search_params=sp,
                ).points
            except AttributeError:
                # 더 옛 버전 fallback
                res = self._client.search(
                    collection_name=self._collection_name,
                    query_vector=queries[i].astype(np.float32).tolist(),
                    limit=k,
                    search_params=sp,
                )
            for j, hit in enumerate(res[:k]):
                ids[i, j] = int(hit.id)
                scores[i, j] = float(hit.score)
        return ids, scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._db_dir and self._db_dir.exists():
            target = path / "qdrant_db"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(self._db_dir, target)
        np.save(path / "item_ids.npy", self._item_ids)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        target = path / "qdrant_db"
        if target.exists():
            self._db_dir = target
            self._client = self._QdrantClient(path=str(self._db_dir))
        self._item_ids = np.load(path / "item_ids.npy")

    def device_info(self) -> dict[str, Any]:
        import qdrant_client

        return {
            "device": "cpu",
            "name": "qdrant_local",
            "qdrant_client": getattr(qdrant_client, "__version__", "unknown"),
        }
