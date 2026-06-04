"""Qdrant server-mode retriever (gRPC client/server).

in-process(QdrantLocal) 와 달리 별도 Qdrant 서버 프로세스에 gRPC 로
연결한다. production 환경의 client/server 측정 (network 경유) 을 대표.
서버는 standalone 바이너리 또는 Docker 로 띄우면 된다 — 본 retriever 는
host/port 만 알면 됨.

설계: reports/history/2026-06-04_vectordb-server-eval.md.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BuildStats, Retriever


class QdrantServer(Retriever):
    name = "qdrant_server"
    device = "cpu"  # 서버 측 hardware. cost model 매칭은 CPU SKU.

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self._qm = qmodels
        self._QdrantClient = QdrantClient
        self._client = None
        self._collection = "reco_bench_server"
        self._item_ids: np.ndarray | None = None
        self._search_hnsw_ef = 64
        self._host = "localhost"
        self._grpc_port = 6334

    def _connect(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        dep = cfg.get("deployment", {})
        self._host = dep.get("client_host", "localhost")
        self._grpc_port = int(dep.get("grpc_port", 6334))
        # gRPC 사용 (HTTP 보다 throughput 높음)
        self._client = self._QdrantClient(
            host=self._host,
            grpc_port=self._grpc_port,
            prefer_grpc=True,
            check_compatibility=False,  # client 1.18 / server 1.12 minor 차이 허용
        )

    def build(self, item_emb: np.ndarray, item_ids: np.ndarray, cfg: dict[str, Any]) -> BuildStats:
        qm = self._qm
        self._collection = cfg.get("collection_name", "reco_bench_server")
        vc = cfg.get("vectors_config", {})
        distance_name = vc.get("distance", "Dot")
        hnsw = vc.get("hnsw_config", {})
        m = int(hnsw.get("m", 32))
        ef_construct = int(hnsw.get("ef_construct", 200))

        if self._client is None:
            self._connect(cfg)

        distance = {
            "Cosine": qm.Distance.COSINE,
            "Dot": qm.Distance.DOT,
            "Euclid": qm.Distance.EUCLID,
        }.get(distance_name, qm.Distance.DOT)

        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection in existing:
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=int(item_emb.shape[1]), distance=distance),
            hnsw_config=qm.HnswConfigDiff(m=m, ef_construct=ef_construct),
        )

        t0 = time.monotonic()
        bs = 2048
        for s in range(0, len(item_ids), bs):
            e = min(s + bs, len(item_ids))
            self._client.upsert(
                collection_name=self._collection,
                points=qm.Batch(
                    ids=[int(item_ids[i]) for i in range(s, e)],
                    vectors=[item_emb[i].astype(np.float32).tolist() for i in range(s, e)],
                ),
                wait=True,
            )
        elapsed = time.monotonic() - t0

        self._item_ids = item_ids.astype(np.int64)
        return BuildStats(
            wall_seconds=elapsed,
            peak_host_mb=0.0,
            peak_device_mb=0.0,
            index_disk_bytes=0,  # 서버 측 — 직접 측정 불가
            num_items=int(item_emb.shape[0]),
            dim=int(item_emb.shape[1]),
            extra={"m": m, "ef_construct": ef_construct, "transport": "grpc"},
        )

    def set_search_param(self, ef: int) -> None:
        self._search_hnsw_ef = int(ef)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        qm = self._qm
        b = len(queries)
        ids = np.full((b, k), -1, dtype=np.int64)
        scores = np.zeros((b, k), dtype=np.float32)
        sp = qm.SearchParams(hnsw_ef=self._search_hnsw_ef)
        for i in range(b):
            res = self._client.query_points(
                collection_name=self._collection,
                query=queries[i].astype(np.float32).tolist(),
                limit=k,
                search_params=sp,
            ).points
            for j, hit in enumerate(res[:k]):
                ids[i, j] = int(hit.id)
                scores[i, j] = float(hit.score)
        return ids, scores

    def save(self, path: str | Path) -> None:
        # 서버가 storage 를 보유 → client 측은 item_ids 와 연결 정보만 저장.
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "item_ids.npy", self._item_ids)
        (path / "conn.txt").write_text(f"{self._host}:{self._grpc_port}:{self._collection}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._item_ids = np.load(path / "item_ids.npy")
        conn = (path / "conn.txt")
        if conn.exists():
            host, port, coll = conn.read_text().strip().split(":")
            self._host, self._grpc_port, self._collection = host, int(port), coll
        if self._client is None:
            self._connect()

    def device_info(self) -> dict[str, Any]:
        import qdrant_client

        return {
            "device": "cpu",
            "name": "qdrant_server",
            "transport": "grpc",
            "qdrant_client": getattr(qdrant_client, "__version__", "unknown"),
        }
