"""Milvus Lite — 오픈소스 vector DB 의 in-process 구현체.

pymilvus 의 ``MilvusClient(uri="<file>.db")`` 로 in-process 모드 사용.
서버 시작 없이 동작 → benchmark 에서 fair 한 비교 가능.

설계: reports/history/2026-06-04_user-requirements-clarified.md (Phase 1.5).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BuildStats, Retriever


class MilvusLite(Retriever):
    name = "milvus_lite"
    device = "cpu"

    def __init__(self) -> None:
        from pymilvus import MilvusClient

        self._MilvusClient = MilvusClient
        self._client = None
        self._collection_name: str = "reco_bench"
        self._item_ids: np.ndarray | None = None
        self._db_path: Path | None = None
        self._search_params: dict[str, Any] = {}

    def build(self, item_emb: np.ndarray, item_ids: np.ndarray, cfg: dict[str, Any]) -> BuildStats:
        self._collection_name = cfg.get("collection_name", "reco_bench")
        index_params = cfg.get("index_params", {})
        # in-process db 파일 위치 — 인덱스 디렉토리 안에
        # save() 시 외부 path 로 복사할 수 있도록 lazy 처리

        # 임시 directory 생성
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="milvus_lite_"))
        self._db_path = tmp / "milvus.db"
        self._client = self._MilvusClient(uri=str(self._db_path))

        dim = int(item_emb.shape[1])
        # Drop existing if any
        if self._collection_name in self._client.list_collections():
            self._client.drop_collection(self._collection_name)

        # Milvus 3.0 의 quick setup mode 로 collection 생성
        self._client.create_collection(
            collection_name=self._collection_name,
            dimension=dim,
            metric_type=cfg.get("metric_type", "IP"),
        )

        # Insert
        records = [
            {"id": int(item_ids[i]), "vector": item_emb[i].astype(np.float32).tolist()}
            for i in range(len(item_ids))
        ]
        t0 = time.monotonic()
        # batch insert for memory
        bs = 4096
        for s in range(0, len(records), bs):
            self._client.insert(collection_name=self._collection_name, data=records[s : s + bs])
        # explicit index/load — Milvus Lite 의 default index 는 FLAT (small data 면 sufficient)
        # large data 라면 HNSW 로 변경 필요
        try:
            self._client.flush(collection_name=self._collection_name)
        except Exception:
            pass
        elapsed = time.monotonic() - t0
        self._item_ids = item_ids.astype(np.int64)

        if self._db_path.is_dir():
            disk_b = sum(p.stat().st_size for p in self._db_path.rglob("*") if p.is_file())
        elif self._db_path.exists():
            disk_b = int(self._db_path.stat().st_size)
        else:
            disk_b = 0
        return BuildStats(
            wall_seconds=elapsed,
            peak_host_mb=0.0,
            peak_device_mb=0.0,
            index_disk_bytes=disk_b,
            num_items=int(item_emb.shape[0]),
            dim=dim,
            extra={"index_type": "default", **index_params},
        )

    def set_search_param(self, ef: int) -> None:
        # Milvus Lite default index 는 FLAT → ef 는 무시되지만 호환성 위해 받기만
        self._search_params = {"ef": int(ef)}

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._client is not None
        # collection 이 released 상태일 수 있어 매 호출 ensure (idempotent, cheap)
        try:
            if not self._client.get_load_state(self._collection_name).get("state").name == "Loaded":
                self._client.load_collection(self._collection_name)
        except Exception:
            try:
                self._client.load_collection(self._collection_name)
            except Exception:
                pass
        results = self._client.search(
            collection_name=self._collection_name,
            data=queries.astype(np.float32).tolist(),
            limit=k,
            search_params={"params": self._search_params},
            output_fields=["id"],
        )
        # results: list[list[{id, distance, ...}]] of length B
        b = len(queries)
        ids = np.full((b, k), -1, dtype=np.int64)
        scores = np.zeros((b, k), dtype=np.float32)
        for i, row in enumerate(results):
            for j, hit in enumerate(row[:k]):
                ids[i, j] = int(hit.get("id", hit.get("entity", {}).get("id", -1)))
                scores[i, j] = float(hit.get("distance", 0.0))
        return ids, scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._db_path and self._db_path.exists():
            target = path / "milvus.db"
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if self._db_path.is_dir():
                shutil.copytree(self._db_path, target)
            else:
                shutil.copy(self._db_path, target)
        np.save(path / "item_ids.npy", self._item_ids)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if (path / "milvus.db").exists():
            self._db_path = path / "milvus.db"
            self._client = self._MilvusClient(uri=str(self._db_path))
            # Milvus 의 collection 은 새 client 가 열 때 release 상태일 수 있음 → 명시적 load
            try:
                self._client.load_collection(self._collection_name)
            except Exception:
                pass
        self._item_ids = np.load(path / "item_ids.npy")

    def device_info(self) -> dict[str, Any]:
        import pymilvus

        return {"device": "cpu", "name": "milvus_lite", "pymilvus": pymilvus.__version__}
