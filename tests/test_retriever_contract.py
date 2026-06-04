"""Retriever 추상 클래스의 인터페이스 계약 test.

설계 근거: reports/03_baseline_methodology.md §7.
실제 retriever 구현체가 추가되면 본 파일을 parametrize 하여 각각의
구현이 contract 를 준수하는지 검증한다 (build / save / load round-trip,
CPU-side numpy 반환, device sync 등).
"""

from __future__ import annotations

import numpy as np

from reco_bench.retrievers.base import BuildStats, Retriever


class DummyRetriever(Retriever):
    """Contract 테스트 용 더미. 단순 brute-force 로 동작."""

    name = "dummy"
    device = "cpu"

    def __init__(self) -> None:
        self._emb: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    def build(self, item_emb, item_ids, cfg):
        self._emb = item_emb.astype(np.float32)
        self._ids = item_ids.astype(np.int64)
        return BuildStats(
            wall_seconds=0.0,
            peak_host_mb=0.0,
            peak_device_mb=0.0,
            index_disk_bytes=0,
            num_items=int(item_emb.shape[0]),
            dim=int(item_emb.shape[1]),
        )

    def search(self, queries, k):
        scores = queries.astype(np.float32) @ self._emb.T
        top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        # partial-sort 이므로 행 별 내림차순 정렬
        row_indices = np.arange(top_idx.shape[0])[:, None]
        order = np.argsort(-scores[row_indices, top_idx], axis=1)
        top_idx = top_idx[row_indices, order]
        ids = self._ids[top_idx]
        out_scores = scores[row_indices, top_idx]
        return ids.astype(np.int64), out_scores.astype(np.float32)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def device_info(self):
        return {"device": "cpu", "name": "dummy"}


def test_build_returns_buildstats():
    r = DummyRetriever()
    emb = np.random.randn(20, 8).astype(np.float32)
    ids = np.arange(20, dtype=np.int64)
    stats = r.build(emb, ids, cfg={})
    assert isinstance(stats, BuildStats)
    assert stats.num_items == 20
    assert stats.dim == 8


def test_search_returns_cpu_numpy_with_correct_shape():
    r = DummyRetriever()
    emb = np.random.randn(100, 16).astype(np.float32)
    ids = np.arange(1000, 1100, dtype=np.int64)   # ID remap 검증
    r.build(emb, ids, cfg={})

    queries = np.random.randn(7, 16).astype(np.float32)
    out_ids, out_scores = r.search(queries, k=5)
    assert out_ids.dtype == np.int64
    assert out_scores.dtype == np.float32
    assert out_ids.shape == (7, 5)
    assert out_scores.shape == (7, 5)
    # ID 가 build 시 받은 ids 범위 안에 있는지
    assert out_ids.min() >= 1000
    assert out_ids.max() < 1100


def test_search_scores_descending():
    r = DummyRetriever()
    emb = np.random.randn(50, 4).astype(np.float32)
    ids = np.arange(50, dtype=np.int64)
    r.build(emb, ids, cfg={})
    queries = np.random.randn(3, 4).astype(np.float32)
    _, scores = r.search(queries, k=10)
    for row in scores:
        assert all(row[i] >= row[i + 1] for i in range(len(row) - 1))


def test_warmup_does_not_raise():
    r = DummyRetriever()
    emb = np.random.randn(10, 4).astype(np.float32)
    ids = np.arange(10, dtype=np.int64)
    r.build(emb, ids, cfg={})
    queries = np.random.randn(20, 4).astype(np.float32)
    r.warmup(queries, n=10)
