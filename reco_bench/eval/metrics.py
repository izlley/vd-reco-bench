"""Retrieval quality 메트릭.

수식과 정의는 ``reports/01_metric_design.md`` §2 에서 확정되어 있다.
본 모듈은 그 수식을 그대로 구현한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

GroundTruth = Mapping[int, Sequence[int] | set[int]]
"""사용자 id → 정답 item id 들."""


def _gt_set(gt: GroundTruth, user_id: int) -> set[int]:
    g = gt.get(user_id)
    return set(g) if g is not None else set()


def recall_at_k(
    user_ids: np.ndarray,
    retrieved: np.ndarray,
    gt: GroundTruth,
    k: int,
) -> float:
    """Recall@K.

    Args:
        user_ids: shape ``(U,)``. 평가 사용자 ID.
        retrieved: shape ``(U, K_max)``. retriever 가 반환한 top-K item ID.
            K_max >= k 이어야 한다.
        gt: 사용자별 ground-truth item 집합.
        k: 측정할 K.

    Returns:
        평균 Recall@K (0~1).
    """
    if k > retrieved.shape[1]:
        raise ValueError(f"k={k} > retrieved.shape[1]={retrieved.shape[1]}")
    n_hit = 0.0
    n_pos_total = 0.0
    for u, row in zip(user_ids, retrieved[:, :k]):
        g = _gt_set(gt, int(u))
        if not g:
            continue
        hit = sum(1 for it in row if int(it) in g)
        n_hit += hit / len(g)
        n_pos_total += 1
    return n_hit / max(n_pos_total, 1.0)


def hit_rate_at_k(
    user_ids: np.ndarray,
    retrieved: np.ndarray,
    gt: GroundTruth,
    k: int,
) -> float:
    """HitRate@K. ground truth 가 단일 item 일 때 유용."""
    if k > retrieved.shape[1]:
        raise ValueError(f"k={k} > retrieved.shape[1]={retrieved.shape[1]}")
    hits = 0
    n = 0
    for u, row in zip(user_ids, retrieved[:, :k]):
        g = _gt_set(gt, int(u))
        if not g:
            continue
        n += 1
        if any(int(it) in g for it in row):
            hits += 1
    return hits / max(n, 1)


def ndcg_at_k(
    user_ids: np.ndarray,
    retrieved: np.ndarray,
    gt: GroundTruth,
    k: int,
) -> float:
    """NDCG@K. binary relevance 가정 (관련 = 1 / 비관련 = 0)."""
    if k > retrieved.shape[1]:
        raise ValueError(f"k={k} > retrieved.shape[1]={retrieved.shape[1]}")
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    total = 0.0
    n = 0
    for u, row in zip(user_ids, retrieved[:, :k]):
        g = _gt_set(gt, int(u))
        if not g:
            continue
        rel = np.array([1.0 if int(it) in g else 0.0 for it in row], dtype=np.float64)
        dcg = float((rel * discounts).sum())
        ideal_rel = np.zeros(k, dtype=np.float64)
        ideal_rel[: min(len(g), k)] = 1.0
        idcg = float((ideal_rel * discounts).sum())
        if idcg > 0:
            total += dcg / idcg
            n += 1
    return total / max(n, 1)


def mrr(
    user_ids: np.ndarray,
    retrieved: np.ndarray,
    gt: GroundTruth,
) -> float:
    """Mean Reciprocal Rank. 가장 첫 hit 의 1/rank 의 평균."""
    total = 0.0
    n = 0
    for u, row in zip(user_ids, retrieved):
        g = _gt_set(gt, int(u))
        if not g:
            continue
        n += 1
        for rank, it in enumerate(row, start=1):
            if int(it) in g:
                total += 1.0 / rank
                break
    return total / max(n, 1)


def recall_at_k_vs_exact(
    retrieved: np.ndarray,
    exact_topk: np.ndarray,
    k: int,
) -> float:
    """ANN 의 순수 정확도 — exact top-K 와의 일치율.

    모델 오차와 ANN 오차의 분리에 사용. 정의는
    ``reports/01_metric_design.md`` §2.4 참조.

    Args:
        retrieved: shape ``(U, K_max)``, ANN retriever 의 결과.
        exact_topk: shape ``(U, K_exact)``, brute-force top-K 결과.
            K_exact >= k 이어야 한다.
        k: 비교할 K.
    """
    if k > retrieved.shape[1]:
        raise ValueError(f"k={k} > retrieved.shape[1]={retrieved.shape[1]}")
    if k > exact_topk.shape[1]:
        raise ValueError(f"k={k} > exact_topk.shape[1]={exact_topk.shape[1]}")
    total = 0.0
    n = retrieved.shape[0]
    for ann_row, exact_row in zip(retrieved[:, :k], exact_topk[:, :k]):
        ann_set = set(int(x) for x in ann_row)
        exact_set = set(int(x) for x in exact_row)
        total += len(ann_set & exact_set) / k
    return total / max(n, 1)


def compute_all(
    user_ids: np.ndarray,
    retrieved: np.ndarray,
    gt: GroundTruth,
    exact_topk: np.ndarray | None = None,
    ks: Sequence[int] = (10, 100),
) -> dict[str, float]:
    """편의 함수. baseline_results.md 의 메인 표에 들어갈 모든 메트릭."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(user_ids, retrieved, gt, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(user_ids, retrieved, gt, k)
        out[f"ndcg@{k}"] = ndcg_at_k(user_ids, retrieved, gt, k)
        if exact_topk is not None:
            out[f"recall_vs_exact@{k}"] = recall_at_k_vs_exact(retrieved, exact_topk, k)
    out["mrr"] = mrr(user_ids, retrieved, gt)
    return out
