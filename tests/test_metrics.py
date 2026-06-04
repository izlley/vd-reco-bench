"""metrics.py 의 sanity test.

수식 정의는 reports/01_metric_design.md §2.
"""

from __future__ import annotations

import math

import numpy as np

from reco_bench.eval.metrics import (
    compute_all,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    recall_at_k_vs_exact,
)


def test_recall_at_k_perfect():
    user_ids = np.array([0, 1])
    retrieved = np.array([[10, 20, 30], [40, 50, 60]])
    gt = {0: [10], 1: [40, 50]}
    # user 0: 1/1 hit, user 1: 2/2 hit, 평균 1.0
    assert recall_at_k(user_ids, retrieved, gt, k=2) == 1.0


def test_recall_at_k_partial():
    user_ids = np.array([0, 1])
    retrieved = np.array([[10, 20], [99, 50]])
    gt = {0: [10, 99], 1: [40, 50]}
    # user 0: 1/2, user 1: 1/2 → 평균 0.5
    assert recall_at_k(user_ids, retrieved, gt, k=2) == 0.5


def test_hit_rate_at_k():
    user_ids = np.array([0, 1, 2])
    retrieved = np.array([[1, 2], [3, 4], [5, 6]])
    gt = {0: [1], 1: [99], 2: [6]}
    # user 0: hit, user 1: miss, user 2: hit → 2/3
    assert math.isclose(hit_rate_at_k(user_ids, retrieved, gt, k=2), 2 / 3)


def test_ndcg_at_k_first_position():
    user_ids = np.array([0])
    retrieved = np.array([[10, 20, 30]])
    gt = {0: [10]}
    # 첫 자리에 정답 → NDCG = 1.0
    assert math.isclose(ndcg_at_k(user_ids, retrieved, gt, k=3), 1.0)


def test_ndcg_at_k_second_position():
    user_ids = np.array([0])
    retrieved = np.array([[20, 10, 30]])
    gt = {0: [10]}
    # 두 번째 자리: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    expected = 1.0 / math.log2(3)
    assert math.isclose(ndcg_at_k(user_ids, retrieved, gt, k=3), expected)


def test_mrr_basic():
    user_ids = np.array([0, 1])
    retrieved = np.array([[10, 20, 30], [99, 88, 77]])
    gt = {0: [20], 1: [88]}
    # user 0: rank 2 → 1/2, user 1: rank 2 → 1/2, 평균 0.5
    assert math.isclose(mrr(user_ids, retrieved, gt), 0.5)


def test_recall_at_k_vs_exact_perfect():
    retrieved = np.array([[1, 2, 3], [4, 5, 6]])
    exact = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k_vs_exact(retrieved, exact, k=3) == 1.0


def test_recall_at_k_vs_exact_partial():
    retrieved = np.array([[1, 2, 99]])
    exact = np.array([[1, 2, 3]])
    # k=3, ANN 결과 중 2 개만 exact 와 일치 → 2/3
    assert math.isclose(recall_at_k_vs_exact(retrieved, exact, k=3), 2 / 3)


def test_compute_all_smoke():
    user_ids = np.array([0, 1])
    retrieved = np.array([[10, 20, 30], [40, 50, 60]])
    gt = {0: [10], 1: [40]}
    exact = np.array([[10, 20, 30], [40, 50, 60]])
    out = compute_all(user_ids, retrieved, gt, exact_topk=exact, ks=(1, 3))
    assert set(out.keys()) >= {
        "recall@1",
        "recall@3",
        "hit_rate@1",
        "hit_rate@3",
        "ndcg@1",
        "ndcg@3",
        "recall_vs_exact@1",
        "recall_vs_exact@3",
        "mrr",
    }
    assert out["mrr"] == 1.0  # 두 사용자 모두 rank 1 hit
