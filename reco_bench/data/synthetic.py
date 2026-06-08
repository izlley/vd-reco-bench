"""Synthetic embedding generator — Phase 2 scaling/sensitivity sweep 용.

corpus 크기와 dim 을 **통제 변수**로 두고 ANN 가속의 scaling 거동을
측정하기 위해, 학습 없이 합성 item/query embedding 을 생성한다.

추천 임베딩의 현실성을 일부 반영하기 위해 단순 i.i.d. gaussian 이 아니라
**clustered gaussian** (item 이 K 개 cluster 중심 주변에 분포) 을 쓴다.
이는 popularity/장르 cluster 가 있는 추천 item 분포에 더 가깝고, ANN
graph/IVF 의 거동도 더 현실적으로 만든다.

설계: reports/07_scaling.md.
"""

from __future__ import annotations

import numpy as np


def make_clustered_embeddings(
    n_items: int,
    dim: int,
    n_clusters: int = 256,
    cluster_std: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """L2-normalized clustered gaussian item embedding 생성.

    Args:
        n_items: item 수.
        dim: embedding 차원.
        n_clusters: cluster 중심 수.
        cluster_std: cluster 내 분산 (작을수록 빡빡한 cluster).
        seed: RNG seed.

    Returns:
        ``(n_items, dim)`` float32, 각 행 L2-norm = 1.
    """
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    out = np.empty((n_items, dim), dtype=np.float32)
    bs = 1_000_000
    for start in range(0, n_items, bs):
        end = min(start + bs, n_items)
        m = end - start
        assign = rng.randint(0, n_clusters, size=m)
        noise = rng.randn(m, dim).astype(np.float32) * cluster_std
        vecs = centers[assign] + noise
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        out[start:end] = vecs
    return out


def make_queries(
    item_emb: np.ndarray,
    n_queries: int = 1000,
    perturb: float = 0.1,
    seed: int = 123,
) -> np.ndarray:
    """query embedding 생성.

    실제 추천 query(user embedding)는 item 공간 근처에 분포하므로,
    임의 item 을 골라 약간 perturb 한 벡터를 query 로 사용한다 (순수
    random 보다 의미 있는 top-K 가 나옴).
    """
    rng = np.random.RandomState(seed)
    n_items, dim = item_emb.shape
    idx = rng.randint(0, n_items, size=n_queries)
    q = item_emb[idx].astype(np.float32) + rng.randn(n_queries, dim).astype(np.float32) * perturb
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q
