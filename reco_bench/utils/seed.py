"""시드 정책. reports/05_reproducibility.md §3 의 단일 source.

본 모듈의 ``set_global_seed`` 한 번만 호출하면 random/numpy/torch/PYTHONHASHSEED
가 일관되게 고정된다. 학습/평가 진입 직후 즉시 호출하는 것이 권칙.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int | None = None) -> int:
    """모든 RNG 를 단일 시드로 고정.

    Args:
        seed: 사용할 시드. None 이면 ``RECO_BENCH_SEED`` 환경 변수 → 42 순서.

    Returns:
        실제 사용된 시드.
    """
    if seed is None:
        env = os.environ.get("RECO_BENCH_SEED")
        seed = int(env) if env else DEFAULT_SEED

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


def worker_init_fn(worker_id: int) -> None:
    """``torch.utils.data.DataLoader`` 의 ``worker_init_fn`` 으로 사용."""
    env = os.environ.get("RECO_BENCH_SEED", str(DEFAULT_SEED))
    seed = int(env) + worker_id
    random.seed(seed)
    np.random.seed(seed)
