"""Retriever 추상 클래스.

본 벤치마크의 유일한 가속기 플러그인 seam.
설계 근거: ``reports/03_baseline_methodology.md`` §7,
``reports/01_metric_design.md`` §5.4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BuildStats:
    """``Retriever.build()`` 의 측정 결과.

    ``reports/03_baseline_methodology.md`` §3.3 의 ``build.json`` 스키마와
    1:1 매칭. 빌드는 일회성 비용이지만 VDPU 가 빌드 자체를 가속하는
    경우 영업 포인트가 되므로 분리 보고한다.
    """

    wall_seconds: float
    peak_host_mb: float
    peak_device_mb: float
    index_disk_bytes: int
    num_items: int
    dim: int
    extra: dict[str, Any] = field(default_factory=dict)


class Retriever(ABC):
    """ANN retriever 의 추상 클래스.

    모든 retriever (FAISS-CPU/GPU, cuVS, ScaNN, 추후 VDPU) 는 본 클래스
    를 상속하고 4개의 메서드만 구현하면 된다. ``pipelines/`` 와
    ``eval/`` 의 모든 비교 로직은 본 추상에만 의존한다.
    """

    name: str = "abstract"
    device: str = "abstract"

    @abstractmethod
    def build(
        self,
        item_emb: np.ndarray,
        item_ids: np.ndarray,
        cfg: dict[str, Any],
    ) -> BuildStats:
        """Item embedding 으로 ANN index 빌드.

        Args:
            item_emb: shape ``(N, D)``, dtype ``float32``. L2-normalize 된
                상태로 들어온다고 가정 (inner product = cosine).
            item_ids: shape ``(N,)``, dtype ``int64``. ``item_emb`` 의 각
                행에 대응하는 원본 item ID. ``search()`` 의 반환에서
                이 ID 가 그대로 나와야 한다.
            cfg: ``configs/retrievers/*.yaml`` 의 build 블록.

        Returns:
            BuildStats: 빌드 측정 결과.
        """

    @abstractmethod
    def search(
        self,
        queries: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch query 의 top-k retrieval.

        Args:
            queries: shape ``(B, D)``, dtype ``float32``. L2-normalize 가정.
            k: 반환할 top-k.

        Returns:
            ids: shape ``(B, k)``, dtype ``int64``. ``build()`` 시 받은
                ``item_ids`` 의 값.
            scores: shape ``(B, k)``, dtype ``float32``. cosine 유사도
                (= inner product).

        Notes:
            구현자는 반드시 **CPU-side numpy** 로 반환해야 한다. device
            tensor 를 그대로 반환하면 retriever 간 시간 비교가 apples-to-
            apples 가 아니게 된다 (transfer cost 가 일부만 빠지는 효과).

            장시간 동기 보장: 반환 직전에 device synchronize (CUDA: ``cudaStreamSynchronize``,
            VDPU: 동등 API) 가 호출되어 있어야 한다. ``eval/profiler.py``
            의 latency 측정이 이 가정을 사용한다.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Index 를 디스크에 직렬화."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Index 를 디스크에서 복원. ``build()`` 와 둘 중 하나만 호출됨."""

    @abstractmethod
    def device_info(self) -> dict[str, Any]:
        """현재 retriever 가 사용하는 device 의 메타데이터.

        ``results/<exp_id>/hardware.json`` 에 dump 되어 cost model 의 SKU
        매칭과 재현성 검증에 사용된다.
        """

    # --- Default helpers — 서브클래스가 override 할 필요 없음 ---

    def warmup(self, queries: np.ndarray, n: int = 1000) -> None:
        """측정 전 warmup. 결과는 폐기.

        ``reports/01_metric_design.md`` §3.2 에 따라 latency profiling 전에
        반드시 호출된다. 기본 구현은 ``search()`` 를 n 번 호출하지만
        가속기 별로 더 효율적인 방법이 있다면 override 가능.
        """
        if len(queries) == 0:
            return
        batch = queries[: min(64, len(queries))]
        for _ in range(max(1, n // len(batch))):
            self.search(batch, k=10)

    def __repr__(self) -> str:
        return f"<Retriever name={self.name!r} device={self.device!r}>"
