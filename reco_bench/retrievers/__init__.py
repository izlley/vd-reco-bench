"""ANN retriever 구현체들의 공통 모듈.

새 가속기 (예: VDPU) 통합 시 본 디렉토리에 ``Retriever`` 의 새 서브
클래스 파일을 추가하고, ``configs/retrievers/`` 에 YAML 한 개와
``configs/cost_model.yaml`` 에 한 행을 더하면 끝이다. ``pipelines/``
와 ``eval/`` 는 손대지 않는다 — 이것이 본 벤치마크의 핵심 설계 계약.
"""

from .base import BuildStats, Retriever

__all__ = ["Retriever", "BuildStats"]
