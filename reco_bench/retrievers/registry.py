"""YAML config 의 ``impl`` 필드를 보고 Retriever 클래스 동적 로드.

새 retriever 추가 시 본 모듈은 수정 불필요 — YAML 의 ``impl`` 만 가리
키면 된다. 이것이 **VDPU 통합 seam** 의 본질.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import Retriever


def load_retriever(cfg: dict[str, Any]) -> Retriever:
    """``cfg['impl']`` (예: ``reco_bench.retrievers.cuvs_cagra:CuvsCagra``)
    을 파싱해 인스턴스 반환."""
    impl = cfg["impl"]
    module_path, class_name = impl.split(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, Retriever):
        raise TypeError(f"{impl} is not a Retriever subclass")
    return cls()
