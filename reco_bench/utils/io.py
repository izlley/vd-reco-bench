"""IO 헬퍼: YAML config 로드, JSON / parquet 직렬화, 경로 정규화."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """YAML config 로드."""
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    """YAML 저장 (재현용 config 스냅샷)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    import datetime as _dt

    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    try:
        return str(obj)
    except Exception:
        raise TypeError(f"Object of type {type(obj).__name__} not JSON serializable") from None


def dump_json(data: Any, path: str | Path) -> None:
    """JSON 저장. results/<exp_id>/*.json 의 표준 출력.

    Numpy scalar, datetime/date, 기타 stringifiable 객체는 default
    encoder 가 처리.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def ensure_dir(path: str | Path) -> Path:
    """디렉토리 보장 후 ``Path`` 반환."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
