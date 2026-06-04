"""하드웨어 캡처. reports/05_reproducibility.md §4 및 reports/03_baseline_methodology.md §4.5
의 ``hardware.json`` 스키마와 매칭.
"""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
from typing import Any


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as e:  # noqa: BLE001
        return f"<failed: {e}>"


def capture_hardware() -> dict[str, Any]:
    """현재 노드의 하드웨어 메타데이터를 dict 로 캡처.

    ``results/<exp_id>/hardware.json`` 으로 dump 되는 형식.
    """
    return {
        "captured_at": _dt.datetime.now().astimezone().isoformat(),
        "gpu": _capture_gpu(),
        "cpu": _capture_cpu(),
        "os": _capture_os(),
        "python": _capture_python(),
    }


def _capture_gpu() -> dict[str, Any]:
    info: dict[str, Any] = {
        "smi_query": _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,vbios_version,memory.total,power.limit",
                "--format=csv,noheader",
            ]
        ),
        "smi_dump": _run(["nvidia-smi", "-q"]),
        "nvcc": _run(["nvcc", "--version"]),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["torch"] = "not_installed"
    return info


def _capture_cpu() -> dict[str, Any]:
    return {
        "platform": platform.processor() or platform.machine(),
        "lscpu": _run(["lscpu"]),
        "meminfo": _run(["bash", "-lc", "head -3 /proc/meminfo"]),
    }


def _capture_os() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "uname": _run(["uname", "-a"]),
    }


def _capture_python() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": _run(["bash", "-lc", "which python"]),
    }
