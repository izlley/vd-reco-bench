"""Latency / QPS / Power profiler.

설계: reports/01_metric_design.md §3.2 (single-stream + max-throughput).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LatencyReport:
    mode: str                                  # "single_stream" | "max_throughput"
    concurrency: int
    n_queries: int
    elapsed_seconds: float
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    samples_ms: list[float] = field(default_factory=list)  # single_stream only


@dataclass
class PowerReport:
    sampled: bool
    sample_interval_ms: int
    duration_seconds: float
    mean_power_w: float
    peak_power_w: float
    energy_wh: float
    baseline_idle_w: float


class PowerSampler:
    """``nvidia-smi power.draw`` 를 백그라운드에서 일정 간격으로 샘플."""

    def __init__(self, sample_interval_ms: int = 100, device_index: int = 0) -> None:
        self.sample_interval_ms = sample_interval_ms
        self.device_index = device_index
        self._samples: list[tuple[float, float]] = []  # (t, watt)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._pynvml = pynvml
        except Exception:
            self._pynvml = None

    def baseline(self, seconds: float = 5.0) -> float:
        if self._pynvml is None:
            return 0.0
        end = time.monotonic() + seconds
        samples = []
        while time.monotonic() < end:
            try:
                p = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                samples.append(p)
            except Exception:
                pass
            time.sleep(self.sample_interval_ms / 1000.0)
        return float(np.mean(samples)) if samples else 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                p = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self._samples.append((time.monotonic(), p))
            except Exception:
                pass
            self._stop.wait(self.sample_interval_ms / 1000.0)

    def start(self) -> None:
        if self._pynvml is None:
            return
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[list[tuple[float, float]], bool]:
        if self._thread is None:
            return [], False
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self._samples, True

    def report(self, baseline_w: float = 0.0) -> PowerReport:
        if not self._samples:
            return PowerReport(False, self.sample_interval_ms, 0.0, 0.0, 0.0, 0.0, baseline_w)
        ts = np.array([s[0] for s in self._samples])
        ws = np.array([s[1] for s in self._samples])
        dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        mean_p = float(ws.mean())
        peak_p = float(ws.max())
        # 적분 (사다리꼴)
        # numpy 2.x: np.trapz → np.trapezoid
        trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
        energy_ws = float(trap(ws, ts)) if (trap is not None and len(ts) > 1) else 0.0
        return PowerReport(
            sampled=True,
            sample_interval_ms=self.sample_interval_ms,
            duration_seconds=dur,
            mean_power_w=mean_p,
            peak_power_w=peak_p,
            energy_wh=energy_ws / 3600.0,
            baseline_idle_w=baseline_w,
        )


def measure_single_stream(
    retriever,
    queries: np.ndarray,
    k: int,
    warmup_queries: int = 1000,
    measure_queries: int = 5000,
) -> LatencyReport:
    """Single-stream latency 측정 (직렬, 1 query 씩).

    ``cudaStreamSynchronize`` 는 ``Retriever.search`` 의 contract 에 따라
    자체적으로 보장된다고 가정.
    """
    n = len(queries)
    if n == 0:
        raise ValueError("empty queries")

    # warmup
    warmup_n = min(warmup_queries, n)
    retriever.warmup(queries[:warmup_n], n=warmup_n)

    # measure
    samples: list[float] = []
    measure_n = min(measure_queries, n)
    t_overall = time.monotonic()
    for i in range(measure_n):
        q = queries[i : i + 1]
        t0 = time.perf_counter()
        retriever.search(q, k=k)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    elapsed = time.monotonic() - t_overall

    arr = np.array(samples)
    return LatencyReport(
        mode="single_stream",
        concurrency=1,
        n_queries=measure_n,
        elapsed_seconds=elapsed,
        qps=measure_n / elapsed if elapsed > 0 else 0.0,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
        samples_ms=samples,
    )


def measure_max_throughput(
    retriever,
    queries: np.ndarray,
    k: int,
    concurrency: int,
    warmup_queries: int = 1000,
    min_seconds: float = 30.0,
    min_queries: int = 10_000,
) -> LatencyReport:
    """Batch concurrency 로 dispatch 하여 throughput 측정."""
    n = len(queries)
    if n == 0:
        raise ValueError("empty queries")

    retriever.warmup(queries[: min(warmup_queries, n)], n=min(warmup_queries, n))

    t0 = time.monotonic()
    n_done = 0
    samples: list[float] = []
    i = 0
    while True:
        if i + concurrency > n:
            i = 0  # query 가 부족하면 wrap
        batch = queries[i : i + concurrency]
        t_b0 = time.perf_counter()
        retriever.search(batch, k=k)
        t_b1 = time.perf_counter()
        samples.append((t_b1 - t_b0) * 1000.0 / concurrency)
        n_done += concurrency
        i += concurrency
        if n_done >= min_queries and time.monotonic() - t0 >= min_seconds:
            break
        if n_done >= 200_000:
            break  # 안전 cap
    elapsed = time.monotonic() - t0

    arr = np.array(samples)
    return LatencyReport(
        mode="max_throughput",
        concurrency=concurrency,
        n_queries=n_done,
        elapsed_seconds=elapsed,
        qps=n_done / elapsed,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
    )
