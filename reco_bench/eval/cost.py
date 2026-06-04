"""비용 메트릭 계산.

설계: reports/01_metric_design.md §4, reports/04_vdpu_value_proposition.md §3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CostReport:
    sku: str
    hourly_usd: float
    qps: float
    usd_per_qps: float          # qps 1 을 1시간 유지하는 비용
    usd_per_1m_queries: float   # 100만 query 의 비용
    watts_per_qps: float
    energy_usd_per_1m: float    # 전력 비용만 (PUE 적용)
    snapshot_date: str | None


def _cost_for(qps: float, mean_w: float, sku_cfg: dict, ops: dict) -> CostReport:
    hourly_usd = float(sku_cfg["hourly_usd"])
    if qps <= 0:
        return CostReport(
            sku=sku_cfg.get("description", "?"),
            hourly_usd=hourly_usd,
            qps=qps,
            usd_per_qps=math.inf,
            usd_per_1m_queries=math.inf,
            watts_per_qps=math.inf,
            energy_usd_per_1m=math.inf,
            snapshot_date=sku_cfg.get("snapshot_date"),
        )
    usd_per_query = hourly_usd / (qps * 3600.0)
    pue = float(ops.get("data_center_pue", 1.0))
    elec = float(ops.get("electricity_usd_per_kwh", 0.10))
    watts_per_qps = mean_w / qps if qps > 0 else math.inf
    # 100만 query 처리 시간 (sec) = 1e6 / qps; 그동안 소비된 에너지 = mean_w * t / 3600 kWh
    energy_kwh_1m = (mean_w * 1_000_000 / qps) / 3600.0 / 1000.0
    energy_usd_1m = energy_kwh_1m * elec * pue
    return CostReport(
        sku=sku_cfg.get("description", "?"),
        hourly_usd=hourly_usd,
        qps=qps,
        usd_per_qps=usd_per_query * 3600.0,
        usd_per_1m_queries=usd_per_query * 1_000_000,
        watts_per_qps=watts_per_qps,
        energy_usd_per_1m=energy_usd_1m,
        snapshot_date=sku_cfg.get("snapshot_date"),
    )


def pick_sku(retriever_name: str, device: str, cost_model: dict) -> tuple[str, dict]:
    """retriever name / device 를 보고 cost_model 에서 매칭 SKU 추정."""
    skus = cost_model["skus"]
    if device == "cuda":
        # H100 우선
        for k, v in skus.items():
            if v.get("hourly_usd") and "h100" in k.lower():
                return k, v
    elif device == "cpu":
        for k, v in skus.items():
            if v.get("hourly_usd") and "epyc" in k.lower():
                return k, v
    # fallback
    for k, v in skus.items():
        if v.get("hourly_usd"):
            return k, v
    raise ValueError("no SKU with hourly_usd available")


def cost_for_run(
    retriever_name: str,
    device: str,
    qps: float,
    mean_power_w: float,
    cost_model: dict,
) -> CostReport:
    sku_key, sku_cfg = pick_sku(retriever_name, device, cost_model)
    return _cost_for(qps, mean_power_w, sku_cfg, cost_model.get("ops_assumptions", {}))
