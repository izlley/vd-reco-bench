"""Two-tower loss: sampled softmax with in-batch negatives + logQ correction.

수식 정의: reports/03_baseline_methodology.md §1.4.
참고: Yi et al., "Sampling-bias-corrected neural modeling for large
corpus item recommendations" (RecSys 2019).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class ItemFrequencyEstimator:
    """Item 출현 빈도를 EMA 기반으로 추정 — logQ 보정의 ``Q(v)`` 계산.

    원 논문은 hash-based count-min sketch 를 쓰지만, 본 구현은 단순한
    배열-기반 EMA 로 구현 (item 수가 수십만 단위라 hash 가 불필요).
    """

    def __init__(self, num_items: int, ema_alpha: float = 0.01) -> None:
        # log(Q(v)) 의 EMA 추정. 초기값 = log(1/N) 균등분포.
        self.num_items = num_items
        self.ema_alpha = ema_alpha
        init = math.log(1.0 / max(1, num_items))
        self.log_q = torch.full((num_items,), init, dtype=torch.float32)
        self._counts = torch.zeros(num_items, dtype=torch.int64)
        self._total = 0

    def to(self, device) -> "ItemFrequencyEstimator":
        self.log_q = self.log_q.to(device)
        self._counts = self._counts.to(device)
        return self

    def update(self, item_ids: torch.Tensor) -> None:
        """배치의 item id 들로 빈도 추정 갱신."""
        with torch.no_grad():
            self._counts.scatter_add_(
                0, item_ids.long(), torch.ones_like(item_ids, dtype=torch.int64)
            )
            self._total += int(item_ids.numel())
            if self._total > 0:
                # 실측 빈도와 EMA 혼합
                empirical = (self._counts.float() + 1.0) / (self._total + self.num_items)
                log_empirical = empirical.log()
                self.log_q = (1.0 - self.ema_alpha) * self.log_q + self.ema_alpha * log_empirical


def sampled_softmax_in_batch_loss(
    user_emb: torch.Tensor,     # (B, d), L2-norm
    item_emb: torch.Tensor,     # (B, d), L2-norm
    item_ids: torch.Tensor,     # (B,)
    temperature: float = 0.07,
    item_freq: ItemFrequencyEstimator | None = None,
) -> torch.Tensor:
    """in-batch negatives + (optional) logQ correction.

    Args:
        user_emb / item_emb: paired positives, 같은 배치 내 다른 item 이
            negative 가 됨.
        item_ids: ``item_emb`` 의 각 행에 해당하는 item id.
        temperature: softmax temperature. 작을수록 sharp.
        item_freq: None 이면 logQ 보정 미적용 (참고용).

    Returns:
        스칼라 loss.
    """
    # (B, B) logits
    logits = (user_emb @ item_emb.T) / temperature

    if item_freq is not None:
        log_q = item_freq.log_q[item_ids].to(logits.device)  # (B,)
        # 각 column (negative item) 에서 log_q 를 빼면 popular item 의 점수가 낮아짐
        logits = logits - log_q.unsqueeze(0)

    targets = torch.arange(user_emb.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)
