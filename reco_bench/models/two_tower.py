"""Two-tower retrieval 모델.

설계: reports/03_baseline_methodology.md §1.
- User tower: ID embedding + recent-N item-ID mean-pool + MLP → L2 norm
- Item tower: ID embedding (+ optional text) + MLP → L2 norm
- Score: cosine = dot product (둘 다 L2-norm 후)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    """Two-tower 모델. ML-25M (ID-only) 와 Amazon (ID+text) 모두 지원.

    Args:
        num_users: user 수 (dense remap 후 [0, num_users)).
        num_items: item 수.
        embed_dim_id: ID embedding 차원 (default 64).
        embed_dim_out: 두 tower 의 최종 출력 차원 (= ANN dim, default 128).
        mlp_hidden: 중간 MLP 폭 (default 256).
        text_dim: text encoder 출력 차원 (Amazon only). None 이면 text 없음.
        share_item_embedding: user tower 의 recent_item 과 item tower 의
            ItemEmbedding 공유.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim_id: int = 64,
        embed_dim_out: int = 128,
        mlp_hidden: int = 256,
        text_dim: int | None = None,
        share_item_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim_id = embed_dim_id
        self.embed_dim_out = embed_dim_out
        self.share_item_embedding = share_item_embedding

        self.user_embedding = nn.Embedding(num_users, embed_dim_id)
        self.item_embedding = nn.Embedding(num_items, embed_dim_id, padding_idx=None)

        # user tower: [u_id_emb, recent_mean_item_emb] → MLP
        user_in = 2 * embed_dim_id
        self.user_mlp = nn.Sequential(
            nn.Linear(user_in, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, embed_dim_out),
        )

        # item tower: [v_id_emb, (v_text_emb if text)] → MLP
        item_in = embed_dim_id + (text_dim if text_dim else 0)
        self.text_dim = text_dim
        self.item_mlp = nn.Sequential(
            nn.Linear(item_in, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, embed_dim_out),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)

    def encode_user(
        self,
        user_ids: torch.Tensor,        # (B,)
        recent_items: torch.Tensor,    # (B, N), -1 = pad
        recent_lengths: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """user tower forward → L2-normalize 된 ``(B, embed_dim_out)``."""
        u = self.user_embedding(user_ids)  # (B, d_id)

        # recent items mean-pool with -1 padding
        # padding_idx 가 없으므로 mask 로 처리
        mask = (recent_items >= 0).float().unsqueeze(-1)  # (B, N, 1)
        safe_ids = recent_items.clamp(min=0)
        emb = self.item_embedding(safe_ids)  # (B, N, d_id)
        emb = emb * mask
        recent_sum = emb.sum(dim=1)                            # (B, d_id)
        denom = recent_lengths.clamp(min=1).unsqueeze(-1).float()
        recent_mean = recent_sum / denom                       # (B, d_id)

        x = torch.cat([u, recent_mean], dim=-1)
        x = self.user_mlp(x)
        return F.normalize(x, p=2, dim=-1)

    def encode_item(
        self,
        item_ids: torch.Tensor,            # (B,)
        item_text_emb: torch.Tensor | None = None,  # (B, text_dim)
    ) -> torch.Tensor:
        """item tower forward → L2-normalize 된 ``(B, embed_dim_out)``."""
        v = self.item_embedding(item_ids)  # (B, d_id)
        if self.text_dim is not None:
            assert item_text_emb is not None, "text_dim 이 설정됐는데 item_text_emb=None"
            v = torch.cat([v, item_text_emb], dim=-1)
        v = self.item_mlp(v)
        return F.normalize(v, p=2, dim=-1)

    def encode_all_items(
        self,
        device: torch.device | str = "cuda",
        batch_size: int = 8192,
        item_text_emb_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """전체 item 의 final embedding 반환 (`(num_items, embed_dim_out)`).

        ANN 인덱스 빌드용. eval 모드에서 호출하는 것 권장.
        """
        self.eval()
        num_items = self.item_embedding.num_embeddings
        out = torch.empty((num_items, self.embed_dim_out), device=device, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, num_items, batch_size):
                end = min(start + batch_size, num_items)
                ids = torch.arange(start, end, device=device)
                text = None
                if item_text_emb_table is not None:
                    text = item_text_emb_table[start:end].to(device, dtype=torch.float32)
                v = self.encode_item(ids, text)
                out[start:end] = v.to(torch.float32)
        return out
