"""Amazon Reviews 2023 (McAuley UCSD) loader.

설계: reports/02_dataset_selection.md §2.2.

HuggingFace 의 ``datasets`` 4.x 가 dataset loading script 지원을 끊었으므로
(``Amazon-Reviews-2023.py`` 가 동작 안 함), 본 loader 는 직접
HuggingFace Hub 의 raw jsonl.gz 파일을 ``huggingface_hub`` 로 받아 처리한다.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from .base import RecoDataset


class AmazonReviews2023(RecoDataset):
    name = "amazon2023"

    def _category_key(self) -> str:
        """YAML 의 config_review (raw_review_<Category>) 에서 <Category> 추출."""
        cfg_review = self.cfg["source"]["config_review"]
        return cfg_review.removeprefix("raw_review_")

    def download(self) -> None:
        from huggingface_hub import hf_hub_download

        src = self.cfg["source"]
        repo = src["hub_repo"]
        cat = self._category_key()
        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        review_local = raw_dir / "review.jsonl"
        meta_local = raw_dir / "meta.jsonl"

        if not review_local.exists():
            print(f"[amazon2023] downloading review: {cat}")
            p = hf_hub_download(
                repo_id=repo,
                filename=f"raw/review_categories/{cat}.jsonl",
                repo_type="dataset",
                local_dir=str(raw_dir),
            )
            src_path = Path(p)
            if src_path != review_local and src_path.exists():
                src_path.rename(review_local)

        if not meta_local.exists():
            print(f"[amazon2023] downloading meta: {cat}")
            p = hf_hub_download(
                repo_id=repo,
                filename=f"raw/meta_categories/meta_{cat}.jsonl",
                repo_type="dataset",
                local_dir=str(raw_dir),
            )
            src_path = Path(p)
            if src_path != meta_local and src_path.exists():
                src_path.rename(meta_local)

    def _load_jsonl(self, path: Path, fields: list[str]) -> pd.DataFrame:
        """매 row 가 JSON 인 .jsonl (.gz 도 자동 처리) 를 DataFrame 으로 로드."""
        open_fn = gzip.open if str(path).endswith(".gz") else open
        rows: list[dict] = []
        with open_fn(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append({k: obj.get(k) for k in fields})
        return pd.DataFrame(rows)

    def _load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw_dir = Path(self.raw_dir)
        review_path = raw_dir / "review.jsonl"
        meta_path = raw_dir / "meta.jsonl"

        # 사용할 필드만 선택 — 메모리 절약 (전체 schema 는 매우 큼)
        review = self._load_jsonl(
            review_path, ["user_id", "parent_asin", "rating", "timestamp"]
        )
        review = review.rename(columns={"parent_asin": "item_id", "timestamp": "ts"})
        review = review.dropna(subset=["user_id", "item_id", "rating", "ts"])
        # ts 가 milliseconds 이면 seconds 로
        ts0 = float(review["ts"].iloc[0])
        if ts0 > 1e12:
            review["ts"] = (review["ts"].astype("int64") // 1000)
        else:
            review["ts"] = review["ts"].astype("int64")
        review["rating"] = review["rating"].astype("float32")

        meta = self._load_jsonl(
            meta_path,
            ["parent_asin", "title", "store", "features", "categories"],
        )
        meta = meta.rename(columns={"parent_asin": "item_id"})

        def _to_text(row) -> str:
            parts = []
            for k in ("title", "store"):
                v = row.get(k)
                if isinstance(v, str):
                    parts.append(v)
            feats = row.get("features")
            if isinstance(feats, list) and feats:
                parts.append(" ".join(str(x) for x in feats[:3]))
            return " ".join(parts)[:512]

        meta["text"] = meta.apply(_to_text, axis=1)
        meta["category"] = meta["categories"].apply(
            lambda c: " > ".join(c) if isinstance(c, (list, tuple)) else (c if isinstance(c, str) else "")
        )

        item_meta = meta[["item_id", "title", "category", "text"]].copy()
        item_meta["title"] = item_meta["title"].astype("string")
        item_meta["category"] = item_meta["category"].astype("string")
        item_meta["text"] = item_meta["text"].astype("string")

        return review[["user_id", "item_id", "ts", "rating"]], item_meta
