"""MovieLens-25M loader.

출처/라이센스: reports/02_dataset_selection.md §2.1.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .base import RecoDataset


class MovieLens25M(RecoDataset):
    name = "ml25m"

    def download(self) -> None:
        """grouplens.org zip 을 받아 ``self.raw_dir`` 에 풀어 둔다."""
        url = self.cfg["source"]["url"]
        out_dir = Path(self.raw_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / "ml-25m.zip"

        if not (out_dir / "ratings.csv").exists():
            if not zip_path.exists():
                print(f"[ml25m] downloading {url} → {zip_path}")
                urllib.request.urlretrieve(url, zip_path)
            print(f"[ml25m] unzipping {zip_path}")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out_dir)
            # ml-25m/ 하위 파일들을 out_dir 로 올린다
            inner = out_dir / "ml-25m"
            if inner.exists():
                for p in inner.iterdir():
                    shutil.move(str(p), str(out_dir / p.name))
                inner.rmdir()
            zip_path.unlink(missing_ok=True)

        # sha 확인 (선택; YAML 의 expected_sha256 가 placeholder 라 skip 가능)
        expected = self.cfg["source"].get("expected_sha256")
        if expected and expected != "placeholder":
            sha = hashlib.sha256()
            with open(out_dir / "ratings.csv", "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    sha.update(chunk)
            actual = sha.hexdigest()
            # 다운로드 mirror 가 동일하면 일치해야 함; 다르면 경고만
            if actual != expected:
                print(f"[ml25m] sha256 mismatch (expected={expected}, got={actual}). 계속 진행.")

    def _load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ratings.csv + movies.csv 를 canonical schema 로 변환."""
        raw = Path(self.raw_dir)
        ratings = pd.read_csv(
            raw / "ratings.csv",
            dtype={"userId": "int64", "movieId": "int64", "rating": "float32"},
        )
        ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
        ratings["ts"] = ratings["ts"].astype("int64")

        movies = pd.read_csv(
            raw / "movies.csv",
            dtype={"movieId": "int64", "title": "string", "genres": "string"},
        )
        movies = movies.rename(columns={"movieId": "item_id", "genres": "category"})
        movies["text"] = movies["title"]
        item_meta = movies[["item_id", "title", "category", "text"]]
        return ratings[["user_id", "item_id", "ts", "rating"]], item_meta
