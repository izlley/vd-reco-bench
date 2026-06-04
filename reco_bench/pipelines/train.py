"""Two-tower 학습 파이프라인.

CLI: ``python -m reco_bench.pipelines.train <experiment_yaml>``

설계: reports/03_baseline_methodology.md §2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import InteractionsDataset, collate_batch
from ..data.pipeline import load_train_positives
from ..eval.metrics import recall_at_k
from ..models.losses import ItemFrequencyEstimator, sampled_softmax_in_batch_loss
from ..models.two_tower import TwoTower
from ..utils.gpu_info import capture_hardware
from ..utils.io import dump_json, ensure_dir, load_yaml
from ..utils.seed import set_global_seed, worker_init_fn


def _build_model(num_users: int, num_items: int, model_cfg: dict) -> TwoTower:
    arch = model_cfg["architecture"]
    return TwoTower(
        num_users=num_users,
        num_items=num_items,
        embed_dim_id=arch["embed_dim_id"],
        embed_dim_out=arch["embed_dim_out"],
        mlp_hidden=arch["mlp_hidden"],
        text_dim=None,                   # Step 7 에서 Amazon 시 채워짐
        share_item_embedding=arch.get("share_item_embedding", True),
    )


def _cosine_lr_lambda(step: int, *, warmup: int, total: int) -> float:
    """warmup 후 cosine decay."""
    if step < warmup:
        return float(step) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


@torch.no_grad()
def evaluate_recall(
    model: TwoTower,
    val_df: pd.DataFrame,
    train_positives: dict[int, np.ndarray],
    num_items: int,
    recent_n: int,
    device: torch.device,
    ks: tuple[int, ...] = (10, 100),
    max_users: int = 5000,
) -> dict[str, float]:
    """val split 에서 Recall@K 측정. 빠른 학습 신호용이라 user sub-sample.

    학습된 item embedding 전체에 대해 brute-force matmul 로 top-K 를
    계산 (item 수가 적당히 작아 GPU 한 번에 가능).
    """
    model.eval()
    # 1. item embedding table
    item_emb = model.encode_all_items(device=device, batch_size=8192)  # (N, d)

    # 2. val 의 사용자 sub-sample
    val_users_all = val_df["user_id"].unique()
    rng = np.random.RandomState(42)
    if len(val_users_all) > max_users:
        sel = rng.choice(val_users_all, size=max_users, replace=False)
    else:
        sel = val_users_all

    # ground truth: user → val 의 positive item set
    gt_df = val_df[val_df["user_id"].isin(set(sel.tolist()))]
    gt = {int(u): set(int(x) for x in g.values) for u, g in gt_df.groupby("user_id")["item_id"]}

    # user embedding 계산 (recent items 사용)
    user_ids = torch.tensor(sel, dtype=torch.long, device=device)
    recent_batch = np.full((len(sel), recent_n), -1, dtype=np.int64)
    lengths = np.zeros(len(sel), dtype=np.int64)
    for i, u in enumerate(sel.tolist()):
        pos = train_positives.get(int(u))
        if pos is None or len(pos) == 0:
            continue
        n = min(len(pos), recent_n)
        recent_batch[i, :n] = pos[-n:]
        lengths[i] = n
    recent = torch.tensor(recent_batch, device=device)
    lens = torch.tensor(lengths, device=device)
    user_emb = model.encode_user(user_ids, recent, lens)

    # top-K via single matmul
    max_k = max(ks)
    scores = user_emb @ item_emb.T              # (U, N)
    top_scores, top_idx = scores.topk(max_k, dim=1)
    top_idx = top_idx.cpu().numpy()

    metrics = {}
    user_ids_np = sel.astype(np.int64)
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(user_ids_np, top_idx, gt, k)
    return metrics


def train(config_path: str | Path, output_dir: str | Path | None = None) -> dict:
    """experiment YAML 한 개를 받아 dataset 별로 학습 → checkpoint 저장."""
    cfg = load_yaml(config_path)
    set_global_seed()

    exp_id = cfg["experiment_id"]
    model_cfg = load_yaml(cfg["model"])
    arch = model_cfg["architecture"]
    train_cfg = model_cfg["training"]
    recent_n = arch["recent_n"]

    out_root = Path(output_dir or cfg.get("reporting", {}).get("output_dir", f"results/{exp_id}"))
    ensure_dir(out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] exp={exp_id} device={device}")

    # 하드웨어 캡처
    dump_json(capture_hardware(), out_root / "hardware.json")

    summary = {"experiment_id": exp_id, "datasets": {}}

    for ds_cfg_path in cfg["datasets"]:
        ds_cfg = load_yaml(ds_cfg_path)
        ds_name = ds_cfg["name"]
        print(f"\n[train] dataset={ds_name}")
        proc_dir = Path(ds_cfg["processed_dir"])

        if not (proc_dir / "train.parquet").exists():
            print(f"  [skip] processed data not found at {proc_dir}. Run scripts/00_download_data.sh {ds_name} first.")
            summary["datasets"][ds_name] = {"skipped": True, "reason": "no_processed_data"}
            continue

        # 로딩
        train_df = pd.read_parquet(proc_dir / "train.parquet")
        val_df = pd.read_parquet(proc_dir / "val.parquet")
        stats = json.loads((proc_dir / "stats.json").read_text())
        num_users = stats["num_users"]
        num_items = stats["num_items"]
        print(f"  users={num_users:,}, items={num_items:,}, train={len(train_df):,}")

        positives = load_train_positives(proc_dir / "train_positives.npz")

        train_ds = InteractionsDataset(train_df, positives, recent_n=recent_n)
        bs = int(ds_cfg.get("batch_size_override", train_cfg["batch_size"]))
        loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=int(train_cfg["num_workers"]),
            collate_fn=collate_batch,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

        # 모델
        model = _build_model(num_users, num_items, model_cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params: {n_params:,}")

        # Optimizer + scheduler
        opt_cfg = model_cfg["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
        )
        max_epochs = int(train_cfg["max_epochs"])
        total_steps = max_epochs * len(loader)
        warmup = int(model_cfg["scheduler"]["warmup_steps"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: _cosine_lr_lambda(s, warmup=warmup, total=total_steps)
        )

        # logQ EMA
        item_freq = ItemFrequencyEstimator(num_items=num_items).to(device)

        # bf16 학습 (matmul) — autocast
        use_bf16 = train_cfg.get("precision", "bf16") == "bf16"
        grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))

        best = {"recall@100": -1.0, "epoch": -1}
        patience = int(train_cfg.get("early_stop_patience_epochs", 3))
        no_improve = 0
        epoch_metrics = []

        global_step = 0
        t_train_start = time.monotonic()
        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            n_batches = 0
            t0 = time.monotonic()
            pbar = tqdm(loader, desc=f"epoch {epoch+1}/{max_epochs}", leave=False)
            for users, items, recent, lengths in pbar:
                users = users.to(device, non_blocking=True)
                items = items.to(device, non_blocking=True)
                recent = recent.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
                ):
                    u_emb = model.encode_user(users, recent, lengths)
                    v_emb = model.encode_item(items)
                    loss = sampled_softmax_in_batch_loss(
                        u_emb.float(),
                        v_emb.float(),
                        items,
                        temperature=float(model_cfg["loss"]["temperature"]),
                        item_freq=item_freq if bool(model_cfg["loss"].get("logq_correction", True)) else None,
                    )
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                item_freq.update(items)

                running_loss += float(loss.detach().item())
                n_batches += 1
                global_step += 1
                if global_step % 100 == 0:
                    pbar.set_postfix(loss=running_loss / n_batches, lr=scheduler.get_last_lr()[0])

            epoch_loss = running_loss / max(1, n_batches)
            epoch_time = time.monotonic() - t0

            val_metrics = evaluate_recall(
                model, val_df, positives, num_items, recent_n, device, ks=(10, 100)
            )
            print(
                f"  [epoch {epoch+1}/{max_epochs}] loss={epoch_loss:.4f} "
                f"val_recall@10={val_metrics['recall@10']:.4f} "
                f"val_recall@100={val_metrics['recall@100']:.4f} "
                f"time={epoch_time:.1f}s"
            )
            epoch_metrics.append(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "val": val_metrics,
                    "epoch_seconds": epoch_time,
                }
            )

            r100 = val_metrics["recall@100"]
            if r100 > best["recall@100"]:
                best = {"recall@100": r100, "recall@10": val_metrics["recall@10"], "epoch": epoch + 1}
                # checkpoint 저장
                ckpt_dir = ensure_dir(Path("checkpoints") / exp_id / ds_name)
                torch.save(model.state_dict(), ckpt_dir / "model.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  early stop (no improvement for {patience} epochs)")
                    break

        total_train_time = time.monotonic() - t_train_start

        # 최고 모델 다시 로드 후 item embedding 저장
        ckpt_dir = Path("checkpoints") / exp_id / ds_name
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
        item_emb = model.encode_all_items(device=device, batch_size=8192)
        item_emb_np = item_emb.detach().cpu().numpy().astype(np.float32)
        item_ids_np = np.arange(num_items, dtype=np.int64)
        np.save(ckpt_dir / "item_embeddings.npy", item_emb_np)
        np.save(ckpt_dir / "item_ids.npy", item_ids_np)
        print(f"  saved item_embeddings.npy shape={item_emb_np.shape}")

        ds_summary = {
            "best": best,
            "epochs": epoch_metrics,
            "total_train_seconds": total_train_time,
            "num_users": num_users,
            "num_items": num_items,
            "num_params": int(n_params),
        }
        dump_json(ds_summary, ckpt_dir / "train_meta.json")
        summary["datasets"][ds_name] = ds_summary

    dump_json(summary, out_root / "train_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    train(args.config, args.output_dir)


if __name__ == "__main__":
    main()
