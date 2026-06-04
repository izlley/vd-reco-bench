"""평가용 numpy 배열 (query_emb, exact top-k ids, ground truth) 을 dump.

FAISS-GPU 등 별도 환경(venv/conda)의 worker 가 PyTorch 없이 측정할 수
있도록, main python 으로 query embedding / exact top-K / ground truth 를
미리 계산해 npz 로 저장한다.

사용: python scripts/dump_eval_arrays.py <experiment_yaml>
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reco_bench.data.base import build_ground_truth
from reco_bench.data.pipeline import load_train_positives
from reco_bench.models.two_tower import TwoTower
from reco_bench.retrievers.exact_topk import ExactTopK
from reco_bench.utils.io import load_yaml
from reco_bench.utils.seed import set_global_seed

cfg = load_yaml(sys.argv[1])
set_global_seed()
exp_id = cfg["experiment_id"]
model_cfg = load_yaml(cfg["model"])
recent_n = int(model_cfg["architecture"]["recent_n"])
arch = model_cfg["architecture"]
ks = list(cfg["evaluation"]["ks"]); max_k = max(ks)
device = "cuda" if torch.cuda.is_available() else "cpu"
out_root = Path("results/_eval_arrays"); out_root.mkdir(parents=True, exist_ok=True)

for ds_cfg_path in cfg["datasets"]:
    ds_cfg = load_yaml(ds_cfg_path); ds_name = ds_cfg["name"]
    ckpt = Path("checkpoints")/exp_id/ds_name
    if not (ckpt/"item_embeddings.npy").exists():
        print(f"skip {ds_name}: no ckpt"); continue
    proc = Path(ds_cfg["processed_dir"])
    test_df = pd.read_parquet(proc/"test.parquet")
    pos = load_train_positives(proc/"train_positives.npz")
    stats = json.loads((proc/"stats.json").read_text())
    m = TwoTower(num_users=stats["num_users"], num_items=stats["num_items"],
                 embed_dim_id=arch["embed_dim_id"], embed_dim_out=arch["embed_dim_out"],
                 mlp_hidden=arch["mlp_hidden"])
    m.load_state_dict(torch.load(ckpt/"model.pt", map_location=device)); m=m.to(device).eval()

    users = test_df["user_id"].unique()
    rng = np.random.RandomState(42)
    sel = rng.choice(users, size=5000, replace=False) if len(users)>5000 else users
    recent = np.full((len(sel),recent_n),-1,dtype=np.int64); lengths=np.zeros(len(sel),dtype=np.int64)
    for i,u in enumerate(sel.tolist()):
        p=pos.get(int(u))
        if p is not None and len(p)>0:
            n=min(len(p),recent_n); recent[i,:n]=p[-n:]; lengths[i]=n
    with torch.no_grad():
        uemb=m.encode_user(torch.tensor(sel,device=device),torch.tensor(recent,device=device),
                           torch.tensor(lengths,device=device)).cpu().numpy().astype(np.float32)
    gt = build_ground_truth(test_df, users=sel)
    item_emb=np.load(ckpt/"item_embeddings.npy"); item_ids=np.load(ckpt/"item_ids.npy")
    ex=ExactTopK(device=device); ex.build(item_emb,item_ids,{})
    exact_ids,_=ex.search(uemb,k=max_k)
    # gt 를 ragged → padded 로 저장 (worker 가 set 으로 복원)
    gt_users=np.array(sorted(gt.keys()),dtype=np.int64)
    gt_lens=np.array([len(gt[u]) for u in gt_users.tolist()],dtype=np.int64)
    gt_items=np.concatenate([np.array(sorted(gt[u]),dtype=np.int64) for u in gt_users.tolist()]) if len(gt_users) else np.array([],dtype=np.int64)
    np.savez(out_root/f"{ds_name}.npz", query_emb=uemb, user_ids=sel.astype(np.int64),
             item_emb=item_emb, item_ids=item_ids, exact_ids=exact_ids.astype(np.int64),
             gt_users=gt_users, gt_lens=gt_lens, gt_items=gt_items, max_k=max_k)
    print(f"{ds_name}: query={len(sel)}, items={len(item_ids)}, saved → results/_eval_arrays/{ds_name}.npz")
