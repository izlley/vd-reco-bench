"""FAISS-GPU IVF-PQ 측정 worker (conda faiss env 전용, PyTorch 미사용).

results/_eval_arrays/<dataset>.npz (dump_eval_arrays.py 생성) 를 읽어
FAISS-GPU IVF-PQ 를 build → param grid search → recall(vs exact/gt) +
latency/QPS 측정 → results/phase1_faiss_gpu/metrics.json (rows append).

사용: <conda faiss python> scripts/faiss_gpu_worker.py
"""
import json, time
from pathlib import Path
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parent.parent
ARR = ROOT / "results/_eval_arrays"
OUT = ROOT / "results/phase1_faiss_gpu"; OUT.mkdir(parents=True, exist_ok=True)

NLIST_DEFAULT = 4096
NPROBE_GRID = [1, 4, 16, 64, 256]
KS = [10, 100]
MAXK = 100
CONCURRENCIES = [1, 4, 16, 64]

def recall_at_k(user_ids, retrieved, gt, k):
    hit=0.0; n=0.0
    for u,row in zip(user_ids, retrieved[:,:k]):
        g=gt.get(int(u))
        if not g: continue
        hit+=sum(1 for it in row if int(it) in g)/len(g); n+=1
    return hit/max(n,1.0)

def recall_vs_exact(retrieved, exact, k):
    tot=0.0
    for a,e in zip(retrieved[:,:k], exact[:,:k]):
        tot += len(set(int(x) for x in a) & set(int(x) for x in e))/k
    return tot/max(len(retrieved),1)

def ndcg_at_k(user_ids, retrieved, gt, k):
    disc=1.0/np.log2(np.arange(2,k+2)); tot=0.0; n=0
    for u,row in zip(user_ids, retrieved[:,:k]):
        g=gt.get(int(u))
        if not g: continue
        rel=np.array([1.0 if int(it) in g else 0.0 for it in row])
        dcg=float((rel*disc).sum())
        ideal=np.zeros(k); ideal[:min(len(g),k)]=1.0; idcg=float((ideal*disc).sum())
        if idcg>0: tot+=dcg/idcg; n+=1
    return tot/max(n,1)

rows=[]
res = faiss.StandardGpuResources()
for npz_path in sorted(ARR.glob("*.npz")):
    ds = npz_path.stem
    z = np.load(npz_path)
    item_emb=z["item_emb"].astype(np.float32); item_ids=z["item_ids"]
    q=z["query_emb"].astype(np.float32); uids=z["user_ids"]; exact=z["exact_ids"]
    # gt 복원
    gt={}; cur=0
    for u,ln in zip(z["gt_users"].tolist(), z["gt_lens"].tolist()):
        gt[int(u)]=set(int(x) for x in z["gt_items"][cur:cur+ln]); cur+=ln
    n,d = item_emb.shape
    print(f"[{ds}] items={n}, dim={d}, queries={len(q)}")

    quant=faiss.IndexFlatIP(d)
    cpu_idx=faiss.IndexIVFPQ(quant,d,NLIST_DEFAULT,16,8,faiss.METRIC_INNER_PRODUCT)
    gidx=faiss.index_cpu_to_gpu(res,0,cpu_idx)
    t0=time.monotonic(); gidx.train(item_emb); gidx.add(item_emb)
    faiss.GpuParameterSpace()  # noop guard
    build_s=time.monotonic()-t0
    print(f"  build: {build_s:.2f}s")

    for nprobe in NPROBE_GRID:
        gidx.setNumProbes(nprobe) if hasattr(gidx,'setNumProbes') else setattr(gidx,'nprobe',nprobe)
        # quality
        D,I = gidx.search(q, MAXK)
        ann_ids = item_ids[np.clip(I,0,n-1)]; ann_ids[I<0]=-1
        metrics={}
        for k in KS:
            metrics[f"recall@{k}"]=recall_at_k(uids,ann_ids,gt,k)
            metrics[f"ndcg@{k}"]=ndcg_at_k(uids,ann_ids,gt,k)
            metrics[f"recall_vs_exact@{k}"]=recall_vs_exact(ann_ids,exact,k)
        # single-stream latency
        for _ in range(min(200,len(q))): gidx.search(q[:1],MAXK)
        ss=[]
        for i in range(min(2000,len(q))):
            t=time.perf_counter(); gidx.search(q[i:i+1],MAXK); ss.append((time.perf_counter()-t)*1000)
        ss=np.array(ss)
        # max-throughput
        lat_mt={}
        for c in CONCURRENCIES:
            for _ in range(5): gidx.search(q[:c],MAXK)
            t0=time.monotonic(); nd=0; i=0
            while nd<2000 or time.monotonic()-t0<2.0:
                if i+c>len(q): i=0
                gidx.search(q[i:i+c],MAXK); nd+=c; i+=c
                if nd>=200000: break
            el=time.monotonic()-t0
            lat_mt[c]={"qps":nd/el,"concurrency":c,"n_queries":nd,"elapsed_seconds":el}
        max_qps=max(lat_mt[c]["qps"] for c in CONCURRENCIES)
        rows.append({
            "experiment_id":"phase1_faiss_gpu","dataset":ds,"retriever":"faiss_ivfpq_gpu",
            "device":"cuda","grid":{"nprobe":nprobe},"metrics":metrics,
            "latency_single_stream":{"p50_ms":float(np.percentile(ss,50)),
                "p95_ms":float(np.percentile(ss,95)),"p99_ms":float(np.percentile(ss,99)),
                "mean_ms":float(ss.mean()),"qps":1000/ss.mean()},
            "latency_max_throughput":lat_mt,
            "power":{"mean_power_w":0.0,"baseline_idle_w":0.0,"sampled":False},
            "cost":{"usd_per_1m_queries":0.0,"usd_per_qps":0.0,"watts_per_qps":0.0},
            "build":{"wall_seconds":build_s,"num_items":int(n),"dim":int(d)},
        })
        print(f"  nprobe={nprobe} recall@10={metrics['recall@10']:.4f} "
              f"recall_vs_exact@10={metrics['recall_vs_exact@10']:.4f} qps_max={max_qps:.0f}")

(OUT/"metrics.json").write_text(json.dumps({"experiment_id":"phase1_faiss_gpu","rows":rows},indent=2,default=str))
print(f"\nsaved {len(rows)} rows → results/phase1_faiss_gpu/metrics.json")
