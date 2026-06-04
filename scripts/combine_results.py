"""여러 results/<exp>/metrics.json 을 합쳐 results/phase1_combined/metrics.json 생성."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
srcs = sys.argv[1:] or [
    "results/phase1_baseline_v0/metrics.json",   # ml25m: faiss, cuvs_ivfpq, cuvs_cagra
    "results/phase1_amazon/metrics.json",        # amazon: faiss, cuvs_ivfpq, cuvs_cagra, scann
    "results/phase1_qdrant_server/metrics.json", # both: qdrant_server
]
rows = []
for s in srcs:
    p = ROOT / s
    if p.exists():
        rows += json.loads(p.read_text()).get("rows", [])
        print(f"  + {s}: {len(json.loads(p.read_text()).get('rows', []))} rows")
    else:
        print(f"  - {s}: (없음, skip)")
out = ROOT / "results/phase1_combined"
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps({"experiment_id":"phase1_combined","rows":rows}, indent=2, default=str))
print(f"\ncombined {len(rows)} rows → results/phase1_combined/metrics.json")
