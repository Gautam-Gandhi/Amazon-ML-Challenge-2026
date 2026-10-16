"""Compare distributions of key matcher features between train (OOF world) and test, per country.
Looks only at each S2/S3 record's best S1 (r_rank_j==1) - the pairs that decide the predictions."""
import os, sys, glob
import numpy as np, polars as pl
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from er_common import CACHE
FEAT = os.path.join(CACHE, sys.argv[1] if len(sys.argv) > 1 else "feat_v1")  # usage: drift.py [feat_dir] [splits]
cols = ["s1", "r_rank_j", "s1_rank_j", "cos_j", "cos_n", "cos_a", "r_margin2", "r_ns1", "s1_ncand", "nc_tsort", "ad_tset", "num1_eq", "b_addr_empty"]
def load(split):
    c = pl.read_parquet(os.path.join(CACHE, "prep_v1", f"{split}_s1.parquet"), columns=["country"])["country"].to_numpy()
    parts = [pl.read_parquet(p, columns=cols) for p in sorted(glob.glob(os.path.join(FEAT, split, "part*.parquet")))]
    d = pl.concat(parts)
    return d.with_columns(pl.Series("country", c[d["s1"].to_numpy()]))
q = [0.1, 0.25, 0.5, 0.75, 0.9]
for split in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["train", "test"]):
    d = load(split)
    print(f"== {split}: pairs/S1 {d.height / d['s1'].n_unique():.2f}")
    for c in sorted(d["country"].unique()):
        x = d.filter((pl.col("country") == c))
        top = x.filter(pl.col("r_rank_j") == 1)
        out = {"pairs/S1": round(x.height / x["s1"].n_unique(), 2), "r_ns1 mean": round(x["r_ns1"].mean(), 2)}
        for f in ["cos_j", "cos_n", "cos_a", "r_margin2", "nc_tsort", "ad_tset"]:
            v = top[f].drop_nulls().to_numpy()
            out[f] = [round(float(t), 3) for t in np.quantile(v, q)]
        out["margin2<0.2"] = round(float((top["r_margin2"].fill_null(9) < 0.2).mean()), 4)
        out["addr_empty"] = round(float(top["b_addr_empty"].mean()), 4)
        print(f"  {c}: " + " | ".join(f"{k}={v}" for k, v in out.items()))
