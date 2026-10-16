"""Where is F0.5 lost?  Splits (1 - F) per S1 into: FP present / FN inside candidates / FN outside candidates.
Usage: python tools/loss_breakdown.py <run> <thr> [keep_frac world_seed]   (dense world if keep_frac given)"""
import os, sys
import numpy as np, polars as pl
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from er_common import CACHE, RUNS, macro_f05
run, thr = sys.argv[1], float(sys.argv[2])
res = pl.read_parquet(os.path.join(RUNS, run, "oof.parquet"))
gt = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_gt.parquet")).rename({"s1_idx": "s1", "r_idx": "r"})
country = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
n1 = len(country)
keep = np.ones(n1, bool)
if len(sys.argv) > 3:
    import argparse, exp03_dense as M3
    keep = M3.keep_mask(argparse.Namespace(keep_frac=float(sys.argv[3]), world_seed=int(sys.argv[4])))
gt = gt.filter(pl.Series(keep[gt["s1"].to_numpy()]))
a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first").filter(pl.col("p") > thr)
per = macro_f05(a["s1"].to_numpy(), a["r"].to_numpy(), gt["s1"].to_numpy(), gt["r"].to_numpy(), n1, return_per_entity=True)
cand = res.filter(pl.col("y") == 1).select("s1", "r").with_columns(pl.lit(1).alias("inc"))
g = gt.join(cand, on=["s1", "r"], how="left").join(a.select("s1", "r").with_columns(pl.lit(1).alias("pred")), on=["s1", "r"], how="left")
fn_in = np.bincount(g.filter(pl.col("inc").is_not_null() & pl.col("pred").is_null())["s1"].to_numpy(), minlength=n1)
fn_out = np.bincount(g.filter(pl.col("inc").is_null())["s1"].to_numpy(), minlength=n1)
fp = np.bincount(a.join(gt.with_columns(pl.lit(1).alias("t")), on=["s1", "r"], how="left").filter(pl.col("t").is_null())["s1"].to_numpy(), minlength=n1)
ngt = np.bincount(gt["s1"].to_numpy(), minlength=n1)
loss = (1 - per) * keep
N = keep.sum()
print(f"{run} thr={thr}: F0.5={per[keep].mean():.5f}  total loss={loss.sum() / N:.5f}")
cats = {
    "FP on singleton": (fp > 0) & (ngt == 0),
    "FP (non-singleton)": (fp > 0) & (ngt > 0),
    "FN in-cand only": (fp == 0) & (fn_in > 0) & (fn_out == 0),
    "FN out-of-cand only": (fp == 0) & (fn_in == 0) & (fn_out > 0),
    "FN both": (fp == 0) & (fn_in > 0) & (fn_out > 0),
}
for k, m in cats.items():
    m = m & keep
    print(f"  {k:22s}: S1 {m.sum():7d} ({m.sum() / N:.4f})  loss share {loss[m].sum() / N:.5f}")
for c in np.unique(country):
    m = keep & (country == c)
    print(f"  {c}: F0.5 {per[m].mean():.5f}")
