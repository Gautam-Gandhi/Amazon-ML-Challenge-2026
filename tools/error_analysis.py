"""Error analysis: sample FP / FN pairs of a run's OOF predictions at assign@thr.  Usage: python tools/error_analysis.py <run> <thr>"""
import os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
import exp01_match as M
run = sys.argv[1] if len(sys.argv) > 1 else "exp01"; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
res = pl.read_parquet(os.path.join(M.RUNS, run, "oof.parquet"))
s1 = pl.read_parquet(os.path.join(M.PREP, "train_s1.parquet"), columns=["name_raw", "addr_raw", "country"])
r = pl.read_parquet(os.path.join(M.PREP, "train_r.parquet"), columns=["name_raw", "addr_raw"])
gt = pl.read_parquet(os.path.join(M.PREP, "train_gt.parquet")).rename({"s1_idx": "s1", "r_idx": "r"})
a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
pred = a.filter(pl.col("p") > thr).with_columns(pl.lit(1).alias("pred"))
j = res.join(pred.select("s1", "r", "pred"), on=["s1", "r"], how="left").with_columns(pl.col("pred").fill_null(0))
fp = j.filter((pl.col("pred") == 1) & (pl.col("y") == 0))
fn = j.filter((pl.col("pred") == 0) & (pl.col("y") == 1))
print("FP", fp.height, "FN(in cand)", fn.height, "missing from cand", gt.height - res["y"].sum())
# FN reason: r assigned to a different S1?
fn_r_taken = fn.join(pred.select("r", pl.col("s1").alias("s1_other")), on="r", how="left")
print("FN where r was assigned to another S1:", fn_r_taken["s1_other"].is_not_null().sum())
# for FP: true owner of r
fp2 = fp.join(gt.rename({"s1": "s1_true"}), on="r", how="left")
print("FP where r truly belongs to another S1:", fp2["s1_true"].is_not_null().sum(), " r is distractor:", fp2["s1_true"].is_null().sum())
rng = np.random.default_rng(0)
def show(df, title, n=12):
    print("=" * 30, title)
    for row in df.sample(min(n, df.height), seed=1).iter_rows(named=True):
        A = s1.row(row["s1"], named=True); B = r.row(row["r"], named=True)
        extra = ""
        if row.get("s1_true") is not None:
            T = s1.row(row["s1_true"], named=True); extra = f"\n      TRUE S1: {T['name_raw']} | {T['addr_raw']}"
        if row.get("s1_other") is not None:
            T = s1.row(row["s1_other"], named=True); extra = f"\n      TAKEN BY: {T['name_raw']} | {T['addr_raw']}"
        print(f"p={row['p']:.3f} [{A['country']}] S1: {A['name_raw']} | {A['addr_raw']}\n      R : {B['name_raw']} | {B['addr_raw']}{extra}")
show(fp2, "False positives", 14)
show(fn_r_taken, "False negatives (in candidates)", 26)
