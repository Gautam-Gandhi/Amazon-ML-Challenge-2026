"""Estimate per-country F0.5 from the model's own probabilities (no labels needed).
E[F] for a predicted set S: (1+b2)*sum_{i in S} p_i / (|S| + b2*sum_all p_i); for an empty prediction: prod(1-p_i).
Validated on train OOF (compare with the true F0.5), then applied to test per country.
Usage: python tools/expected_f.py <run> <thr>"""
import os, sys
import numpy as np, polars as pl
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from er_common import CACHE, macro_f05
run, thr = sys.argv[1], float(sys.argv[2])
b2 = 0.25
def expected(res, n1):
    a = res.select("s1", "r", "p").with_columns(
        (pl.col("p") == pl.col("p").max().over("r")).alias("best"))
    a = a.with_columns((pl.col("best") & (pl.col("p") > thr)).alias("sel"))
    g = a.group_by("s1").agg(
        pl.col("p").filter(pl.col("sel")).sum().alias("sp_sel"),
        pl.col("sel").sum().alias("nsel"),
        pl.col("p").filter(pl.col("best")).sum().alias("sp_all"),
        (1 - pl.col("p").filter(pl.col("best")).clip(0, 1 - 1e-6)).log().sum().exp().alias("p_empty"))
    ef = np.ones(n1)  # S1 with no candidates: predicted empty; assume singleton-ish
    s = g["s1"].to_numpy()
    nsel = g["nsel"].to_numpy(); sp = g["sp_sel"].to_numpy(); sa = g["sp_all"].to_numpy(); pe = g["p_empty"].to_numpy()
    ef[s] = np.where(nsel > 0, (1 + b2) * sp / (nsel + b2 * np.maximum(sa, 1e-9)), pe)
    return ef, a
for split in ["train", "test"]:
    s1c = pl.read_parquet(os.path.join(CACHE, "prep_v1", f"{split}_s1.parquet"), columns=["country"])["country"].to_numpy()
    n1 = len(s1c)
    res = pl.read_parquet(os.path.join(ROOT, "runs", run, "oof.parquet" if split == "train" else "test_pred.parquet"))
    ef, a = expected(res, n1)
    line = {c: round(float(ef[s1c == c].mean()), 5) for c in np.unique(s1c)}
    line["ALL"] = round(float(ef.mean()), 5)
    if split == "train":
        gt = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_gt.parquet"))
        d = a.filter(pl.col("sel"))
        true = macro_f05(d["s1"].to_numpy(), d["r"].to_numpy(), gt["s1_idx"].to_numpy(), gt["r_idx"].to_numpy(), n1, return_per_entity=True)
        print("train TRUE F0.5    ", {c: round(float(true[s1c == c].mean()), 5) for c in np.unique(s1c)}, "ALL", round(float(true.mean()), 5))
    print(f"{split} EXPECTED F0.5", line)
