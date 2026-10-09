"""Systematic label-shift scan: for each fine group x p-bin, test accepted per 1K S1 minus OOF (TP+FP) per 1K S1.
Positive excess concentrated where OOF has few TPs => test-only distractors the model accepts."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
os.environ["ER_WORK_DIR"] = _WORK
import exp03_dense as M3
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(30)
CACHE = _os.path.join(_WORK, 'data', 'cache')
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
s1tr = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
s1te = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
to = pl.read_parquet(os.path.join(SP, "oof_typed.parquet")); te = pl.read_parquet(os.path.join(SP, "test_typed.parquet"))
dh = (pl.col("h1").str.extract(r"(\d+)").cast(pl.Int64, strict=False) - pl.col("h2").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).abs()
def enrich(t):
    t = t.with_columns(dh.alias("dh"))
    return t.with_columns(
        pl.when(pl.col("num") != "diffnum").then(pl.lit("-")).when(pl.col("dh") <= 2).then(pl.lit("<=2")).when(pl.col("dh") <= 12).then(pl.lit("3-12"))
        .when(pl.col("dh") <= 100).then(pl.lit("13-100")).otherwise(pl.lit(">100")).alias("dist"),
        pl.when(pl.col("f1") == pl.col("f2")).then(pl.lit("same")).when(pl.col("f1") == "").then(pl.lit("on_R")).when(pl.col("f2") == "").then(pl.lit("on_S1")).otherwise(pl.lit("diff")).alias("side"),
        pl.when(pl.col("p") <= 0.7).then(pl.lit("0")).when(pl.col("p") <= 0.8).then(pl.lit("p.7")).when(pl.col("p") <= 0.9).then(pl.lit("p.8"))
        .when(pl.col("p") <= 0.95).then(pl.lit("p.9")).when(pl.col("p") <= 0.98).then(pl.lit("p.95")).when(pl.col("p") <= 0.99).then(pl.lit("p.98")).otherwise(pl.lit("p.99")).alias("pb"))
to = enrich(to).filter(pl.col("p") > 0.7); te = enrich(te).filter(pl.col("p") > 0.7)
G = ["ntype", "num", "side", "dist"]
out = []
for c in ["US", "India"]:
    ntr = int((keep & (s1tr == c)).sum()); nte = int((s1te == c).sum())
    o = to.filter(pl.col("country") == c).group_by(G + ["pb"]).agg((pl.col("y").sum() * 1000 / ntr).alias("otp"), ((1 - pl.col("y")).sum() * 1000 / ntr).alias("ofp"))
    t = te.filter(pl.col("country") == c).group_by(G + ["pb"]).agg((pl.len() * 1000 / nte).alias("tacc"), pl.len().alias("tn"))
    m = o.join(t, on=G + ["pb"], how="full", coalesce=True).fill_null(0).with_columns((pl.col("tacc") - pl.col("otp") - pl.col("ofp")).alias("excess"), pl.lit(c).alias("country"))
    out.append(m)
m = pl.concat(out)
# aggregate over the uncertain p-bins (0.7-0.98) per group
agg = m.filter(pl.col("pb").is_in(["p.7", "p.8", "p.9", "p.95"])).group_by(["country"] + G).agg(
    pl.col("otp").sum(), pl.col("ofp").sum(), pl.col("tacc").sum(), pl.col("excess").sum(), pl.col("tn").sum())
print("groups with |excess| >= 0.15 per 1K S1 in p (0.7, 0.98]:")
print(agg.filter(pl.col("excess").abs() >= 0.15).sort("excess", descending=True))
hi = m.filter(pl.col("pb").is_in(["p.98", "p.99"])).group_by(["country"] + G).agg(pl.col("otp").sum(), pl.col("ofp").sum(), pl.col("tacc").sum(), pl.col("excess").sum())
print("same groups, p > 0.98:")
print(hi.filter(pl.col("excess").abs() >= 0.3).sort("excess", descending=True))
m.write_parquet(os.path.join(SP, "scan.parquet"))
