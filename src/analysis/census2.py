"""Accepted census: dense OOF (accepted + truth + FP) vs test accepted, per country, by (name type x number x sfx)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ["ER_WORK_DIR"] = _WORK
import exp03_dense as M3
from typer import typed, RUNS, CACHE
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32), pl.lit(1).alias("y"))
oof = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
to = typed(oof, "train").join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
to = to.filter(pl.col("p") > 0.7)
s1tr = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
te = typed(pl.read_parquet(os.path.join(RUNS, "exp29gf", "test_pred.parquet")), "test").filter(pl.col("p") > 0.7)
s1te = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
for c in ["US", "India"]:
    ntr = int((keep & (s1tr == c)).sum()); nte = int((s1te == c).sum())
    a = to.filter(pl.col("country") == c).group_by(["ntype", "num", "sfx"]).agg(
        (pl.len() * 1000 / ntr).alias("oof_acc"), (pl.col("y").sum() * 1000 / ntr).alias("oof_tp"), ((1 - pl.col("y")).sum() * 1000 / ntr).alias("oof_fp"))
    b = te.filter(pl.col("country") == c).group_by(["ntype", "num", "sfx"]).agg((pl.len() * 1000 / nte).alias("test_acc"))
    m = a.join(b, on=["ntype", "num", "sfx"], how="full", coalesce=True).fill_null(0).with_columns((pl.col("test_acc") - pl.col("oof_acc")).alias("diff"))
    print(f"== {c}: accepted per 1K S1 (p>0.7), dense OOF vs test; sorted by |diff|")
    print(m.sort(pl.col("diff").abs(), descending=True).head(25))
    print("   totals: oof_acc %.1f oof_fp %.1f test_acc %.1f" % (m["oof_acc"].sum(), m["oof_fp"].sum(), m["test_acc"].sum()))
