"""identical name_core, different first house number, legal form on one side only: OOF vs test, split further."""
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
te = typed(pl.read_parquet(os.path.join(RUNS, "exp29gf", "test_pred.parquet")), "test")
s1tr = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
s1te = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
dh = (pl.col("h1").str.extract(r"(\d+)").cast(pl.Int64, strict=False) - pl.col("h2").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).abs()
def prep(t):
    t = t.filter((pl.col("ntype") == "identical") & (pl.col("num") == "diffnum") & (pl.col("sfx") == "sfx1"))
    return t.with_columns(dh.alias("dh")).with_columns(
        pl.when(pl.col("dh") <= 2).then(pl.lit("a<=2")).when(pl.col("dh") <= 12).then(pl.lit("b3-12")).when(pl.col("dh") <= 100).then(pl.lit("c13-100")).otherwise(pl.lit("d>100")).alias("dist"),
        pl.when(pl.col("f1") == "").then(pl.lit("sfx_on_R")).otherwise(pl.lit("sfx_on_S1")).alias("side"))
to = prep(to); te = prep(te)
for c in ["US", "India"]:
    ntr = int((keep & (s1tr == c)).sum()); nte = int((s1te == c).sum())
    a = to.filter(pl.col("country") == c).group_by(["dist", "side"]).agg(
        (pl.len() * 1000 / ntr).alias("oof_all"), pl.col("y").mean().alias("oof_true"),
        ((pl.col("p") > 0.7).sum() * 1000 / ntr).alias("oof_acc"), (((pl.col("p") > 0.7) & (pl.col("y") == 0)).sum() * 1000 / ntr).alias("oof_accFP"))
    b = te.filter(pl.col("country") == c).group_by(["dist", "side"]).agg((pl.len() * 1000 / nte).alias("test_all"), ((pl.col("p") > 0.7).sum() * 1000 / nte).alias("test_acc"),
                                                                         pl.col("p").mean().alias("test_mp"))
    print(f"== {c}")
    print(a.join(b, on=["dist", "side"], how="full", coalesce=True).sort(["dist", "side"]))
