"""Full table for the sibling group + France + samples of accepted US test pairs."""
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
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20)
SP = os.path.dirname(os.path.abspath(__file__))
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32), pl.lit(1).alias("y"))
oof = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
to = typed(oof, "train").join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
te = typed(pl.read_parquet(os.path.join(RUNS, "exp29gf", "test_pred.parquet")), "test")
to.write_parquet(os.path.join(SP, "oof_typed.parquet")); te.write_parquet(os.path.join(SP, "test_typed.parquet"))
s1tr = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
s1te = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
dh = (pl.col("h1").str.extract(r"(\d+)").cast(pl.Int64, strict=False) - pl.col("h2").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).abs()
def prep(t):
    t = t.filter((pl.col("ntype") == "identical") & (pl.col("num") == "diffnum"))
    return t.with_columns(dh.alias("dh")).with_columns(
        pl.when(pl.col("dh") <= 2).then(pl.lit("a<=2")).when(pl.col("dh") <= 12).then(pl.lit("b3-12")).when(pl.col("dh") <= 100).then(pl.lit("c13-100")).otherwise(pl.lit("d>100")).alias("dist"),
        pl.when(pl.col("f1") == pl.col("f2")).then(pl.lit("same")).when(pl.col("f1") == "").then(pl.lit("on_R")).when(pl.col("f2") == "").then(pl.lit("on_S1")).otherwise(pl.lit("diff")).alias("side"))
to2 = prep(to); te2 = prep(te)
for c in ["US", "India", "France"]:
    nte = int((s1te == c).sum()); thr = 0.9 if c == "France" else 0.7
    b = te2.filter(pl.col("country") == c).group_by(["side", "dist"]).agg((pl.len() * 1000 / nte).alias("T_all"), ((pl.col("p") > thr).sum() * 1000 / nte).alias("T_acc"), pl.col("p").mean().alias("T_mp"))
    if c != "France":
        ntr = int((keep & (s1tr == c)).sum())
        a = to2.filter(pl.col("country") == c).group_by(["side", "dist"]).agg(
            (pl.len() * 1000 / ntr).alias("O_all"), (pl.col("y").sum() * 1000 / ntr).alias("O_true"), ((pl.col("p") > 0.7).sum() * 1000 / ntr).alias("O_acc"),
            (((pl.col("p") > 0.7) & (pl.col("y") == 1)).sum() * 1000 / ntr).alias("O_accTP"), pl.col("p").mean().alias("O_mp"))
        b = a.join(b, on=["side", "dist"], how="full", coalesce=True)
    print(f"== {c}")
    print(b.sort(["side", "dist"]))
x = te2.filter((pl.col("country") == "US") & (pl.col("side") == "on_R") & (pl.col("dist").is_in(["a<=2", "b3-12"])) & (pl.col("p") > 0.7))
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["name_raw", "addr_raw"])
r = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_r.parquet"), columns=["name_raw", "addr_raw"])
print("US test accepted on_R near samples:")
for row in x.sample(15, seed=1).iter_rows(named=True):
    print(f"  p={row['p']:.3f} S1: {s1['name_raw'][row['s1']]} | {s1['addr_raw'][row['s1']]}\n            R: {r['name_raw'][row['r']]} | {r['addr_raw'][row['r']]}")
