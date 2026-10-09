"""Identical name_core, different first house number: split by number distance and legal form.
Train (dense OOF, labels) vs test (by country)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
sys.path.insert(0, _REPO)
os.environ["ER_WORK_DIR"] = _WORK
from er_common import RUNS, CACHE
import exp03_dense as M3


def typed(pred, split):
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    cols = ["country", "name_core", "addr_clean", "addr_nums", "name_sfx"]
    s1 = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=cols)
    r = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=cols[1:])
    a = pred.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    si, ri = a["s1"].to_numpy(), a["r"].to_numpy()
    a = a.with_columns(
        pl.Series("country", s1["country"].to_numpy()[si]),
        (s1["name_core"].gather(si) == r["name_core"].gather(ri)).alias("same_name"),
        s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null("").alias("h1"),
        r["addr_nums"].gather(ri).str.split(" ").list.first().fill_null("").alias("h2"),
        s1["name_sfx"].gather(si).fill_null("").alias("f1"), r["name_sfx"].gather(ri).fill_null("").alias("f2"),
        (r["addr_clean"].gather(ri) == "").alias("r_noaddr"))
    a = a.filter(pl.col("same_name") & (pl.col("h1") != "") & (pl.col("h2") != "") & (pl.col("h1") != pl.col("h2")))
    dh = (pl.col("h1").str.extract(r"(\d+)").cast(pl.Int64, strict=False) - pl.col("h2").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).abs()
    a = a.with_columns(dh.alias("dh"))
    a = a.with_columns(pl.when(pl.col("dh") <= 12).then(pl.lit("near")).otherwise(pl.lit("far")).alias("dist"),
                       pl.when(pl.col("f1") == pl.col("f2")).then(pl.lit("sfx_same"))
                       .when((pl.col("f1") == "") | (pl.col("f2") == "")).then(pl.lit("sfx_missing1"))
                       .otherwise(pl.lit("sfx_diff")).alias("sfx"))
    return a


keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32), pl.lit(1).alias("y"))
oof = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
tr = typed(oof, "train").join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
te = typed(pl.read_parquet(os.path.join(RUNS, "exp29gf", "test_pred.parquet")), "test")
nS1_tr = keep.sum()
s1c = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].value_counts()
nS1_te = dict(zip(s1c["country"].to_list(), s1c["count"].to_list()))
print("TRAIN (dense, US+India): per 1K S1 | true rate | mean p | acc@0.7")
g = tr.group_by(["dist", "sfx"]).agg(pl.len().alias("n"), pl.col("y").mean().alias("true"), pl.col("p").mean().alias("mp"), (pl.col("p") > 0.7).mean().alias("acc")).sort(["dist", "sfx"])
for row in g.iter_rows(named=True):
    print(f"  {row['dist']:5s} {row['sfx']:13s} n={row['n']:7d} ({1000*row['n']/nS1_tr:6.2f}/1K) true {row['true']:.3f} mp {row['mp']:.3f} acc {row['acc']:.3f}")
for c in ["US", "India", "France"]:
    thr = 0.9 if c == "France" else 0.7
    g = te.filter(pl.col("country") == c).group_by(["dist", "sfx"]).agg(pl.len().alias("n"), pl.col("p").mean().alias("mp"), (pl.col("p") > thr).mean().alias("acc")).sort(["dist", "sfx"])
    print(f"TEST {c}: per 1K S1 | mean p | acc@{thr}")
    for row in g.iter_rows(named=True):
        print(f"  {row['dist']:5s} {row['sfx']:13s} n={row['n']:7d} ({1000*row['n']/nS1_te[c]:6.2f}/1K) mp {row['mp']:.3f} acc {row['acc']:.3f}")
