"""How much can an exact-address key recover? (dense world, labels)"""
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
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32))
res = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
acc_r = a.filter(pl.col("p") > 0.7).select("r").with_columns(pl.lit(True).alias("r_acc"))
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country", "addr_clean", "addr_nums", "name_core"]).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32))
r = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["country", "addr_clean", "addr_nums", "name_core"]).with_row_index("r").with_columns(pl.col("r").cast(pl.Int32))
s1k = s1.filter(pl.Series(keep))
for keyname, key in [("addr_clean", ["country", "addr_clean"])]:
    sk = s1k.filter(pl.col("addr_clean") != "").select(key + ["s1"])
    nsame = sk.group_by(key).len("n_s1_addr")
    rk = r.filter(pl.col("addr_clean") != "").select(key + ["r"])
    j = rk.join(sk, on=key).join(nsame, on=key)
    print(keyname, "join pairs", j.height, "per kept S1", j.height / keep.sum())
    j = j.join(gt.with_columns(pl.lit(1).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
    j = j.join(res.select("s1", "r", pl.col("p").alias("pc")), on=["s1", "r"], how="left")
    j = j.join(acc_r, on="r", how="left").with_columns(pl.col("r_acc").fill_null(False))
    print("  true rate all", j["y"].mean(), "n true", j["y"].sum())
    nc = j.filter(pl.col("pc").is_null())
    print("  not in candidates:", nc.height, "true", nc["y"].sum(), "rate", nc["y"].mean())
    for u in [1, 2]:
        sub = nc.filter((pl.col("n_s1_addr") == u) & ~pl.col("r_acc"))
        print(f"   notcand & unique-addr S1 (n={u}) & R not accepted elsewhere: {sub.height} true {sub['y'].sum()} rate {sub['y'].mean():.4f}")
    sub = nc.filter((pl.col("n_s1_addr") == 1))
    print(f"   notcand & unique-addr S1 (any R status): {sub.height} true {sub['y'].sum()} rate {sub['y'].mean():.4f}")
    inc = j.filter(pl.col("pc").is_not_null())
    print("  in candidates:", inc.height, "true rate", inc["y"].mean())
    for lo, hi in [(0, 0.3), (0.3, 0.7), (0.7, 1.01)]:
        sub = inc.filter((pl.col("pc") > lo) & (pl.col("pc") <= hi))
        print(f"    p in ({lo},{hi}]: {sub.height} true {sub['y'].mean():.4f}; unique-addr & R not acc: ", end="")
        sub2 = sub.filter((pl.col("n_s1_addr") == 1) & ~pl.col("r_acc"))
        print(sub2.height, f"{sub2['y'].mean():.4f}")
