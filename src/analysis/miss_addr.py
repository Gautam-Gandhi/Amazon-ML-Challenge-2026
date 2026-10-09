"""Examples of with-address true pairs missing from the compact candidate set (dense world)."""
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
gt = M3.labels_kept(keep)
res = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet")).select("s1", "r")
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country", "name_raw", "addr_raw", "addr_clean"])
r = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["name_raw", "addr_raw", "addr_clean"])
g = gt.rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32))
miss = g.join(res.with_columns(pl.lit(1).alias("c")), on=["s1", "r"], how="left").filter(pl.col("c").is_null())
ra = (r["addr_clean"] != "").to_numpy()
miss = miss.filter(pl.Series(ra[miss["r"].to_numpy()]))
print("with-address misses:", miss.height)
# were they in the raw ANN candidates (cand_v1) before stage-1 filtering?
# how many does each R have in candidates at all?
rc = res.group_by("r").len("nc")
miss = miss.join(rc, on="r", how="left").with_columns(pl.col("nc").fill_null(0))
print("R has other candidates:", (miss["nc"] > 0).mean())
smp = miss.sample(30, seed=1)
for row in smp.iter_rows(named=True):
    s, ri = row["s1"], row["r"]
    print(f"[{s1['country'][s]}] S1: {s1['name_raw'][s]} | {s1['addr_raw'][s]}")
    print(f"      R: {r['name_raw'][ri]} | {r['addr_raw'][ri]}   (R cands {row['nc']})")
