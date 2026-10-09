"""Address-less R whose true S1 has a unique name (among kept S1), missing from candidates: what are they?"""
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
from rapidfuzz.distance import Levenshtein
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32))
res = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet")).select("s1", "r")
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country", "name_core", "name_raw", "addr_raw"]).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32))
r = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["name_core", "name_raw", "addr_clean"]).with_row_index("r").with_columns(pl.col("r").cast(pl.Int32))
s1k = s1.filter(pl.Series(keep))
cnt = s1k.group_by(["country", "name_core"]).len("nn")
g = gt.join(r.filter(pl.col("addr_clean") == ""), on="r").join(s1.select("s1", "country", pl.col("name_core").alias("n1"), pl.col("name_raw").alias("raw1")), on="s1")
g = g.join(cnt.rename({"name_core": "n1"}), on=["country", "n1"], how="left").filter(pl.col("nn") == 1)
g = g.join(res.with_columns(pl.lit(1).alias("c")), on=["s1", "r"], how="left").filter(pl.col("c").is_null())
print("address-less, unique true-S1 name, not in candidates:", g.height)
# how does R's name_core relate to S1's name_core?
ed = [Levenshtein.distance(a, b) for a, b in zip(g["name_core"].to_list(), g["n1"].to_list())]
g = g.with_columns(pl.Series("ed", ed))
print("edit distance name_core (R vs true S1): ", np.percentile(ed, [10, 25, 50, 75, 90]))
print("share R name_core == S1 name_core exactly:", (g["ed"] == 0).mean())
# how many kept S1 share R's own name_core exactly?
rc = g.join(cnt.rename({"name_core": "name_core"}), on=["country", "name_core"], how="left").with_columns(pl.col("nn_right").fill_null(0) if "nn_right" in g.columns else pl.lit(0))
print(rc.select(pl.col("nn_right").value_counts()).head(10) if "nn_right" in rc.columns else "")
for row in g.sample(25, seed=3).iter_rows(named=True):
    print(f"  ed={row['ed']:2d} S1: {row['raw1']}  <-  R: {row['name_raw']}")
