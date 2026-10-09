"""Is there residual signal to pick the right S1 for address-less R whose name_core is shared by >=2 kept S1s?"""
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
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32))
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country", "name_raw", "name_core", "name_clean", "name_sfx", "addr_raw"]).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32))
r = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["country", "name_raw", "name_core", "name_clean", "name_sfx", "addr_clean", "src"]).with_row_index("r").with_columns(pl.col("r").cast(pl.Int32))
s1k = s1.filter(pl.Series(keep))
# address-less true pairs whose S1 name_core is shared
t = gt.join(r.filter(pl.col("addr_clean") == "").select("r", "name_raw", "name_core", "name_clean", "name_sfx", "country", "src"), on="r")
t = t.join(s1k.select("s1", pl.col("name_core").alias("n1")), on="s1")
# all same-name_core kept S1s for the R's name_core (candidates by exact S1 name_core == S1 true name_core)
grp = s1k.group_by(["country", "name_core"]).agg(pl.col("s1").alias("sibs"), pl.len().alias("nsib"))
t = t.join(grp.rename({"name_core": "n1"}), on=["country", "n1"]).filter(pl.col("nsib") >= 2).filter(pl.col("nsib") <= 5)
print("address-less true pairs with 2..5 same-name S1:", t.height)
t = t.sample(min(40000, t.height), seed=0)
s1_raw = s1["name_raw"].to_list(); s1_clean = s1["name_clean"].to_list(); s1_sfx = s1["name_sfx"].to_list()
wins = {"raw_ratio": 0, "clean_ratio": 0, "sfx_eq": 0, "raw_exact": 0}; ties = dict.fromkeys(wins, 0); n = 0
chance = 0.0
for row in t.iter_rows(named=True):
    sibs = row["sibs"]; true = row["s1"]
    chance += 1 / len(sibs)
    for key in wins:
        if key == "raw_ratio":
            sc = [fuzz.ratio(row["name_raw"], s1_raw[s]) for s in sibs]
        elif key == "clean_ratio":
            sc = [fuzz.ratio(row["name_clean"], s1_clean[s]) for s in sibs]
        elif key == "sfx_eq":
            sc = [float((row["name_sfx"] or "") == (s1_sfx[s] or "")) for s in sibs]
        else:
            sc = [float(row["name_raw"].lower() == s1_raw[s].lower()) for s in sibs]
        sc = np.array(sc); tv = sc[sibs.index(true)]
        if (sc == tv).all():
            ties[key] += 1
        elif tv == sc.max() and (sc == tv).sum() == 1:
            wins[key] += 1
    n += 1
print("n", n, "chance of random pick", chance / n)
for k in wins:
    nt = n - ties[k]
    print(f"{k:12s}: all-tie {ties[k]/n:.3f}; among non-tie: true is unique max {wins[k]/max(nt,1):.3f} (n={nt})")
