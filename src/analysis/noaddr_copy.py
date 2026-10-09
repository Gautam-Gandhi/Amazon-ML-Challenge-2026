"""Do address-less records share their exact noisy name with another record of the SAME entity (derivation),
more than with records of a same-name sibling S1?  (train GT, full world)"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys
import numpy as np, polars as pl
sys.path.insert(0, _REPO)
os.environ["ER_WORK_DIR"] = _WORK
from er_common import CACHE
gt = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_gt.parquet")).rename({"s1_idx": "s1", "r_idx": "r"})
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country", "name_core", "name_raw"]).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32))
r = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["name_raw", "name_clean", "addr_clean", "src"]).with_row_index("r").with_columns(pl.col("r").cast(pl.Int32))
r = r.with_columns(pl.col("name_raw").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars().alias("nlow"))
g = gt.join(r.select("r", "nlow", "name_clean", "addr_clean", "src"), on="r").join(s1.select("s1", "country", "name_core"), on="s1")
# ambiguous address-less true pairs: S1 name_core shared by >= 2 S1 (same country)
cnt = s1.group_by(["country", "name_core"]).agg(pl.col("s1").alias("sibs"), pl.len().alias("nsib"))
na = g.filter(pl.col("addr_clean") == "").join(cnt, on=["country", "name_core"]).filter(pl.col("nsib") == 2)
print("address-less true pairs whose S1 name is shared by exactly 2 S1:", na.height)
# for each S1, the multiset of (nlow) among its WITH-address true records, and name_clean
wa = g.filter(pl.col("addr_clean") != "")
for key in ["nlow", "name_clean"]:
    k1 = wa.group_by(["s1", key]).len("cnt")
    x = na.select("r", "s1", "sibs", key).explode("sibs").rename({"sibs": "cand"})
    x = x.join(k1.rename({"s1": "cand"}), on=["cand", key], how="left").with_columns(pl.col("cnt").fill_null(0))
    x = x.with_columns((pl.col("cand") == pl.col("s1")).alias("is_true"))
    agg = x.group_by("r").agg(pl.col("cnt").filter(pl.col("is_true")).first().alias("c_true"), pl.col("cnt").filter(~pl.col("is_true")).first().alias("c_other"))
    n = agg.height
    tw = (agg["c_true"] > agg["c_other"]).sum(); ow = (agg["c_true"] < agg["c_other"]).sum(); tie = n - tw - ow
    print(f"[{key}] exact-copy count among each S1's with-address true records: true wins {tw} ({tw/n:.3f}), other wins {ow} ({ow/n:.3f}), tie {tie} ({tie/n:.3f})")
    print(f"   P(true has >=1 copy) {(agg['c_true']>0).mean():.3f}  P(other has >=1 copy) {(agg['c_other']>0).mean():.3f}")
