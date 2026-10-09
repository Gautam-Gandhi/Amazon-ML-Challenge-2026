"""US swap1 samenum same-sfx pairs: which words are swapped, by p band, OOF (with labels) vs test."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
SP = os.path.dirname(os.path.abspath(__file__))
CACHE = _os.path.join(_WORK, 'data', 'cache')
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
country = sys.argv[1] if len(sys.argv) > 1 else "US"
def words(t, split):
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    a = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=["name_core"])["name_core"]
    b = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=["name_core"])["name_core"]
    A = a.gather(t["s1"].to_numpy()).str.split(" ").list.unique()
    B = b.gather(t["r"].to_numpy()).str.split(" ").list.unique()
    return t.with_columns(A.list.set_difference(B).list.first().alias("miss"), B.list.set_difference(A).list.first().alias("extra"))
to = pl.read_parquet(os.path.join(SP, "oof_typed.parquet")); te = pl.read_parquet(os.path.join(SP, "test_typed.parquet"))
sel = (pl.col("country") == country) & (pl.col("ntype") == "swap1") & (pl.col("num") == "samenum") & (pl.col("f1") == pl.col("f2"))
to = words(to.filter(sel), "train"); te = words(te.filter(sel), "test")
for lo, hi in [(0.7, 0.98), (0.98, 1.01), (0.0, 0.05)]:
    o = to.filter((pl.col("p") > lo) & (pl.col("p") <= hi)); t = te.filter((pl.col("p") > lo) & (pl.col("p") <= hi))
    print(f"==== p in ({lo},{hi}]  OOF n={o.height} (true {o['y'].mean() if o.height else 0:.3f})  TEST n={t.height}")
    print("OOF top extra words:", o.group_by("extra").agg(pl.len(), pl.col("y").mean().alias("true")).sort("len", descending=True).head(12).rows())
    print("TEST top extra words:", t.group_by("extra").len().sort("len", descending=True).head(15).rows())
    print("TEST top missing words:", t.group_by("miss").len().sort("len", descending=True).head(12).rows())
