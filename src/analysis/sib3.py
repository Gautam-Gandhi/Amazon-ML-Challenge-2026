"""p distributions in the sibling group: OOF TP/FP vs test accepted (US), for group-specific thresholds."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
os.environ["ER_WORK_DIR"] = _WORK
import exp03_dense as M3
CACHE = _os.path.join(_WORK, 'data', 'cache')
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
s1tr = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
s1te = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
to = pl.read_parquet(os.path.join(SP, "oof_typed.parquet")); te = pl.read_parquet(os.path.join(SP, "test_typed.parquet"))
dh = (pl.col("h1").str.extract(r"(\d+)").cast(pl.Int64, strict=False) - pl.col("h2").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).abs()
def grp(t):
    return t.filter((pl.col("ntype") == "identical") & (pl.col("num") == "diffnum") & (pl.col("f1") == "") & (pl.col("f2") != "")).with_columns(dh.alias("dh"))
to = grp(to); te = grp(te)
bins = [0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0001]
for c in ["US", "India"]:
    ntr = int((keep & (s1tr == c)).sum()); nte = int((s1te == c).sum())
    for lo, hi in [(0, 2), (3, 12), (13, 100), (101, 10**9)]:
        o = to.filter((pl.col("country") == c) & (pl.col("dh") >= lo) & (pl.col("dh") <= hi))
        t = te.filter((pl.col("country") == c) & (pl.col("dh") >= lo) & (pl.col("dh") <= hi))
        otp = np.histogram(o.filter(pl.col("y") == 1)["p"].to_numpy(), bins=bins)[0] * 1000 / ntr
        ofp = np.histogram(o.filter(pl.col("y") == 0)["p"].to_numpy(), bins=bins)[0] * 1000 / ntr
        tt = np.histogram(t["p"].to_numpy(), bins=bins)[0] * 1000 / nte
        print(f"{c} dh {lo}-{hi}: per 1K S1 by p bin {bins}")
        print("   OOF TP  ", " ".join(f"{x:6.2f}" for x in otp))
        print("   OOF FP  ", " ".join(f"{x:6.2f}" for x in ofp))
        print("   TEST all", " ".join(f"{x:6.2f}" for x in tt), "  (excess over OOF TP+FP:", " ".join(f"{x:6.2f}" for x in tt - otp - ofp), ")")
