"""exp31 demoted pairs: split by whether S1 and R number sets intersect (parse artefacts) - OOF TP rate and test counts."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
os.environ["ER_DATA_DIR"] = _DATA
import exp03_dense as M3
import exp31_sibprior as E
from er_common import CACHE, RUNS
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
args = argparse.Namespace(countries="US", t_near=0.98, t_far=0.95, demote=0.5)
log = print
def numsets(pred, split):
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    a = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=["addr_nums"])["addr_nums"]
    b = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=["addr_nums"])["addr_nums"]
    A = a.gather(pred["s1"].to_numpy()).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    B = b.gather(pred["r"].to_numpy()).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    return (A.list.set_intersection(B).list.len() > 0).to_numpy()
gt = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_gt.parquet")).select(pl.col("s1_idx").cast(pl.Int32).alias("s1"), pl.col("r_idx").cast(pl.Int32).alias("r"), pl.lit(1).alias("y"))
oof = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
_, hit = E.demote(oof, "train", args, log)
h = oof.filter(pl.Series(hit) & (pl.col("p") > 0.7))
h = h.with_columns(pl.Series("inter", numsets(h, "train"))).join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
print("OOF demoted accepted:", h.group_by("inter").agg(pl.len(), pl.col("y").mean().alias("true_rate")).rows())
te = pl.read_parquet(os.path.join(RUNS, "exp29gf", "test_pred.parquet"))
_, hit = E.demote(te, "test", args, log)
t = te.filter(pl.Series(hit) & (pl.col("p") > 0.7))
t = t.with_columns(pl.Series("inter", numsets(t, "test")))
print("TEST demoted accepted:", t.group_by("inter").len().rows())
nus_tr = int((keep & (pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy() == "US")).sum())
print("per 1K US S1: OOF", h.group_by("inter").len().with_columns((pl.col("len") * 1000 / nus_tr)).rows(), " TEST", t.group_by("inter").len().with_columns(pl.col("len") * 1000 / 663106).rows())
