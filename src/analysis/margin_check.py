"""Train (dense world) vs test: candidate S1s per record (r_ns1) and the runner-up margin (r_margin2), the top stage-1
feature. Compared on (a) all records' best pair and (b) confident matches (train: true pairs; test: p>0.9)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, glob, argparse
import numpy as np, polars as pl
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
import exp03_dense as M3
C = _os.path.join(_WORK, 'data', 'cache')
R = _os.path.join(_WORK, 'runs')
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32), pl.lit(1).alias("y"))
cols = ["s1", "r", "cos_j", "r_ns1", "r_margin2", "r_rank_j"]
tr = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(os.path.join(C, "feat_v3", "k80s0", "part*.parquet")))])
te = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(os.path.join(C, "feat_v1", "test", "part*.parquet")))])
ctr = pl.read_parquet(os.path.join(C, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
cte = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
tr = tr.with_columns(pl.Series("country", ctr[tr["s1"].to_numpy()])).join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
tp = pl.read_parquet(os.path.join(R, "exp29gf", "test_pred.parquet"))
te = te.with_columns(pl.Series("country", cte[te["s1"].to_numpy()])).join(tp, on=["s1", "r"], how="left")
q = [0.1, 0.25, 0.5, 0.75, 0.9]
def show(tag, d):
    b = d.filter(pl.col("r_rank_j") == 1)            # each record's best pair
    m = b["r_margin2"].drop_nulls().to_numpy()
    print(f"  {tag:34s} records {b.height:>9,} | S1 candidates per record: mean {b['r_ns1'].mean():.2f}, single-candidate {(b['r_ns1'] == 1).mean():.3f} "
          f"| r_margin2 quantiles {np.round(np.quantile(m, q), 3).tolist()}")
for c in ["US", "India"]:
    print(f"== {c}")
    show("train (dense world), all records", tr.filter(pl.col("country") == c))
    show("test, all records", te.filter(pl.col("country") == c))
    show("train, true-match records", tr.filter((pl.col("country") == c) & (pl.col("y") == 1)))
    show("test, confident-match records (p>0.9)", te.filter((pl.col("country") == c) & (pl.col("p").fill_null(0) > 0.9)))
