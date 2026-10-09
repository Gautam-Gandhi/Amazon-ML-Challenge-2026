"""In-country blocking funnel with labels (dense world, kept S1): raw search -> pruning -> filter.
And: an extra keep rule 'rk_addr <= K' (S1's top-K by address similarity): true pairs recovered vs extra pairs."""
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
kept = pl.Series("s1", np.where(keep)[0].astype(np.int32))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32), pl.lit(1).cast(pl.Int8).alias("y"))
r = pl.read_parquet(os.path.join(C, "prep_v1", "train_r.parquet"), columns=["addr_clean"])
noaddr = (r["addr_clean"] == "").to_numpy()
pruned = pl.concat([pl.read_parquet(f, columns=["s1", "r"]) for f in sorted(glob.glob(os.path.join(C, "feat_v3", "k80s0", "part*.parquet")))]).with_columns(pl.lit(True).alias("pr"))
filt = pl.read_parquet(os.path.join(R, "exp29g", "oof_combined.parquet"), columns=["s1", "r"]).with_columns(pl.lit(True).alias("fl"))
n_kept = int(keep.sum())
tot = {"true": gt.height, "raw": 0, "pruned": 0, "filt": 0}
Ks = [1, 2, 3, 5]
extra = {k: 0 for k in Ks}; rec = {k: 0 for k in Ks}; rec_addr = {k: 0 for k in Ks}
lost_pr_addr = 0; lost_pr = 0
for f in sorted(glob.glob(os.path.join(C, "cand_v1", "train", "*.parquet"))):
    d = pl.read_parquet(f, columns=["s1", "r", "rk_addr", "rk_joint"]).filter(pl.col("s1").is_in(kept.implode()))
    d = d.join(pruned, on=["s1", "r"], how="left").join(gt, on=["s1", "r"], how="left").with_columns(pl.col("pr").fill_null(False), pl.col("y").fill_null(0))
    t = d.filter(pl.col("y") == 1)
    tot["raw"] += t.height
    tot["pruned"] += t["pr"].sum()
    lp = t.filter(~pl.col("pr"))
    lost_pr += lp.height
    lost_pr_addr += int((~noaddr[lp["r"].to_numpy()]).sum())
    for k in Ks:
        m = d.filter((pl.col("rk_addr") < k) & ~pl.col("pr"))
        extra[k] += m.height
        rec[k] += int(m["y"].sum())
        rec_addr[k] += int(((m["y"] == 1).to_numpy() & ~noaddr[m["r"].to_numpy()]).sum())
    print(os.path.basename(f), "done", flush=True)
tot["filt"] = gt.join(filt, on=["s1", "r"], how="inner").height
print(f"TRUE pairs (dense world, kept S1): {tot['true']:,}")
print(f"  found by raw search: {tot['raw']/tot['true']:.4f} | after pruning: {tot['pruned']/tot['true']:.4f} | after filter: {tot['filt']/tot['true']:.4f}")
print(f"  lost at pruning: {lost_pr:,} (with an address: {lost_pr_addr:,})")
for k in Ks:
    print(f"  extra rule 'S1 top-{k} by address': +{extra[k]:,} pairs ({extra[k]/n_kept:.2f} per S1), recovers {rec[k]:,} true pairs "
          f"({rec[k]/max(extra[k],1):.3f} of the extra are true; {rec_addr[k]:,} with an address)")
