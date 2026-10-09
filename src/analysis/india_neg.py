"""Negative excess groups: test vs OOF counts in p bins including the rejected range, and how many competing S1 the R has."""
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
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(30)
CACHE = _os.path.join(_WORK, 'data', 'cache')
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
s1tr = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
s1te = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
to = pl.read_parquet(os.path.join(SP, "oof_typed.parquet")); te = pl.read_parquet(os.path.join(SP, "test_typed.parquet"))
side = pl.when(pl.col("f1") == pl.col("f2")).then(pl.lit("same")).when(pl.col("f1") == "").then(pl.lit("on_R")).when(pl.col("f2") == "").then(pl.lit("on_S1")).otherwise(pl.lit("diff"))
pb = pl.when(pl.col("p") <= 0.05).then(pl.lit("a<.05")).when(pl.col("p") <= 0.3).then(pl.lit("b<.3")).when(pl.col("p") <= 0.5).then(pl.lit("c<.5")).when(pl.col("p") <= 0.7).then(pl.lit("d<.7")).when(pl.col("p") <= 0.9).then(pl.lit("e<.9")).when(pl.col("p") <= 0.98).then(pl.lit("f<.98")).otherwise(pl.lit("g>.98"))
to = to.with_columns(side.alias("side"), pb.alias("pb")); te = te.with_columns(side.alias("side"), pb.alias("pb"))
for c in ["India", "US"]:
    ntr = int((keep & (s1tr == c)).sum()); nte = int((s1te == c).sum())
    for (nt, nm, sd) in [("identical", "samenum", "same"), ("identical", "noaddr", "same"), ("swap1", "samenum", "same"), ("brand", "samenum", "on_S1")]:
        o = to.filter((pl.col("country") == c) & (pl.col("ntype") == nt) & (pl.col("num") == nm) & (pl.col("side") == sd)).group_by("pb").agg(
            (pl.len() * 1000 / ntr).alias("O_all"), (pl.col("y").sum() * 1000 / ntr).alias("O_tp"))
        t = te.filter((pl.col("country") == c) & (pl.col("ntype") == nt) & (pl.col("num") == nm) & (pl.col("side") == sd)).group_by("pb").agg((pl.len() * 1000 / nte).alias("T_all"))
        m = o.join(t, on="pb", how="full", coalesce=True).fill_null(0).sort("pb")
        print(f"== {c} {nt} {nm} {sd}: totals O_all {m['O_all'].sum():.2f} O_tp {m['O_tp'].sum():.2f} T_all {m['T_all'].sum():.2f}")
        print(m)
