"""LOCO: S1-level completion. At unseen threshold 0.9, for S1s with k accepted (assigned) pairs, true rate of their
assigned candidates with p in lower bands. Positive if a band's true rate exceeds the break-even (~0.7 for k>=1)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os
import numpy as np, polars as pl
RUNS = _os.path.join(_WORK, 'runs')
SP = os.path.dirname(os.path.abspath(__file__))
for f in ["base_US_India", "base_India_US"]:
    pred = pl.read_parquet(os.path.join(RUNS, "loco_pred13", f + ".parquet"))
    a = pred.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    k = a.filter(pl.col("p") > 0.9).group_by("s1").len("k")
    a = a.join(k, on="s1", how="left").with_columns(pl.col("k").fill_null(0))
    a = a.with_columns(pl.when(pl.col("k") >= 3).then(pl.lit("3+")).otherwise(pl.col("k").cast(pl.Utf8)).alias("kk"),
                       pl.when(pl.col("p") <= 0.5).then(pl.lit("a.3-.5")).when(pl.col("p") <= 0.7).then(pl.lit("b.5-.7"))
                       .when(pl.col("p") <= 0.8).then(pl.lit("c.7-.8")).when(pl.col("p") <= 0.9).then(pl.lit("d.8-.9")).otherwise(pl.lit("e>.9")).alias("bin"))
    x = a.filter(pl.col("p") > 0.3)
    g = x.group_by(["kk", "bin"]).agg(pl.len().alias("n"), pl.col("y").mean().alias("true"))
    print("==", f, " true rate of assigned pairs by S1 accepted-count k (at 0.9) and p bin")
    print(g.pivot(on="bin", index="kk", values="true", sort_columns=True).sort("kk"))
    print(g.pivot(on="bin", index="kk", values="n", sort_columns=True).sort("kk"))
