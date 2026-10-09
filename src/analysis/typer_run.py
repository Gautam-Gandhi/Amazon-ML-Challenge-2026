"""Size French uncertain pairs by fine type; true rate of the same types out of country (LOCO) by p bin."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from typer import typed, RUNS
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
BIN = pl.when(pl.col("p") <= 0.05).then(pl.lit("a<=.05")).when(pl.col("p") <= 0.3).then(pl.lit("b.05-.3")).when(pl.col("p") <= 0.6).then(pl.lit("c.3-.6")) \
        .when(pl.col("p") <= 0.8).then(pl.lit("d.6-.8")).when(pl.col("p") <= 0.9).then(pl.lit("e.8-.9")).otherwise(pl.lit("f>.9"))
te = pl.read_parquet(os.path.join(RUNS, "exp29gf", "test_pred.parquet"))
fr_idx = None
t = typed(te, "test")
t = t.filter(pl.col("country") == "France").with_columns(BIN.alias("bin"))
t.write_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fr_typed.parquet"))
g = t.filter((pl.col("p") > 0.05) & (pl.col("p") <= 0.9)).group_by(["ntype", "num"]).agg(pl.len().alias("n_rej"), pl.col("p").mean().alias("mp")).sort("n_rej", descending=True)
print("FRANCE rejected band (0.05,0.9] by type (R's best S1):")
print(g.head(40))
for f in ["base_US_India", "base_India_US"]:
    pred = pl.read_parquet(os.path.join(RUNS, "loco_pred13", f + ".parquet"))
    lt = typed(pred.select("s1", "r", "p", "y"), "train").with_columns(BIN.alias("bin"))
    lt.write_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"loco_typed_{f}.parquet"))
    g = lt.filter(pl.col("p") > 0.05).group_by(["ntype", "num", "bin"]).agg(pl.len().alias("n"), pl.col("y").mean().alias("true")) \
          .pivot(on="bin", index=["ntype", "num"], values="true", sort_columns=True)
    n = lt.filter((pl.col("p") > 0.05) & (pl.col("p") <= 0.9)).group_by(["ntype", "num"]).len("n_band")
    print(f"LOCO {f}: true rate by p bin (R's best S1)")
    print(g.join(n, on=["ntype", "num"], how="left").sort("n_band", descending=True, nulls_last=True).head(40))
