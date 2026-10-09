"""Per-word comparison of the exp13 role tables, train (dense world) vs test, same country: is e_lrate shifted, and is
the shift explained by the normalisation (per S1) rather than by the words' behaviour? Also the density-free
alternative: share of the country's near-duplicate events (n / sum n)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, gc
import numpy as np, polars as pl
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
import exp05_tokfeat as E5
import exp13_rolefeat as E13
C = _os.path.join(_WORK, 'data', 'cache')
out = {}
for split, src in [("train", os.path.join(C, "feat_v3", "k80s0")), ("test", os.path.join(C, "feat_v1", "test"))]:
    s1, r = E13.tables(split)
    te, tm, df = E13.role_tables(split, src, s1, r, lambda m: None)
    # raw counts back from lrate: n = 10**lrate * N1 - 1
    n1 = s1.group_by("country").len("N1")
    te = te.join(n1, on="country").with_columns((10 ** pl.col("e_lrate") * pl.col("N1") - 1).round().alias("n"))
    te = te.with_columns((pl.col("n") / pl.col("n").sum().over("country")).alias("share"))
    out[split] = te.select("country", "t", "n", "N1", "e_lrate", "share", "e_swap", "e_hneq")
    print(split, te.group_by("country").agg(pl.col("n").sum().alias("events"), pl.col("N1").first()).sort("country").rows(), flush=True)
    del s1, r, te, tm, df; gc.collect()
j = out["train"].join(out["test"], on=["country", "t"], suffix="_te").filter((pl.col("n") >= 100) & (pl.col("n_te") >= 100))
res = j.group_by("country").agg(pl.len().alias("words"),
                                (pl.col("e_lrate_te") - pl.col("e_lrate")).median().alias("lrate_shift_median"),
                                (pl.col("share_te") / pl.col("share")).log10().median().alias("share_shift_median"),
                                (pl.col("e_swap_te") - pl.col("e_swap")).median().alias("swap_shift"),
                                (pl.col("e_hneq_te") - pl.col("e_hneq")).median().alias("hneq_shift"))
pl.Config.set_tbl_width_chars(200)
print(res.sort("country"))
print(j.filter(pl.col("country") == "US").sort("n", descending=True).head(8).select("t", "n", "n_te", "e_lrate", "e_lrate_te", "e_swap", "e_swap_te", "e_hneq", "e_hneq_te"))
