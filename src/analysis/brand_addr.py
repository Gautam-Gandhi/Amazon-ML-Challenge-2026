"""Brand-name records at the same house number: true rate by whether the S1 address is unique among S1s (same country),
in-country OOF, LOCO both directions, and French counts."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os
import numpy as np, polars as pl
SP = os.path.dirname(os.path.abspath(__file__))
CACHE = _os.path.join(_WORK, 'data', 'cache')
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
def addr_unique(split):
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    s = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=["country", "addr_clean"]).with_row_index("s1")
    c = s.group_by(["country", "addr_clean"]).len("n_addr")
    return s.join(c, on=["country", "addr_clean"]).sort("s1")["n_addr"].to_numpy()
BIN = pl.when(pl.col("p") <= 0.3).then(pl.lit("a.05-.3")).when(pl.col("p") <= 0.6).then(pl.lit("b.3-.6")).when(pl.col("p") <= 0.8).then(pl.lit("c.6-.8")).when(pl.col("p") <= 0.9).then(pl.lit("d.8-.9")).otherwise(pl.lit("e>.9"))
na_tr = addr_unique("train"); na_te = addr_unique("test")
for tag, f in [("OOF in-country", "oof_typed.parquet"), ("LOCO US->India", "loco_typed_base_US_India.parquet"), ("LOCO India->US", "loco_typed_base_India_US.parquet")]:
    t = pl.read_parquet(os.path.join(SP, f)).filter((pl.col("ntype") == "brand") & (pl.col("num") == "samenum") & (pl.col("p") > 0.05))
    t = t.with_columns(pl.Series("uniq_addr", na_tr[t["s1"].to_numpy()] == 1), BIN.alias("bin"))
    g = t.group_by(["uniq_addr", "bin"]).agg(pl.len().alias("n"), pl.col("y").mean().alias("true")).sort(["uniq_addr", "bin"])
    print("==", tag); print(g.pivot(on="bin", index="uniq_addr", values="true", sort_columns=True)); print(g.pivot(on="bin", index="uniq_addr", values="n", sort_columns=True))
fr = pl.read_parquet(os.path.join(SP, "fr_typed.parquet")).filter((pl.col("ntype") == "brand") & (pl.col("num") == "samenum") & (pl.col("p") > 0.05))
fr = fr.with_columns(pl.Series("uniq_addr", na_te[fr["s1"].to_numpy()] == 1), BIN.alias("bin"))
print("== France test counts"); print(fr.group_by(["uniq_addr", "bin"]).len().pivot(on="bin", index="uniq_addr", values="len", sort_columns=True))
