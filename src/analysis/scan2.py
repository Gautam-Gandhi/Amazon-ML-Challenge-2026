import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os
import polars as pl
SP = os.path.dirname(os.path.abspath(__file__))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(30)
m = pl.read_parquet(os.path.join(SP, "scan.parquet"))
G = ["country", "ntype", "num", "side", "dist"]
tot = m.group_by(G).agg(pl.col("otp").sum(), pl.col("ofp").sum(), pl.col("tacc").sum(), pl.col("excess").sum())
for c in ["US", "India"]:
    t = tot.filter(pl.col("country") == c)
    print(c, "sum excess", t["excess"].sum(), " positive part", t.filter(pl.col("excess") > 0)["excess"].sum(), " negative part", t.filter(pl.col("excess") < 0)["excess"].sum())
    print(t.sort("excess", descending=True).head(15))
    print(t.sort("excess").head(10))
# by coarse ntype
print(tot.group_by(["country", "ntype"]).agg(pl.col("excess").sum(), pl.col("otp").sum()).sort(["country", "excess"], descending=[False, True]))
print(tot.group_by(["country", "num"]).agg(pl.col("excess").sum(), pl.col("otp").sum()).sort(["country", "excess"], descending=[False, True]))
