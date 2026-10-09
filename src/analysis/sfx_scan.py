"""Legal-form tokens on records: per 1K S1 in train vs test by country; and French SNC records vs their best S1."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os
import numpy as np, polars as pl
CACHE = _os.path.join(_WORK, 'data', 'cache')
RUNS = _os.path.join(_WORK, 'runs')
SP = os.path.dirname(os.path.abspath(__file__))
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(250)
rows = []
for split in ["train", "test"]:
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    s1 = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=["country"])
    r = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=["country", "name_sfx"])
    n1 = dict(zip(*[s1["country"].value_counts()[c].to_list() for c in ["country", "count"]]))
    x = r.select("country", pl.col("name_sfx").fill_null("").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "")
    g = x.group_by(["country", "t"]).len().with_columns(pl.struct(["country", "len"]).map_elements(lambda d: 1000 * d["len"] / n1[d["country"]], return_dtype=pl.Float64).alias(split))
    rows.append(g.select("country", "t", split))
m = rows[0].join(rows[1], on=["country", "t"], how="full", coalesce=True).fill_null(0).with_columns((pl.col("test") / (pl.col("train") + 0.01)).alias("ratio"))
print("record legal-form tokens per 1K S1, train vs test")
print(m.filter(pl.col("country").is_in(["US", "India"]) & ((pl.col("train") > 5) | (pl.col("test") > 5))).sort(["country", "ratio"], descending=[False, True]).head(40))
print(m.filter(pl.col("country") == "France").sort("test", descending=True).head(14))
# French SNC records vs their best S1
te = pl.read_parquet(os.path.join(SP, "test_typed.parquet"))
r2 = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_r.parquet"), columns=["name_sfx"])
te = te.with_columns(pl.Series("r_snc", r2["name_sfx"].fill_null("").str.contains("snc").to_numpy()[te["r"].to_numpy()]))
fs = te.filter((pl.col("country") == "France") & pl.col("r_snc"))
print("French SNC records assigned to a best S1:", fs.height, " accepted p>0.9:", (fs["p"] > 0.9).sum())
print(fs.group_by(["ntype", "num"]).agg(pl.len(), (pl.col("p") > 0.9).sum().alias("acc"), pl.col("p").mean().alias("mp")).sort("len", descending=True).head(12))
