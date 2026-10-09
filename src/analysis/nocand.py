"""What are the test records with no candidate (after the compact filter)? Look up same-name / same-address S1s."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import sys
import numpy as np, polars as pl
W = _WORK
country = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3])
s1 = pl.read_parquet(W + r"\data\cache\prep_v2\test_s1.parquet", columns=["eid", "country", "name_raw", "addr_raw", "name_core", "addr_clean", "addr_nums"]).with_row_index("s1")
r = pl.read_parquet(W + r"\data\cache\prep_v2\test_r.parquet", columns=["eid", "country", "name_raw", "addr_raw", "name_core", "addr_clean", "addr_nums"]).with_row_index("r")
pred = pl.read_parquet(W + r"\runs\exp29gf\test_pred.parquet")
has = np.zeros(r.height, bool); has[pred["r"].unique().to_numpy()] = True
rc = r.filter(pl.Series(~has) & (pl.col("country") == country))
print(country, "no-cand records:", rc.height)
s1c = s1.filter(pl.col("country") == country)
# exact name_core matches
nm = rc.join(s1c.select("name_core", pl.col("s1")), on="name_core", how="inner")
print("  no-cand R with an exact name_core S1:", nm["r"].n_unique(), f"({nm['r'].n_unique()/rc.height:.3f})")
# same first number + same addr_clean
am = rc.filter(pl.col("addr_clean") != "").join(s1c.select("addr_clean", pl.col("s1")), on="addr_clean", how="inner")
print("  no-cand R with an exact addr_clean S1:", am["r"].n_unique(), f"({am['r'].n_unique()/rc.height:.3f})")
smp = rc.sample(n, seed=seed)
for row in smp.iter_rows(named=True):
    print(f"R: {row['name_raw']} | {row['addr_raw']}")
    m = s1c.filter(pl.col("name_core") == row["name_core"]).head(3)
    for q in m.iter_rows(named=True):
        print(f"     same-name S1: {q['name_raw']} | {q['addr_raw']}")
    if row["addr_clean"]:
        m = s1c.filter(pl.col("addr_clean") == row["addr_clean"]).head(3)
        for q in m.iter_rows(named=True):
            print(f"     same-addr S1: {q['name_raw']} | {q['addr_raw']}")
