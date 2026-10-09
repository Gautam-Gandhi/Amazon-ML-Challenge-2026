import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import sys, os
import numpy as np, polars as pl
W = _WORK
country = sys.argv[1] if len(sys.argv) > 1 else "France"
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 12
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
s1 = pl.read_parquet(W + r"\data\cache\prep_v1\test_s1.parquet", columns=["eid", "country", "name_raw", "addr_raw"])
r = pl.read_parquet(W + r"\data\cache\prep_v1\test_r.parquet", columns=["eid", "name_raw", "addr_raw"])
pred = pl.read_parquet(W + r"\runs\exp29gf\test_pred.parquet")
# best S1 per r
pred = pred.with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("rk"))
ctry = s1["country"].to_numpy()
pred = pred.filter(pl.Series(ctry[pred["s1"].to_numpy()] == country))
rng = np.random.default_rng(seed)
# pick S1s that have at least one pair in uncertain band
unc = pred.filter((pl.col("p") > 0.1) & (pl.col("p") < 0.95))["s1"].unique().to_numpy()
pick = rng.choice(unc, n_show, replace=False)
for s in pick:
    print("=" * 110)
    print(f"S1 {s1['eid'][int(s)]} | {s1['name_raw'][int(s)]} | {s1['addr_raw'][int(s)]}")
    sub = pred.filter(pl.col("s1") == s).sort("p", descending=True)
    for row in sub.iter_rows(named=True):
        ri = row["r"]
        print(f"   p={row['p']:.3f} rk={row['rk']} {r['eid'][ri]:>14} | {r['name_raw'][ri]} | {r['addr_raw'][ri]}")
