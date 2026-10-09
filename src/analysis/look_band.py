import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import sys
import numpy as np, polars as pl
W = _WORK
country = sys.argv[1]
lo, hi = float(sys.argv[2]), float(sys.argv[3])
n = int(sys.argv[4]); seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
s1 = pl.read_parquet(W + r"\data\cache\prep_v1\test_s1.parquet", columns=["eid", "country", "name_raw", "addr_raw"])
r = pl.read_parquet(W + r"\data\cache\prep_v1\test_r.parquet", columns=["eid", "name_raw", "addr_raw"])
pred = pl.read_parquet(W + r"\runs\exp29gf\test_pred.parquet")
a = pred.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
ctry = s1["country"].to_numpy()
a = a.filter(pl.Series(ctry[a["s1"].to_numpy()] == country)).filter((pl.col("p") > lo) & (pl.col("p") <= hi))
print(country, lo, hi, "assigned pairs:", a.height)
smp = a.sample(n, seed=seed)
for row in smp.iter_rows(named=True):
    s, ri = row["s1"], row["r"]
    # other S1 candidates of this r
    oth = pred.filter((pl.col("r") == ri) & (pl.col("s1") != s)).sort("p", descending=True).head(2)
    print(f"p={row['p']:.2f} S1: {s1['name_raw'][s]} | {s1['addr_raw'][s]}")
    print(f"        R: {r['name_raw'][ri]} | {r['addr_raw'][ri]}")
    for o in oth.iter_rows(named=True):
        print(f"     alt p={o['p']:.2f} S1: {s1['name_raw'][o['s1']]} | {s1['addr_raw'][o['s1']]}")
