import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import numpy as np, polars as pl
W = _WORK
# test
s1 = pl.read_parquet(W + r"\data\cache\prep_v1\test_s1.parquet", columns=["country"])
r = pl.read_parquet(W + r"\data\cache\prep_v1\test_r.parquet", columns=["country", "addr_clean"])
pred = pl.read_parquet(W + r"\runs\exp29gf\test_pred.parquet")
bins = [0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.97, 0.99, 1.0001]
best = pred.group_by("r").agg(pl.col("p").max().alias("pm"))
rc = r["country"].to_numpy()
noaddr = (r["addr_clean"] == "").to_numpy()
pm = np.zeros(len(r), dtype=np.float32) - 1
pm[best["r"].to_numpy()] = best["pm"].to_numpy()
print("TEST: records per S1, and share of records by best-p bin (-1 = no candidate)")
for c in ["US", "India", "France"]:
    m = rc == c
    ns1 = (s1["country"] == c).sum()
    h = np.histogram(pm[m], bins=[-2, -0.5] + bins)[0] / m.sum()
    print(f"{c:7s} R/S1 {m.sum()/ns1:.3f} noaddr {noaddr[m].mean():.4f} | " + " ".join(f"{x:.4f}" for x in h))
    # accepted per S1 at thresholds (best S1 only)
    for t in [0.7, 0.9]:
        print(f"      share of R with best p>{t}: {(pm[m] > t).mean():.4f}; per S1 {(pm[m] > t).sum()/ns1:.3f}")
print("bins:", ["none"] + [f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])])
# train dense OOF: truth
oof = pl.read_parquet(W + r"\runs\exp29g\oof_combined.parquet")
print(oof.shape)
