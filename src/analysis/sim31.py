"""Monte Carlo estimate of exp31's LB effect: P(true) per demoted pair from count matching (OOF TP / test count
per (distance bucket, p bin)); other accepted pairs assumed true; F0.5 per affected S1 before/after."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
os.environ["ER_DATA_DIR"] = _DATA
import exp03_dense as M3
import exp31_sibprior as E
from er_common import CACHE, RUNS
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
cty = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
n_us_tr = int((keep & (cty == "US")).sum()); n_us_te = 663106
args = argparse.Namespace(countries="US", t_near=0.98, t_far=0.95, demote=0.5)
q = lambda *a: None
gt = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_gt.parquet")).select(pl.col("s1_idx").cast(pl.Int32).alias("s1"), pl.col("r_idx").cast(pl.Int32).alias("r"), pl.lit(1).alias("y"))
bins = np.array([0.7, 0.8, 0.9, 0.95, 0.98, 1.01])
def cells(pred, split):
    m, dn = E.sibling_mask(pred, split, ["US"])
    p = pred["p"].to_numpy()
    thr = np.where(dn <= 12, 0.98, 0.95)
    hit = m & (p <= thr) & (p > 0.7)
    db = np.digitize(dn, [2.5, 12.5, 100.5])  # 0: <=2, 1: 3-12, 2: 13-100, 3: >100
    pbn = np.digitize(p, bins) - 1
    return hit, db, pbn
oof = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
h, db, pbn = cells(oof, "train")
o = oof.with_columns(pl.Series("hit", h), pl.Series("db", db), pl.Series("pb", pbn)).filter("hit").join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
otp = o.group_by(["db", "pb"]).agg((pl.col("y").sum() / n_us_tr * 1000).alias("tp"))
te = pl.read_parquet(os.path.join(RUNS, "exp29gf", "test_pred.parquet"))
h, db, pbn = cells(te, "test")
t = te.with_columns(pl.Series("hit", h), pl.Series("db", db), pl.Series("pb", pbn))
d = t.filter("hit")
tc = d.group_by(["db", "pb"]).agg((pl.len() / n_us_te * 1000).alias("tn"))
pr = tc.join(otp, on=["db", "pb"], how="left").fill_null(0).with_columns((pl.col("tp") / pl.col("tn")).clip(0, 1).alias("ptrue"))
print(pr.sort(["db", "pb"]))
d = d.join(pr.select("db", "pb", "ptrue"), on=["db", "pb"])
# only demoted pairs that were ACCEPTED in exp29gf (assigned best S1, p > 0.7)
a = te.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first").filter(pl.col("p") > 0.7)
d = d.join(a.select("s1", "r"), on=["s1", "r"], how="semi")
print("demoted accepted pairs:", d.height, " expected true:", d["ptrue"].sum())
# per S1: n_acc (accepted in exp29gf), demoted list
nacc = a.group_by("s1").len("nacc")
g = d.group_by("s1").agg(pl.col("ptrue")).join(nacc, on="s1")
rng = np.random.default_rng(0)
def f05(tp, npred, ntrue):
    if ntrue == 0:
        return 1.0 if npred == 0 else 0.0
    if tp == 0:
        return 0.0
    P = tp / npred; R = tp / ntrue
    return 1.25 * P * R / (0.25 * P + R)
tot = []
for it in range(200):
    delta = 0.0
    for pt, n in zip(g["ptrue"].to_list(), g["nacc"].to_list()):
        k = len(pt)
        tr = rng.random(k) < np.array(pt)
        others = n - k                      # other accepted pairs, assumed true
        ntrue = others + tr.sum() + 0       # missed true matches ignored
        before = f05(others + tr.sum(), n, ntrue)
        after = f05(others, others, ntrue)
        delta += after - before
    tot.append(delta)
tot = np.array(tot)
print(f"sum of dF over affected S1: mean {tot.mean():.1f} (sd {tot.std():.1f}) -> LB delta {tot.mean()/1732544:+.6f}")
