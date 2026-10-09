"""Step 5 audit (exp15 stage 2 on the filtered candidates).
(1) tree count: OOF with all trees (= what test gets) vs stored best-iteration OOF, with labels
(2) test: stored test_pred (all trees) vs best iteration
(3) the 18 stage-2 features, train vs test, on exact copies that are the record's best candidate (US, India)
(4) France: share of kept pairs that only the looser 0.003 threshold admits, and their effect on p-competition features"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, glob, argparse
import numpy as np, polars as pl, xgboost as xgb
WT = _SRC
sys.path.insert(0, WT); os.environ["ER_WORK_DIR"] = _WORK
import exp01_match as M1, exp03_dense as M3, exp05_tokfeat as E5, exp15_compact as E15
from er_common import s1_fold, CACHE, RUNS
E15.use_layers(argparse.Namespace(test_feat="feat_v1/test"))
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0)); n1 = len(keep); fold = s1_fold(n1)
run = os.path.join(RUNS, "exp15")
bs = []
for k in (0, 1):
    b = xgb.Booster(); b.load_model(os.path.join(run, f"stage2_fold{k}.json")); bs.append(b)
    print(f"stage2 fold{k}: trees {b.num_boosted_rounds()}, best_iteration {b.best_iteration}")
F = bs[0].feature_names; NEW = F[124:] if len(F) == 142 else [c for c in F if c.startswith(("p", "cons_"))]
print("features", len(F), "stage-2 additions:", NEW)
oof = pl.read_parquet(os.path.join(run, "oof.parquet"))
# (1)
pa, drift_tr, st = [], [], 0
ctr = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
cte = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
def exact(part, cmap):
    return part.filter((pl.col("nc_ratio") == 100) & (pl.col("ad_ratio") == 100) & (pl.col("r_rank_j") == 1)) \
               .with_columns(pl.Series("country", cmap[part.filter((pl.col("nc_ratio") == 100) & (pl.col("ad_ratio") == 100) & (pl.col("r_rank_j") == 1))["s1"].to_numpy()])) \
               .select(["country"] + NEW)
for part in E5.iter_s2("train")():
    s1 = part["s1"].to_numpy(); o = oof.slice(st, part.height); st += part.height
    assert (o["s1"].to_numpy() == s1).all() and (o["r"].to_numpy() == part["r"].to_numpy()).all()
    X = part.select([pl.col(c).cast(pl.Float32) for c in F]).to_numpy(); f = fold[s1]; p = np.zeros(len(X), np.float32)
    for k in (0, 1):
        p[f == k] = bs[1 - k].inplace_predict(X[f == k])
    pa.append(p); drift_tr.append(exact(part, ctr))
res = oof.with_columns(pl.Series("p_all", np.concatenate(pa)))
gt = M3.labels_kept(keep)
for col in ["p", "p_all"]:
    a = res.select("s1", "r", pl.col(col).alias("q"))
    a = a.filter(pl.col("q") == pl.col("q").max().over("r")).unique(subset=["r"], keep="first")
    y = res["y"].to_numpy(); q = np.clip(res[col].to_numpy(), 1e-7, 1 - 1e-7)
    out = [f"{t}:{M3.f05_world(a.filter(pl.col('q') > t)['s1'].to_numpy(), a.filter(pl.col('q') > t)['r'].to_numpy(), gt, keep):.5f}" for t in [0.7, 0.75, 0.8]]
    print(f"(1) stage-2 OOF [{'best iteration' if col == 'p' else 'all trees'}] F0.5 {' '.join(out)}  logloss {-np.mean(y*np.log(q)+(1-y)*np.log(1-q)):.5f}")
d = np.abs(res["p"].to_numpy() - res["p_all"].to_numpy()); print(f"    |dp| mean {d.mean():.2e} max {d.max():.3f}")
# (2)+(3) test
tp = pl.read_parquet(os.path.join(run, "test_pred.parquet"))
pb, drift_te, st = [], [], 0
for part in E5.iter_s2("test")():
    X = part.select([pl.col(c).cast(pl.Float32) for c in F]).to_numpy()
    pb.append(np.mean([b.inplace_predict(X, iteration_range=(0, b.best_iteration + 1)) for b in bs], axis=0))
    drift_te.append(exact(part, cte))
    if st == 0:
        pass
    st += part.height
pb = np.concatenate(pb); pall = tp["p"].to_numpy()
assert len(pb) == len(pall)
for t in [0.5, 0.7, 0.75, 0.9]:
    print(f"(2) test crossing {t}: {int(((pall > t) != (pb > t)).sum()):,} of {len(pb):,} (above: all-trees {int((pall > t).sum()):,}, best {int((pb > t).sum()):,})")
# (3)
tr = pl.concat(drift_tr); te = pl.concat(drift_te)
rows = []
for c in ["US", "India", "France"]:
    a_ = tr.filter(pl.col("country") == (c if c != "France" else "US")); b_ = te.filter(pl.col("country") == c)
    for f in NEW:
        x = a_[f].cast(pl.Float64).to_numpy(); z = b_[f].cast(pl.Float64).to_numpy()
        x2, z2 = x[~np.isnan(x)], z[~np.isnan(z)]
        if len(x2) < 100 or len(z2) < 100:
            continue
        sd = np.sqrt((x2.var() + z2.var()) / 2) + 1e-9
        rows.append({"test country": c, "feature": f, "smd": (z2.mean() - x2.mean()) / sd, "null_tr": np.isnan(x).mean(), "null_te": np.isnan(z).mean(),
                     "mean_tr": x2.mean(), "mean_te": z2.mean()})
dd = pl.DataFrame(rows)
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200); pl.Config.set_float_precision(3)
print(f"(3) exact-copy best-candidate pairs: train US {tr.filter(pl.col('country')=='US').height:,} India {tr.filter(pl.col('country')=='India').height:,}; "
      f"test US {te.filter(pl.col('country')=='US').height:,} India {te.filter(pl.col('country')=='India').height:,} France {te.filter(pl.col('country')=='France').height:,}")
print(dd.filter((pl.col("smd").abs() > 0.1) | ((pl.col("null_tr") - pl.col("null_te")).abs() > 0.02)))
# (4) France: pairs admitted only by 0.003
fr = tp.with_columns(pl.Series("c", cte[tp["s1"].to_numpy()])).filter(pl.col("c") == "France")
s2 = sorted(glob.glob(os.path.join(E15.S2DIR15, "test", "part*.parquet")))
p1 = pl.concat([pl.read_parquet(p, columns=["p1"]) for p in s2])["p1"].to_numpy()
cc = cte[tp["s1"].to_numpy()]
for c in ["US", "India", "France"]:
    m = cc == c
    print(f"(4) {c}: kept pairs {m.sum():,}; with p1 <= 0.01 {(p1[m] <= 0.01).sum():,}; of those final stage-2 p > 0.5: {((p1[m] <= 0.01) & (pall[m] > 0.5)).sum():,}")
