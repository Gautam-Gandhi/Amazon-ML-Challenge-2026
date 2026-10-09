"""Step 4 audit, with labels (dense world): stage-1 OOF with ALL trees (what test currently gets) vs the stored OOF
at the best iteration. Filter recall at 0.003 / 0.01, pairs kept, stage-1 assign F0.5, calibration of the stored OOF."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, glob, argparse
import numpy as np, polars as pl, xgboost as xgb
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
import exp01_match as M1
import exp03_dense as M3
from er_common import s1_fold
C = _os.path.join(_WORK, 'data', 'cache')
R = _os.path.join(_WORK, 'runs')
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0)); n1 = len(keep)
fold = s1_fold(n1)
bs = []
for k in (0, 1):
    b = xgb.Booster(); b.load_model(os.path.join(R, "exp13", f"stage1_fold{k}.json")); bs.append(b)
F = bs[0].feature_names
oof = pl.read_parquet(os.path.join(R, "exp13", "s1_oof.parquet"))
layers = [sorted(glob.glob(os.path.join(d, "part*.parquet"))) for d in
          [os.path.join(C, "feat_v3", "k80s0"), os.path.join(C, "feat_v7", "train"), os.path.join(C, "feat_v9", "train"), os.path.join(C, "feat_v13", "train")]]
pa, st = [], 0
for i in range(len(layers[0])):
    X = pl.concat([pl.read_parquet(f) for f in (L[i] for L in layers)], how="horizontal")
    s1 = X["s1"].to_numpy(); r = X["r"].to_numpy()
    o = oof.slice(st, X.height); st += X.height
    assert (o["s1"].to_numpy() == s1).all() and (o["r"].to_numpy() == r).all()
    M = X.select([pl.col(c).cast(pl.Float32) for c in F]).to_numpy(); del X
    f = fold[s1]; p = np.zeros(len(M), np.float32)
    for k in (0, 1):
        m = f == k
        p[m] = bs[1 - k].inplace_predict(M[m])
    del M
    pa.append(p)
    print(f"part {i} done", flush=True)
assert st == oof.height
res = oof.with_columns(pl.Series("p_all", np.concatenate(pa)))
gt = M3.labels_kept(keep); npos = gt.height
print(f"pairs {res.height:,}, true pairs of kept S1 {npos:,}, of which candidates {int(res['y'].sum()):,}")
for t in [0.001, 0.003, 0.01, 0.03]:
    for col in ["p", "p_all"]:
        kept = res.filter(pl.col(col) > t)
        print(f"  filter {t} [{'best iter' if col == 'p' else 'all trees'}]: kept {kept.height:,} ({kept.height / keep.sum():.3f} per S1), "
              f"recall {int(kept['y'].sum()) / npos:.5f}")
for col in ["p", "p_all"]:
    r2 = res.select("s1", "r", pl.col(col).alias("p"))
    a = r2.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    out = []
    for thr in [0.7, 0.8, 0.85]:
        d = a.filter(pl.col("p") > thr)
        out.append(f"{thr}:{M3.f05_world(d['s1'].to_numpy(), d['r'].to_numpy(), gt, keep):.5f}")
    ll = -np.mean(res["y"].to_numpy() * np.log(np.clip(res[col].to_numpy(), 1e-7, 1)) + (1 - res["y"].to_numpy()) * np.log(np.clip(1 - res[col].to_numpy(), 1e-7, 1)))
    print(f"  stage-1 assign F0.5 [{'best iter' if col == 'p' else 'all trees'}] {' '.join(out)}   OOF logloss {ll:.5f}")
d = np.abs(res["p"].to_numpy() - res["p_all"].to_numpy())
print(f"  |p_best - p_all| mean {d.mean():.2e} max {d.max():.3f}")
# calibration of the stored OOF (best iteration)
cal = res.with_columns((pl.col("p") * 10).floor().clip(0, 9).alias("bin")).group_by("bin").agg(
    pl.len().alias("n"), pl.col("p").mean().alias("mean_p"), pl.col("y").mean().alias("rate")).sort("bin")
print("calibration (best iter):", [(int(b), n, round(mp, 3), round(rt, 3)) for b, n, mp, rt in cal.rows()])
