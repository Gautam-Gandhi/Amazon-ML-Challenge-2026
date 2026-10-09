"""Step 4 audit: (1) do test predictions use more trees than the OOF predictions (default inplace_predict vs
iteration_range up to best_iteration)? (2) how much do the two fold models disagree on test (averaging effect)?
Sample: 1M rows from 4 test parts (France / India / US mix)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, glob
import numpy as np, polars as pl, xgboost as xgb
C = _os.path.join(_WORK, 'data', 'cache')
R = _os.path.join(_WORK, 'runs')
print("xgboost", xgb.__version__)
bs = []
for k in (0, 1):
    b = xgb.Booster(); b.load_model(os.path.join(R, "exp13", f"stage1_fold{k}.json")); bs.append(b)
    print(f"fold{k}: trees {b.num_boosted_rounds()}, best_iteration {b.best_iteration}, attrs {b.attributes()}")
F = bs[0].feature_names
cte = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
parts = [sorted(glob.glob(os.path.join(d, "part*.parquet"))) for d in
         [os.path.join(C, "feat_v1", "test"), os.path.join(C, "feat_v7", "test"), os.path.join(C, "feat_v9", "test"), os.path.join(C, "feat_v13", "test")]]
rng = np.random.default_rng(0)
P = {"all0": [], "best0": [], "all1": [], "best1": [], "ctry": []}
for i in [0, 4, 8, 11]:
    n = pl.scan_parquet(parts[0][i]).select(pl.len()).collect().item()
    idx = np.sort(rng.choice(n, 250_000, replace=False))
    X = pl.concat([pl.read_parquet(f)[idx] for f in (p[i] for p in parts)], how="horizontal")
    P["ctry"].append(cte[X["s1"].to_numpy()])
    M = X.select([pl.col(c).cast(pl.Float32) for c in F]).to_numpy(); del X
    for k, b in enumerate(bs):
        P[f"all{k}"].append(b.inplace_predict(M))
        P[f"best{k}"].append(b.inplace_predict(M, iteration_range=(0, b.best_iteration + 1)))
    del M
P = {k: np.concatenate(v) for k, v in P.items()}
print("countries in sample:", {c: int((P['ctry'] == c).sum()) for c in np.unique(P['ctry'])})
avg_all = (P["all0"] + P["all1"]) / 2; avg_best = (P["best0"] + P["best1"]) / 2
d = np.abs(avg_all - avg_best)
print(f"(1) all trees vs best iteration (fold average): mean|dp| {d.mean():.2e}, p99 {np.quantile(d, 0.99):.2e}, max {d.max():.3f}")
for t in [0.003, 0.01, 0.5, 0.7, 0.9]:
    print(f"    crossing {t}: {int(((avg_all > t) != (avg_best > t)).sum())} of {len(d)}   "
          f"(above: all-trees {int((avg_all > t).sum())}, best {int((avg_best > t).sum())})")
lo = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
m = (avg_best > 0.01) & (avg_best < 0.99)
print(f"    logit shift on the uncertain band (0.01-0.99, n={m.sum()}): mean {np.mean(lo(avg_all[m]) - lo(avg_best[m])):+.3f}, "
      f"mean |.| {np.mean(np.abs(lo(avg_all[m]) - lo(avg_best[m]))):.3f}")
dd = np.abs(P["best0"] - P["best1"])
print(f"(2) fold0 vs fold1 (best iteration): mean|dp| {dd.mean():.2e}, p99 {np.quantile(dd, 0.99):.2e}; "
      f"on band 0.01-0.99: mean {dd[m].mean():.3f}, median {np.median(dd[m]):.3f}")
for t in [0.003, 0.01]:
    a = avg_best > t; s0 = P["best0"] > t; s1 = P["best1"] > t
    print(f"    filter {t}: kept by average {a.sum()}, by fold0 alone {s0.sum()}, by fold1 alone {s1.sum()}, by either {(s0 | s1).sum()}")
