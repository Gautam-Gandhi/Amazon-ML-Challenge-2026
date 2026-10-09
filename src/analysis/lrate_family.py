"""Direction of the lrate effects on test (stage-1 fold-0 model, 1M-row sample per part):
 A  : normalisation fix (train divided by kept S1)  == test lrate - log10(1/0.8)
 A+F: additionally remove the residual +0.25 of family-type extra words (rl_e_swap_min < 0.2) that test has more of.
Counts pairs crossing p = 0.5 / 0.01 in each direction, overall and for France."""
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
LR = ["rl_e_lrate_min", "rl_e_lrate_max", "rl_m_lrate_min"]
b = xgb.Booster(); b.load_model(os.path.join(R, "exp13", "stage1_fold0.json")); F = b.feature_names
J = [F.index(f) for f in LR]; JE = [F.index("rl_e_lrate_min"), F.index("rl_e_lrate_max")]; JS = F.index("rl_e_swap_min")
cte = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
parts = [sorted(glob.glob(os.path.join(d, "part*.parquet"))) for d in
         [os.path.join(C, "feat_v1", "test"), os.path.join(C, "feat_v7", "test"), os.path.join(C, "feat_v9", "test"), os.path.join(C, "feat_v13", "test")]]
rng = np.random.default_rng(0)
agg = {}
def add(key, a, b_, m):
    d = agg.setdefault(key, {"rows": 0, "up@0.5": 0, "down@0.5": 0, "up@0.01": 0, "down@0.01": 0, "|dp|>0.05": 0})
    d["rows"] += int(m.sum())
    for t in [0.5, 0.01]:
        d[f"up@{t}"] += int(((a <= t) & (b_ > t) & m).sum()); d[f"down@{t}"] += int(((a > t) & (b_ <= t) & m).sum())
    d["|dp|>0.05"] += int(((np.abs(b_ - a) > 0.05) & m).sum())
for i in range(len(parts[0])):
    n = pl.scan_parquet(parts[0][i]).select(pl.len()).collect().item()
    idx = np.sort(rng.choice(n, min(n, 1_000_000), replace=False))
    X = pl.concat([pl.read_parquet(f)[idx] for f in (p[i] for p in parts)], how="horizontal_extend" if hasattr(pl, "__version__") else "horizontal")
    ctry = cte[X["s1"].to_numpy()]
    M = X.select(F).cast(pl.Float32).to_numpy(); del X
    p0 = b.inplace_predict(M)
    for j in J:
        M[:, j] -= np.log10(1 / 0.8)
    pA = b.inplace_predict(M)
    fam = np.nan_to_num(M[:, JS], nan=9) < 0.2
    for j in JE:
        M[fam, j] -= 0.25
    pAF = b.inplace_predict(M); del M
    allm = np.ones(len(p0), bool); fr = ctry == "France"
    add("A   (fix)          all", p0, pA, allm); add("A   (fix)       France", p0, pA, fr)
    add("A+F (residual)  family rows", pA, pAF, fam); add("A+F (residual)  family France", pA, pAF, fam & fr)
    print(f"part {i} done", flush=True)
pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame([{"variant": k, **v} for k, v in agg.items()]))
