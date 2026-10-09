"""exp13 role feature 'lrate' = log10((n+1)/N1). On train, n comes from the dense world (80% of S1 kept) but N1 counts
ALL train S1 -> train lrate is ~log10(0.8) = -0.097 lower than the test convention. How much does that matter?
(1) gain share of the lrate features, (2) same-country (US, India) median lrate train vs test on the records' best
candidates, (3) stage-1 sensitivity: predict test rows with lrate shifted into the train convention, count changes."""
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
g = b.get_score(importance_type="total_gain"); tot = sum(g.values())
print("gain share:", {f: round(g.get(f, 0) / tot, 4) for f in LR})
ctr = pl.read_parquet(os.path.join(C, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
cte = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
LAY = {"train": [os.path.join(C, "feat_v3", "k80s0"), os.path.join(C, "feat_v13", "train")],
       "test": [os.path.join(C, "feat_v1", "test"), os.path.join(C, "feat_v13", "test")]}
for split, cmap in [("train", ctr), ("test", cte)]:
    base = sorted(glob.glob(os.path.join(LAY[split][0], "part*.parquet"))); rl = sorted(glob.glob(os.path.join(LAY[split][1], "part*.parquet")))
    acc = []
    for p, q in zip(base, rl):
        k = pl.read_parquet(p, columns=["s1", "r_rank_j"]); x = pl.read_parquet(q, columns=["rl_e_lrate_max"])
        acc.append(pl.DataFrame({"c": cmap[k["s1"].to_numpy()], "top": k["r_rank_j"].to_numpy() == 1, "v": x["rl_e_lrate_max"].to_numpy()})
                   .filter(pl.col("top") & pl.col("v").is_not_null() & pl.col("c").is_in(["US", "India"])))
    d = pl.concat(acc).group_by("c").agg(pl.col("v").median().alias("median"), pl.col("v").quantile(0.25).alias("q25"), pl.len().alias("n"))
    print(split, d.sort("c").rows())
# (3) sensitivity on a 1M-row sample of every test part
parts = [sorted(glob.glob(os.path.join(d, "part*.parquet"))) for d in
         [os.path.join(C, "feat_v1", "test"), os.path.join(C, "feat_v7", "test"), os.path.join(C, "feat_v9", "test"), os.path.join(C, "feat_v13", "test")]]
rng = np.random.default_rng(0)
tot_n = 0; stats = {"|dp|>0.01": 0, "|dp|>0.05": 0, "cross 0.003": 0, "cross 0.01": 0, "cross 0.5": 0, "cross 0.7": 0}
fr = {"n": 0, "cross 0.003": 0, "|dp|>0.05": 0}
for i in range(len(parts[0])):
    n = pl.read_parquet_metadata(parts[0][i]) if False else pl.scan_parquet(parts[0][i]).select(pl.len()).collect().item()
    idx = np.sort(rng.choice(n, min(n, 1_000_000), replace=False))
    X = pl.concat([pl.read_parquet(f)[idx] for f in (p[i] for p in parts)], how="horizontal")
    ctry = cte[X["s1"].to_numpy()]
    M = X.select(F).cast(pl.Float32).to_numpy()
    del X
    p0 = b.inplace_predict(M)
    for f in LR:
        j = F.index(f); M[:, j] = M[:, j] + np.log10(0.8)
    p1 = b.inplace_predict(M)
    del M
    d = np.abs(p1 - p0); tot_n += len(d)
    stats["|dp|>0.01"] += int((d > 0.01).sum()); stats["|dp|>0.05"] += int((d > 0.05).sum())
    for t in [0.003, 0.01, 0.5, 0.7]:
        stats[f"cross {t}"] += int(((p0 > t) != (p1 > t)).sum())
    m = ctry == "France"
    fr["n"] += int(m.sum()); fr["cross 0.003"] += int(((p0[m] > 0.003) != (p1[m] > 0.003)).sum()); fr["|dp|>0.05"] += int((d[m] > 0.05).sum())
    print(f"part {i}: rows {len(d)}  max|dp| {d.max():.4f}", flush=True)
print("sampled test rows", tot_n, stats)
print("France", fr)
