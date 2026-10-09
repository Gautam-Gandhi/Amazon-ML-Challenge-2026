"""Step 3 audit: train (dense world) vs test distribution of every stage-1 feature on a population that means the same
thing in both worlds: exact copies (nc_ratio == 100 and ad_ratio == 100: identical name core and address), and the
record's best candidate (r_rank_j == 1). A feature that shifts there is computed inconsistently or depends on world size."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, glob, sys
import numpy as np, polars as pl
import xgboost as xgb
C = _os.path.join(_WORK, 'data', 'cache')
R = _os.path.join(_WORK, 'runs')
SP = os.path.dirname(os.path.abspath(__file__))
b = xgb.Booster(); b.load_model(os.path.join(R, "exp13", "stage1_fold0.json")); FEATS = b.feature_names
g = b.get_score(importance_type="total_gain"); tot = sum(g.values()); GAIN = {k: v / tot for k, v in g.items()}
ctr = pl.read_parquet(os.path.join(C, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
cte = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
LAY = {"train": [os.path.join(C, "feat_v3", "k80s0"), os.path.join(C, "feat_v7", "train"), os.path.join(C, "feat_v9", "train"), os.path.join(C, "feat_v13", "train")],
       "test": [os.path.join(C, "feat_v1", "test"), os.path.join(C, "feat_v7", "test"), os.path.join(C, "feat_v9", "test"), os.path.join(C, "feat_v13", "test")]}
rng = np.random.default_rng(0)
def collect(split, ctry, per_part=40000):
    out = []
    parts = [sorted(glob.glob(os.path.join(d, "part*.parquet"))) for d in LAY[split]]
    cmap = ctr if split == "train" else cte
    for i in range(len(parts[0])):
        base = pl.read_parquet(parts[0][i], columns=["s1", "nc_ratio", "ad_ratio", "r_rank_j"])
        m = (cmap[base["s1"].to_numpy()] == ctry) & (base["nc_ratio"].to_numpy() == 100) & (base["ad_ratio"].to_numpy() == 100) & (base["r_rank_j"].to_numpy() == 1)
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        idx = np.sort(rng.choice(idx, min(per_part, len(idx)), replace=False))
        cols = []
        for L in range(4):
            have = [c for c in pl.read_parquet_schema(parts[L][i]) if c in FEATS]
            cols.append(pl.read_parquet(parts[L][i], columns=have)[idx])
        out.append(pl.concat(cols, how="horizontal"))
    return pl.concat(out).select(FEATS)
rows = []
for ctry in ["US", "India"]:
    tr = collect("train", ctry); te = collect("test", ctry)
    print(ctry, "exact-copy best-candidate pairs sampled: train", tr.height, "test", te.height, flush=True)
    for f in FEATS:
        a = tr[f].cast(pl.Float64).to_numpy(); c = te[f].cast(pl.Float64).to_numpy()
        na, nc = np.isnan(a).mean(), np.isnan(c).mean()
        a2, c2 = a[~np.isnan(a)], c[~np.isnan(c)]
        if len(a2) < 100 or len(c2) < 100:
            continue
        sd = np.sqrt((a2.var() + c2.var()) / 2) + 1e-9
        smd = (c2.mean() - a2.mean()) / sd
        edges = np.unique(np.quantile(a2, np.linspace(0, 1, 11)))
        if len(edges) > 2:
            pa = np.histogram(a2, bins=edges)[0] / len(a2) + 1e-4; pc = np.histogram(np.clip(c2, edges[0], edges[-1]), bins=edges)[0] / len(c2) + 1e-4
            psi = float(((pc - pa) * np.log(pc / pa)).sum())
        else:
            psi = float("nan")
        rows.append({"country": ctry, "feature": f, "gain": GAIN.get(f, 0.0), "smd": smd, "psi": psi, "null_tr": na, "null_te": nc,
                     "mean_tr": a2.mean(), "mean_te": c2.mean(), "med_tr": np.median(a2), "med_te": np.median(c2)})
d = pl.DataFrame(rows)
d.write_parquet(os.path.join(SP, "feat_drift.parquet"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_float_precision(4)
flag = d.filter((pl.col("smd").abs() > 0.1) | (pl.col("psi") > 0.1) | ((pl.col("null_tr") - pl.col("null_te")).abs() > 0.02))
print("FLAGGED features (|mean shift| > 0.1 sd, or PSI > 0.1, or missing-rate gap > 2pp), sorted by model gain:")
print(flag.sort("gain", descending=True))
