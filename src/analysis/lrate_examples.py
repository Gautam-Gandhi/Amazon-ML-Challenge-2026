"""Which pairs does the family-lrate inflation push over p=0.5 on test? Same sample/seed as lrate_family.py.
Prints examples and whether each pair is accepted in the final exp35 output."""
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
SP = os.path.dirname(os.path.abspath(__file__))
LR = ["rl_e_lrate_min", "rl_e_lrate_max", "rl_m_lrate_min"]
b = xgb.Booster(); b.load_model(os.path.join(R, "exp13", "stage1_fold0.json")); F = b.feature_names
J = [F.index(f) for f in LR]; JE = [F.index("rl_e_lrate_min"), F.index("rl_e_lrate_max")]; JS = F.index("rl_e_swap_min")
s1t = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["eid", "name_raw", "addr_raw", "country"])
rt = pl.read_parquet(os.path.join(C, "prep_v2", "test_r.parquet"), columns=["eid", "name_raw", "addr_raw"])
cte = s1t["country"].to_numpy()
parts = [sorted(glob.glob(os.path.join(d, "part*.parquet"))) for d in
         [os.path.join(C, "feat_v1", "test"), os.path.join(C, "feat_v7", "test"), os.path.join(C, "feat_v9", "test"), os.path.join(C, "feat_v13", "test")]]
rng = np.random.default_rng(0)
hits = []
for i in range(len(parts[0])):
    n = pl.scan_parquet(parts[0][i]).select(pl.len()).collect().item()
    idx = np.sort(rng.choice(n, min(n, 1_000_000), replace=False))
    X = pl.concat([pl.read_parquet(f)[idx] for f in (p[i] for p in parts)], how="horizontal")
    keys = X.select("s1", "r", "rl_e_swap_min", "rl_e_hneq_min", "rl_e_lrate_max", "ad_tset", "num1_eq", "r_rank_j")
    M = X.select(F).cast(pl.Float32).to_numpy(); del X
    for j in J:
        M[:, j] -= np.log10(1 / 0.8)
    pA = b.inplace_predict(M)
    fam = np.nan_to_num(M[:, JS], nan=9) < 0.2
    for j in JE:
        M[fam, j] -= 0.25
    pAF = b.inplace_predict(M); del M
    m = fam & (pA > 0.5) & (pAF <= 0.5)
    hits.append(keys.filter(pl.Series(m)).with_columns(pl.Series("pA", pA[m]), pl.Series("pAF", pAF[m])))
h = pl.concat(hits)
h = h.with_columns(pl.Series("country", cte[h["s1"].to_numpy()]))
# final exp35 decisions
out = pl.read_csv(os.path.join(R, "exp35", "output", "matching_results.tsv"), separator="\t")
acc = out.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids") \
         .rename({"source1_entity_id": "s1_eid", "matched_entity_ids": "r_eid"})
h = h.with_columns(pl.Series("s1_eid", s1t["eid"].to_numpy()[h["s1"].to_numpy()]), pl.Series("r_eid", rt["eid"].to_numpy()[h["r"].to_numpy()]))
h = h.join(acc.with_columns(pl.lit(1).alias("final")), on=["s1_eid", "r_eid"], how="left").with_columns(pl.col("final").fill_null(0))
print("pairs pushed over 0.5 by the family-lrate inflation (sample):", h.height)
print(h.group_by("country").agg(pl.len().alias("n"), pl.col("final").mean().alias("accepted_in_exp35"),
                                pl.col("num1_eq").eq(1).mean().alias("same_number"), pl.col("ad_tset").mean().alias("ad_tset_mean")).sort("country"))
h.write_parquet(os.path.join(SP, "lrate_examples.parquet"))
ex = h.sample(n=min(24, h.height), seed=1).with_columns(
    pl.Series("S1", [f"{a} | {b}" for a, b in s1t.select("name_raw", "addr_raw")[h.sample(n=min(24, h.height), seed=1)["s1"].to_numpy()].rows()]),
    pl.Series("record", [f"{a} | {b}" for a, b in rt.select("name_raw", "addr_raw")[h.sample(n=min(24, h.height), seed=1)["r"].to_numpy()].rows()]))
pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_rows(30)
print(ex.select("country", "S1", "record", "pA", "pAF", "final"))
