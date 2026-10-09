"""exp28_gbstack: gradient-boosted stacker over [stage-2 p, all cross-encoder scores, structural features]
for training countries (replaces the logistic stacker where every CE score is present).

Why (09-27, dense validation, 3 CEs): logistic stack 0.99015 -> GBDT on [logit p, 3 CE] 0.99024 -> GBDT + 12 structural
features 0.99031. The stacker learns WHEN each cross-encoder is reliable (house numbers equal or not, address
missing, how common the name is, embedding cosines and margin), i.e. non-linear interplay a linear stack cannot use.
Pairs lacking some CE score keep the --fallback run's (logistic-stacked) probability; unseen countries keep it too.
Stages:
  eval    : 2-fold cross-fit (S1 folds) on the dense OOF band pairs -> F0.5, oof_combined.parquet, metrics.json
  predict : fit on all train band pairs -> score seen-country test pairs -> test_pred.parquet (+ outputs)
--comp r (09-27, exp29g): + rival features, i.e. for logit p and each CE score, the score minus the best score among the
record's other S1 candidates (a CE-level competition signal; dense 0.99032 -> 0.99050, the same gain with another seed).
S1-side rivals and the rival S1's structural features were tested and add nothing.
Downstream (unchanged): exp11_rolerule.py (unseen countries), exp19_align.py --types none (final decode).
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, save_json, s1_fold  # noqa: E402
import exp03_dense as M3  # noqa: E402
import exp05_tokfeat as E5  # noqa: E402

SEEN = {"US", "India"}
STRUCT = ["num1_eq", "cos_a", "cos_n", "ad_tset", "ncl_tset", "nc_tset", "r_margin2", "num_jacc", "sfx_eq",
          "b_addr_empty", "a_s1cnt", "b_s1cnt"]
PARAMS = {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 5, "eta": 0.05, "nthread": 6,
          "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "seed": 0}
ROUNDS = 500


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def rival(t, col, key):
    """col minus the best col among the other pairs of the same key (null when the key has no other scored pair)."""
    g = t.filter(pl.col(col).is_not_null()).group_by(key).agg(
        pl.col(col).max().alias("_m1"), pl.col(col).top_k(2).min().alias("_m2"), pl.len().alias("_n"))
    t = t.join(g, on=key, how="left")
    other = pl.when(pl.col("_n") < 2).then(None).when(pl.col(col) == pl.col("_m1")).then(pl.col("_m2")).otherwise(pl.col("_m1"))
    return t.with_columns((pl.col(col) - other).alias(f"{col}_rv{key}")).drop("_m1", "_m2", "_n")


def build(split, base, ce_runs, feat_dir, comp=False):
    t = pl.read_parquet(os.path.join(RUNS, base, "oof.parquet" if split == "train" else "test_pred.parquet"))
    t = t.select(["s1", "r", "p"] + (["y"] if "y" in t.columns else []))
    for k, run in enumerate(ce_runs):
        c = pl.read_parquet(os.path.join(RUNS, run, f"{split}_ce.parquet")).select("s1", "r", pl.col("ce").alias(f"c{k}"))
        t = t.join(c.with_columns(pl.col("s1").cast(t["s1"].dtype), pl.col("r").cast(t["r"].dtype)), on=["s1", "r"], how="left")
    comp_cols = []
    if comp:
        # competition of the CE scores (and of the stage-2 p) across the other S1 candidates of the same record
        t = t.with_columns(pl.col("p").clip(1e-6, 1 - 1e-6).pipe(lambda e: (e / (1 - e)).log()).alias("lp"))
        for col in ["lp"] + [f"c{k}" for k in range(len(ce_runs))]:
            t = rival(t, col, "r")
            comp_cols.append(f"{col}_rvr")
    has = t.filter(pl.all_horizontal([pl.col(f"c{k}").is_not_null() for k in range(len(ce_runs))]))
    keys = has.select("s1", "r")
    st = pl.concat([pl.read_parquet(f, columns=["s1", "r"] + STRUCT).join(keys, on=["s1", "r"], how="semi")
                    for f in sorted(glob.glob(os.path.join(feat_dir, "part*.parquet")))])
    has = has.join(st, on=["s1", "r"], how="left")
    X = np.stack([logit(has["p"].to_numpy())] + [has[f"c{k}"].to_numpy() for k in range(len(ce_runs))]
                 + [has[c].cast(pl.Float32).to_numpy() for c in STRUCT]
                 + [has[c].cast(pl.Float32).fill_null(np.nan).to_numpy() for c in comp_cols], 1).astype(np.float32)
    return t, has, X


def stage_eval(args, log):
    import xgboost as xgb
    ces = args.ce_runs.split(",")
    keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
    t, has, X = build("train", args.base, ces, E5.FEAT3, args.comp)
    y = has["y"].to_numpy()
    fold = s1_fold(len(keep))[has["s1"].to_numpy()]
    newp = np.zeros(len(y), np.float32)
    for k in (0, 1):
        tr, te = fold == k, fold != k
        newp[te] = xgb.train(PARAMS, xgb.DMatrix(X[tr], y[tr]), ROUNDS).predict(xgb.DMatrix(X[te]))
    log(f"band pairs with all {len(ces)} CE scores: {has.height:,}")
    ref = pl.read_parquet(os.path.join(RUNS, args.fallback, "oof_combined.parquet")).select("s1", "r", pl.col("p").alias("pref"))
    new = t.join(has.select("s1", "r").with_columns(pl.Series("pn", newp)), on=["s1", "r"], how="left") \
           .join(ref, on=["s1", "r"], how="left").with_columns(pl.coalesce(["pn", "pref", "p"]).alias("p")).select("s1", "r", "p")
    m_ref = M3.decode_eval(pl.read_parquet(os.path.join(RUNS, args.fallback, "oof_combined.parquet")), keep, log, f"{args.fallback} (reference)")
    m = M3.decode_eval(new, keep, log, f"GBDT stack ({','.join(ces)})")
    new.write_parquet(os.path.join(RUNS, args.run, "oof_combined.parquet"))
    save_json({"reference": m_ref, "combined": m}, os.path.join(RUNS, args.run, "metrics.json"))


def stage_predict(args, log):
    import xgboost as xgb
    ces = args.ce_runs.split(",")
    _, has_tr, Xtr = build("train", args.base, ces, E5.FEAT3, args.comp)
    bst = xgb.train(PARAMS, xgb.DMatrix(Xtr, has_tr["y"].to_numpy()), ROUNDS)
    bst.save_model(os.path.join(RUNS, args.run, "gbstack.json"))
    del Xtr
    t, has, X = build("test", args.base, ces, os.path.join(CACHE, args.test_feat), args.comp)
    country = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
    seen = np.isin(country[has["s1"].to_numpy()], list(SEEN))
    pn = bst.predict(xgb.DMatrix(X)).astype(np.float32)
    upd = has.select("s1", "r").with_columns(pl.Series("pn", pn)).filter(pl.Series(seen))
    log(f"test: GBDT-stacked seen-country pairs {upd.height:,}")
    fb = pl.read_parquet(os.path.join(RUNS, args.fallback, "test_pred.parquet")).select("s1", "r", pl.col("p").alias("pref"))
    res = t.select("s1", "r", "p").join(upd, on=["s1", "r"], how="left").join(fb, on=["s1", "r"], how="left") \
           .with_columns(pl.coalesce(["pn", "pref", "p"]).cast(pl.Float32).alias("p")).select("s1", "r", "p")
    res.write_parquet(os.path.join(RUNS, args.run, "test_pred.parquet"))
    log(f"wrote test_pred ({res.height:,} pairs); decode with exp11_rolerule + exp19_align --types none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eval", "predict"])
    ap.add_argument("--run", default="exp28")
    ap.add_argument("--base", default="exp15")
    ap.add_argument("--ce_runs", default="exp17,exp18,exp23")
    ap.add_argument("--fallback", default="exp24c", help="logistic-stacked run used where not every CE score exists")
    ap.add_argument("--test_feat", default="feat_v1/test")
    ap.add_argument("--comp", default="", choices=["", "r"],
                    help="r: rival features = score minus the best score among the record's other S1 candidates")
    ap.add_argument("--nthread", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    PARAMS["nthread"] = args.nthread
    PARAMS["seed"] = args.seed
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    {"eval": stage_eval, "predict": stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
