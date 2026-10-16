"""exp21_iw: covariate-shift importance weighting of the stage-2 matcher for UNSEEN countries (France).

Why: in LOCO, importance weighting (a domain classifier source vs target on unlabeled features; training rows
weighted by the odds of looking like the target, clipped) was the transfer technique with the largest average gain
(exp09-era features: US->India +0.006, India->US -0.0012; re-validated with the current feature layers in
logs/loco_iw13.log before this run is used).
Model: the exp15 stage-2 design (filtered candidates, same features/params), trained on one S1 fold of the labeled
train pairs (RAM) with weights w = clip(odds(unseen | x), 1/C, C), normalized; it re-scores every unseen-country
candidate pair. Seen countries keep --base (CE-stacked) predictions. The exp11 rule is re-applied afterwards.
Stage: fit_predict
"""
import os
import sys
import time
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import RUNS, Logger, s1_fold  # noqa: E402
import exp01_match as M1  # noqa: E402
import exp05_tokfeat as E5  # noqa: E402
import exp15_compact as E15  # noqa: E402

SEEN = {"US", "India"}


def f32(df, feats):
    return df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()


def stage_fit_predict(args, log):
    import xgboost as xgb
    rng = np.random.default_rng(0)
    country = pl.read_parquet(os.path.join(E5.PREP["test"], "test_s1.parquet"), columns=["country"])["country"].to_numpy()
    unseen = ~np.isin(country, list(SEEN))
    fold = s1_fold(pl.read_parquet(os.path.join(E5.PREP["train"], "train_s1.parquet"), columns=["country"]).height)
    X, Y, feats = [], [], None
    for part in E5.iter_s2("train")():
        part = M1.add_labels(part)
        if feats is None:
            feats = [c for c in part.columns if c not in M1.NON_FEATS]
        y = part["y"].to_numpy()
        m = (fold[part["s1"].to_numpy()] == 0) & ((y == 1) | (rng.random(len(y)) < args.neg_frac))
        X.append(f32(part.filter(pl.Series(m)), feats)); Y.append(y[m])
    Xtr, ytr = np.concatenate(X), np.concatenate(Y).astype(np.float32)
    del X, Y
    test_parts = []
    for part in E5.iter_s2("test")():
        u = part.filter(pl.Series(unseen[part["s1"].to_numpy()]))
        if u.height:
            test_parts.append(u)
    Xte = np.concatenate([f32(u, feats) for u in test_parts])
    log(f"train rows {len(ytr):,} (one S1 fold), unseen-country test rows {len(Xte):,}, features {len(feats)}")
    # domain classifier: source (train) vs target (unseen test), features only
    n = min(1_000_000, len(Xtr), len(Xte))
    a = Xtr[rng.choice(len(Xtr), n, replace=False)]
    b = Xte[rng.choice(len(Xte), n, replace=False)]
    prm = {"objective": "binary:logistic", "device": args.device, "tree_method": "hist", "max_depth": 6, "eta": 0.1,
           "subsample": 0.8, "colsample_bytree": 0.8, "seed": 0}
    dom = xgb.train(prm, xgb.QuantileDMatrix(np.vstack([a, b]), np.r_[np.zeros(n), np.ones(n)]), 200)
    del a, b
    pd = np.clip(dom.inplace_predict(Xtr), 1e-4, 1 - 1e-4)
    w_dom = np.clip(pd / (1 - pd), 1.0 / args.clip, args.clip)
    w_dom = (w_dom / w_dom.mean()).astype(np.float32)
    log(f"importance weights: p10 {np.percentile(w_dom, 10):.3f}, median {np.median(w_dom):.3f}, p90 "
        f"{np.percentile(w_dom, 90):.3f}, max {w_dom.max():.3f}")
    w = w_dom * np.where(ytr == 1, 1.0, 1.0 / args.neg_frac).astype(np.float32)
    dtr = xgb.QuantileDMatrix(Xtr, ytr, weight=w, feature_names=feats)
    del Xtr
    t0 = time.time()
    bst = xgb.train(M1.xgb_params(args), dtr, num_boost_round=args.rounds)
    bst.save_model(os.path.join(RUNS, args.run, "stage2_iw.json"))
    log(f"weighted stage 2: {args.rounds} rounds, {time.time() - t0:.0f}s")
    del dtr
    un = pl.concat([u.select("s1", "r") for u in test_parts]).with_columns(
        pl.Series("p", np.concatenate([bst.inplace_predict(Xte[i:i + 1_000_000]) for i in range(0, len(Xte), 1_000_000)]).astype(np.float32)))
    base = pl.read_parquet(os.path.join(RUNS, args.base, "test_pred.parquet")).select("s1", "r", "p")
    res = pl.concat([base.filter(~pl.Series(unseen).gather(base["s1"].to_numpy())),
                     un.with_columns(pl.col("p").cast(base["p"].dtype), pl.col("s1").cast(base["s1"].dtype), pl.col("r").cast(base["r"].dtype))])
    res.write_parquet(os.path.join(RUNS, args.run, "test_pred.parquet"))
    with open(os.path.join(RUNS, args.base, "metrics.json")) as fh:
        open(os.path.join(RUNS, args.run, "metrics.json"), "w").write(fh.read())
    au = un.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    for t in [0.9, 0.93, 0.95, 0.97]:
        log(f"unseen: matches per S1 @{t}: {au.filter(pl.col('p') > t).height / unseen.sum():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit_predict"])
    ap.add_argument("--run", default="exp21")
    ap.add_argument("--base", default="exp20c", help="CE-stacked predictions kept for training countries")
    ap.add_argument("--test_feat", default="feat_v1/test")
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--neg_frac", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--thr_seen", type=float, default=0.01)
    ap.add_argument("--thr_unseen", type=float, default=0.003)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    from er_common import set_determinism
    set_determinism(0)
    E15.use_layers(args)
    stage_fit_predict(args, log)


if __name__ == "__main__":
    main()
