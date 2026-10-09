"""exp16_selftrain: self-training of the stage-2 matcher for UNSEEN countries (France), on the exp15 compact candidates.

Why: LOCO (tools/loco.py config st, EXPERIMENTS.md "Self-training"): one round of pseudo-labels on the target country
+0.0008 (US->India) / +0.0025 (India->US) at threshold 0.98, where both directions peak; at 0.9 it is mixed
(-0.0012 / +0.0008) because the self-trained model becomes over-confident -> the unseen threshold is fixed a priori to
0.98 (not tuned on France). Transductive use of the unlabeled test records is allowed by the competition rules.
Pseudo-labels on the unseen-country test pairs of --base (exp15cr: exp15 stage 2 + exp11 rule):
  positive = assigned (best S1 of its R) and (p > 0.98 or flipped by the exp11 rule; the rule is LB-validated)
  negative = p < 0.02 (sampled with --neg_frac like the labeled negatives)
A stage-2 model is trained on ALL labeled train pairs (dense world, filtered set) + these pseudo-labeled pairs, and
re-scores every unseen-country candidate pair. Seen countries keep --base predictions. The exp11 rule is re-applied
afterwards (exp11_rolerule --base exp16 --p_set 0.99 --unseen_thr 0.98).
Stage: fit_predict
"""
import os
import sys
import time
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, save_json, s1_fold  # noqa: E402
import exp01_match as M1  # noqa: E402
import exp05_tokfeat as E5  # noqa: E402
import exp11_rolerule as R11  # noqa: E402
import exp15_compact as E15  # noqa: E402

SEEN = {"US", "India"}


def f32(df, feats):
    return df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()


def stage_fit_predict(args, log):
    import xgboost as xgb
    rng = np.random.default_rng(0)
    country = pl.read_parquet(os.path.join(E5.PREP["test"], "test_s1.parquet"), columns=["country"])["country"].to_numpy()
    unseen = ~np.isin(country, list(SEEN))
    # pseudo-labels from the base run (rule flips = pairs raised to exactly the rule's p_set 0.95 from below)
    base = pl.read_parquet(os.path.join(RUNS, args.base, "test_pred.parquet")).select("s1", "r", "p")
    pre = pl.read_parquet(os.path.join(RUNS, args.base_pre_rule, "test_pred.parquet")).select("s1", "r", pl.col("p").alias("p0"))
    b = base.join(pre, on=["s1", "r"]).filter(pl.Series(unseen).gather(base.join(pre, on=["s1", "r"])["s1"].to_numpy()))
    a = R11.assigned(b.select("s1", "r", "p"))
    flipped = b.filter((pl.col("p") > pl.col("p0") + 1e-6))
    pos = a.filter(pl.col("p") > 0.98).select("s1", "r").vstack(flipped.select("s1", "r")).unique()
    neg = b.filter(pl.col("p") < 0.02).select("s1", "r")
    neg = neg.filter(pl.Series(rng.random(neg.height) < args.neg_frac))
    lab = pl.concat([pos.with_columns(pl.lit(1, pl.Int8).alias("yp")), neg.with_columns(pl.lit(0, pl.Int8).alias("yp"))])
    log(f"pseudo-labels (unseen countries): positive {pos.height:,} (rule flips {flipped.height:,}), negative {neg.height:,}")
    # labeled train pairs (filtered set), negatives sampled like cross_fit
    # one S1 fold of the labeled train pairs (like one cross-fit fold; the full set does not fit 16 GB with 142 features)
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
    Xp, yp, test_parts = [], [], []
    for part in E5.iter_s2("test")():
        u = part.filter(pl.Series(unseen[part["s1"].to_numpy()]))
        if u.height == 0:
            continue
        test_parts.append(u)
        j = u.select("s1", "r").with_row_index("i").join(lab, on=["s1", "r"])
        Xp.append(f32(u[j["i"].to_numpy()], feats)); yp.append(j["yp"].to_numpy())
    Xp, yp = np.concatenate(Xp), np.concatenate(yp).astype(np.float32)
    log(f"train rows: labeled {len(ytr):,}, pseudo {len(yp):,}; features {len(feats)}")
    w = lambda y: np.where(y == 1, 1.0, 1.0 / args.neg_frac).astype(np.float32)
    Xall = np.concatenate([Xtr, Xp]); del Xtr, Xp
    dtr = xgb.QuantileDMatrix(Xall, np.concatenate([ytr, yp]), weight=np.concatenate([w(ytr), w(yp)]), feature_names=feats)
    del Xall
    t0 = time.time()
    bst = xgb.train(M1.xgb_params(args), dtr, num_boost_round=args.rounds)
    bst.save_model(os.path.join(RUNS, args.run, "stage2_selftrain.json"))
    log(f"self-trained stage 2: {args.rounds} rounds, {time.time() - t0:.0f}s")
    del dtr
    outs = []
    for u in test_parts:
        outs.append(u.select("s1", "r").with_columns(pl.Series("p", bst.inplace_predict(f32(u, feats)).astype(np.float32))))
    un = pl.concat(outs)
    res = pl.concat([base.filter(~pl.Series(unseen).gather(base["s1"].to_numpy())), un.with_columns(pl.col("p").cast(base["p"].dtype))])
    res.write_parquet(os.path.join(RUNS, args.run, "test_pred.parquet"))
    for src in [args.base_pre_rule]:
        with open(os.path.join(RUNS, src, "metrics.json")) as fh:
            open(os.path.join(RUNS, args.run, "metrics.json"), "w").write(fh.read())
    au = R11.assigned(un)
    log(f"unseen: matches per S1 @0.9 {au.filter(pl.col('p') > 0.9).height / unseen.sum():.3f}, "
        f"@0.98 {au.filter(pl.col('p') > 0.98).height / unseen.sum():.3f}; share of p>0.3 in (0.3,0.98]: "
        f"{((au['p'] > 0.3) & (au['p'] <= 0.98)).sum() / max((au['p'] > 0.3).sum(), 1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit_predict"])
    ap.add_argument("--run", default="exp16")
    ap.add_argument("--base", default="exp15cr", help="final predictions used for pseudo-labels (incl. rule flips)")
    ap.add_argument("--base_pre_rule", default="exp15c", help="same predictions before the rule (to detect flips)")
    ap.add_argument("--test_feat", default="feat_v1/test")
    ap.add_argument("--neg_frac", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=150, help="~ exp15 stage-2 best iteration (131)")
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
