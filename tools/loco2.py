"""Two-stage leave-one-country-out: does STAGE 2 help or hurt on a country never seen in training (France proxy)?

For source country A -> target B (dense world k80s0, old or new world depending on ER_WORK_DIR / dirs):
  stage 1 : 2-fold cross-fit on A's S1 (sampled negatives) -> OOF p1 for all A rows; B p1 = mean of the 2 fold models
  stage 2 : exp02/exp05-style features from p1 (group + consistency), cross-fit on A -> B p2 = mean of fold models
Reports F0.5 on B (all kept B S1) for p1 and p2 at several thresholds. Same pipeline as production, only the
training country differs.
Usage: python tools/loco2.py --feat1 data/cache/feat_v3/k80s0 --feat2 dirA,dirB (comma-separated extra layers) [--rounds 600]
"""
import os
import sys
import glob
import argparse
import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from er_common import CACHE, s1_fold  # noqa: E402
import exp01_match as M1  # noqa: E402
import exp02_stack as M2  # noqa: E402
import exp03_dense as M3  # noqa: E402

THRS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97)


def xgb_fit(X, y, rounds, neg_frac):
    import xgboost as xgb
    w = np.where(y == 1, 1.0, 1.0 / neg_frac).astype(np.float32)
    prm = {"objective": "binary:logistic", "device": "cuda", "tree_method": "hist", "max_depth": 8, "eta": 0.1,
           "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "seed": 0}
    return xgb.train(prm, xgb.QuantileDMatrix(X, y, weight=w), rounds)


def predict(b, X, bs=2_000_000):
    return np.concatenate([b.inplace_predict(X[i:i + bs].astype(np.float32)) for i in range(0, len(X), bs)])


def stage2_feats(keys, p, r_tab, log):
    t = M2.group_feats(keys.with_columns(pl.Series("p", p)).with_row_index("row"))
    cons = M2.consistency_feats(t, r_tab, 3, log)
    f = pl.concat([t.drop("row"), cons.drop("row")], how="horizontal").rename({"p": "p1"}).drop("s1", "r")
    return f.to_numpy().astype(np.float16), f.columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat1", required=True)
    ap.add_argument("--feat2", default="")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--neg_frac", type=float, default=0.3)
    args = ap.parse_args()
    keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
    country = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
    fold = s1_fold(len(country))
    p1s = sorted(glob.glob(os.path.join(ROOT, args.feat1, "part*.parquet")))
    extra = [sorted(glob.glob(os.path.join(ROOT, e, "part*.parquet"))) for e in args.feat2.split(",") if e]
    for e in extra:
        assert len(e) == len(p1s)
    X, K, feats = {"US": [], "India": []}, {"US": [], "India": []}, None
    for i, a in enumerate(p1s):
        d = pl.concat([pl.read_parquet(a)] + [pl.read_parquet(e[i]) for e in extra], how="horizontal")
        d = M1.add_labels(d)
        feats = feats or [c for c in d.columns if c not in M1.NON_FEATS]
        c = country[d["s1"].to_numpy()]
        for cc in X:
            m = c == cc
            X[cc].append(d.filter(pl.Series(m)).select(feats).to_numpy().astype(np.float16))
            K[cc].append(d.filter(pl.Series(m)).select("s1", "r", "y"))
    for cc in X:
        X[cc] = np.concatenate(X[cc]); K[cc] = pl.concat(K[cc])
        print(f"{cc}: rows {len(X[cc])}, feats {len(feats)}", flush=True)
    r_tab = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["name_core", "addr_clean", "addr_nums", "src"])
    rng = np.random.default_rng(0)
    for A, B in [("US", "India"), ("India", "US")]:
        XA, KA, XB, KB = X[A], K[A], X[B], K[B]
        yA = KA["y"].to_numpy()
        fA = fold[KA["s1"].to_numpy()]
        samp = (yA == 1) | (rng.random(len(yA)) < args.neg_frac)
        # stage 1: cross-fit on A, fold-model average on B
        p1A = np.zeros(len(yA), np.float32)
        p1B = np.zeros(len(XB), np.float32)
        for k in (0, 1):
            m = samp & (fA == k)
            b = xgb_fit(XA[m].astype(np.float32), yA[m], args.rounds, args.neg_frac)
            p1A[fA != k] = predict(b, XA[fA != k])
            p1B += predict(b, XB) / 2
        print(f"[{A}->{B}] stage 1 done", flush=True)
        # stage 2 features (from p1) for A (OOF) and B
        S2A, s2cols = stage2_feats(KA.select("s1", "r"), p1A, r_tab, lambda *a: None)
        S2B, _ = stage2_feats(KB.select("s1", "r"), p1B, r_tab, lambda *a: None)
        p2B = np.zeros(len(XB), np.float32)
        for k in (0, 1):
            m = samp & (fA == k)
            b = xgb_fit(np.hstack([XA[m], S2A[m]]).astype(np.float32), yA[m], max(200, args.rounds // 3), args.neg_frac)
            p2B += predict(b, np.hstack([XB, S2B])) / 2
        kB = keep & (country == B)
        for nm, p in [("stage1", p1B), ("stage2", p2B)]:
            res = KB.with_columns(pl.Series("p", p))
            M3.decode_eval(res, kB, print, f"LOCO2 {A}->{B} {nm}", thrs=THRS)
        del S2A, S2B


if __name__ == "__main__":
    main()
