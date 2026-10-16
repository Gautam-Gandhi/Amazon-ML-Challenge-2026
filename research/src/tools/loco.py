"""Leave-one-country-out (LOCO) check: a proxy for France (a country never seen in training).

Trains a stage-1 XGB on one country's pairs and scores the other country; compares with the in-country
cross-fit score. Optional --drop removes feature groups to see which features generalize across countries.

Usage: python tools/loco.py --feat_dir data/cache/feat_v3/k80s0 --configs "base;mono;mono,d4;mono,drop=count+len"
  config tokens: mono (monotone constraints on similarity/rank features), dN (max_depth N), drop=g1+g2 (feature groups),
                 iw / iwN (covariate-shift importance weights: domain classifier source-vs-target on features only,
                 weight = odds(target|x), clipped to [1/N, N], default N=10),
                 rank (per-country rank normalization: each feature -> its percentile within the country's own
                 unlabeled candidate pairs, removing per-country scale shifts),
                 st / stN (self-training, N rounds: target pairs the model is confident about become pseudo-labels -
                 assigned & p > 0.98 -> 1, p < 0.02 -> 0 (sampled like the source negatives) - and the model is
                 retrained on source + pseudo-labeled target; transductive, as for France on the test set),
                 qmap (same trained model, extra evaluations where selected target-country features are quantile-
                 mapped onto the source country's distribution: variants in QMAPS; unlabeled target features only)
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
import exp03_dense as M3  # noqa: E402

GROUPS = {
    "cos": ["cos_n", "cos_a", "cos_j", "s1_gap_j", "s1_gap_a", "s1_gap_n", "r_gap_j", "r_margin2"],
    "rank": ["rk_name", "rk_addr", "rk_joint", "rk_rev", "s1_rank_j", "r_rank_j"],
    "count": ["a_s1cnt", "a_rcnt", "b_s1cnt", "b_rcnt", "r_ns1", "s1_ncand"],
    "tok": ["tk_a_min", "tk_a_nun", "tk_a_idfun", "tk_a_cov", "tk_b_min", "tk_b_nun", "tk_b_idfun", "tk_b_cov",
            "atk_a_min", "atk_a_nun", "atk_a_idfun", "atk_a_cov", "atk_b_min", "atk_b_nun", "atk_b_idfun", "atk_b_cov",
            "hn_logdiff", "hn_rel", "hn_prefix", "hn_edit", "hn_nmiss"],
    "len": ["a_nlen", "b_nlen", "a_ntok", "b_ntok", "a_natok", "b_natok", "a_nnum", "b_nnum", "a_ncomp", "b_ncomp"],
}
# +1: higher value -> more likely a match; -1: higher value -> less likely
MONO_POS = ["nc_ratio", "nc_tsort", "nc_tset", "nc_partial", "nc_jw", "ncl_ratio", "ncl_tsort", "ncl_tset",
            "ncl_partial", "ncl_jw", "nns_ratio", "nns_partial", "ad_ratio", "ad_tsort", "ad_tset", "ad_partial",
            "num_tset", "num1_ratio", "num_jacc", "ad_jacc", "nc_jacc", "nc_first_eq", "sfx_eq", "sfx_jacc",
            "cos_n", "cos_a", "cos_j", "s1_gap_j", "s1_gap_a", "s1_gap_n", "r_gap_j", "r_margin2"]
MONO_NEG = ["rk_name", "rk_addr", "rk_joint", "rk_rev", "s1_rank_j", "r_rank_j"]


def parse_config(cfg):
    out = {"mono": False, "depth": 8, "drop": set(), "iw": 0.0, "rank": False, "st": 0, "qmap": False}
    for tok in [t for t in cfg.split(",") if t and t != "base"]:
        if tok == "mono":
            out["mono"] = True
        elif tok.startswith("d") and tok[1:].isdigit():
            out["depth"] = int(tok[1:])
        elif tok.startswith("st"):
            out["st"] = int(tok[2:] or 1)
        elif tok == "qmap":
            out["qmap"] = True
        elif tok == "rank":
            out["rank"] = True
        elif tok.startswith("iw"):
            out["iw"] = float(tok[2:] or 10)
        elif tok.startswith("drop="):
            for g in tok[5:].split("+"):
                out["drop"] |= set(GROUPS[g])
    return out


QMAPS = {
    "margin": ["r_margin2"],
    "cos": ["cos_n", "cos_a", "cos_j", "s1_gap_j", "s1_gap_a", "s1_gap_n", "r_gap_j", "r_margin2"],
    "cos+idf": ["cos_n", "cos_a", "cos_j", "s1_gap_j", "s1_gap_a", "s1_gap_n", "r_gap_j", "r_margin2",
                "tk_a_idfun", "tk_b_idfun", "atk_a_idfun", "atk_b_idfun"],
}


def quantile_map(Xt, Xs, idx, q=1001, n=600_000, seed=0):
    """Map columns idx of target Xt onto the source distribution (per-column quantile matching)."""
    rng = np.random.default_rng(seed)
    St = Xt[rng.choice(len(Xt), min(n, len(Xt)), replace=False)][:, idx].astype(np.float64)
    Ss = Xs[rng.choice(len(Xs), min(n, len(Xs)), replace=False)][:, idx].astype(np.float64)
    qs = np.linspace(0, 1, q)
    out = Xt.copy()
    for j, c in enumerate(idx):
        gt, gs = np.nanquantile(St[:, j], qs), np.nanquantile(Ss[:, j], qs)
        col = out[:, c].astype(np.float64)
        ok = ~np.isnan(col)
        # midpoint of ties: many features have point masses (e.g. cos = 1.0)
        lo = np.searchsorted(gt, col[ok], side="left") / (q - 1)
        hi = np.searchsorted(gt, col[ok], side="right") / (q - 1)
        out[ok, c] = np.interp((lo + hi) / 2, qs, gs).astype(np.float32)
    return out


def rank_fit(X, n=400_000, q=1001, seed=0):
    """Per-feature quantile grids from a sample of a country's (unlabeled) candidate pairs."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(n, len(X)), replace=False)].astype(np.float32)
    return np.nanquantile(S, np.linspace(0, 1, q), axis=0).astype(np.float32)   # (q, F)


def rank_apply(X, grid, bs=2_000_000):
    out = np.empty(X.shape, np.float32)
    for i in range(0, len(X), bs):
        B = X[i:i + bs].astype(np.float32)
        for j in range(B.shape[1]):
            g = grid[:, j]
            col = B[:, j]
            r = np.searchsorted(g, col, side="right") / len(g)
            out[i:i + bs, j] = np.where(np.isnan(col), np.nan, r)
    return out


def importance_weights(Xs, Xt, clip, xgb, n=1_000_000, seed=0):
    """Covariate-shift weights for source rows: odds that a row comes from the target domain (unlabeled features)."""
    rng = np.random.default_rng(seed)
    a = Xs[rng.choice(len(Xs), min(n, len(Xs)), replace=False)]
    b = Xt[rng.choice(len(Xt), min(n, len(Xt)), replace=False)]
    X = np.vstack([a, b]).astype(np.float32)
    d = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    prm = {"objective": "binary:logistic", "device": "cuda", "tree_method": "hist", "max_depth": 6, "eta": 0.1,
           "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed}
    dom = xgb.train(prm, xgb.QuantileDMatrix(X, d), 200)
    p = np.clip(dom.inplace_predict(Xs.astype(np.float32)), 1e-4, 1 - 1e-4)
    w = (p / (1 - p)) * (len(a) / len(b))
    w = np.clip(w, 1.0 / clip, clip)
    print(f"  importance weights: mean {w.mean():.3f}, p10 {np.percentile(w, 10):.3f}, p90 {np.percentile(w, 90):.3f}", flush=True)
    return (w / w.mean()).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--keep_frac", type=float, default=0.8)
    ap.add_argument("--world_seed", type=int, default=0)
    ap.add_argument("--extra_dir", default="", help="comma-separated row-aligned extra feature layers (e.g. data/cache/feat_v5/train)")
    ap.add_argument("--configs", default="base")
    ap.add_argument("--neg_frac", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=900)
    ap.add_argument("--save_pred", default="", help="directory: save each config's target-country predictions")
    ap.add_argument("--tr_frac", type=float, default=0.6, help="share of source S1 used for training")
    ap.add_argument("--ev_frac", type=float, default=0.35, help="share of target S1 used for evaluation")
    args = ap.parse_args()
    import xgboost as xgb
    keep = M3.keep_mask(args)
    country = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
    parts = sorted(glob.glob(os.path.join(ROOT, args.feat_dir, "part*.parquet")))
    # memory-light: train on 60% of the source country's S1 (sampled negatives), evaluate on a fixed 35% of the
    # target country's S1 (all their candidate pairs). Same samples for every config -> fair comparison.
    h = (np.arange(len(country), dtype=np.uint64) * np.uint64(0x9E3779B1) % np.uint64(1000)).astype(np.int32)
    tr_s1 = h < int(1000 * args.tr_frac)
    ev_s1 = (h >= 600) & (h < 600 + int(1000 * args.ev_frac))
    rng = np.random.default_rng(0)
    Xtr = {c: [] for c in ["India", "US"]}; Ytr = {c: [] for c in Xtr}
    Xev = {c: [] for c in Xtr}; Kev = {c: [] for c in Xtr}
    allfeats = None
    extras = [sorted(glob.glob(os.path.join(ROOT, e, "part*.parquet"))) for e in args.extra_dir.split(",") if e]
    for e in extras:
        assert len(e) == len(parts)
    for i, p in enumerate(parts):
        d = pl.concat([pl.read_parquet(p)] + [pl.read_parquet(e[i]) for e in extras], how="horizontal")
        d = M1.add_labels(d)
        if allfeats is None:
            allfeats = [c for c in d.columns if c not in M1.NON_FEATS]
        s1 = d["s1"].to_numpy()
        c = country[s1]
        y = d["y"].to_numpy()
        u = rng.random(len(y))
        X = d.select(allfeats).to_numpy().astype(np.float32)
        for cc in Xtr:
            m = (c == cc) & tr_s1[s1] & ((y == 1) | (u < args.neg_frac))
            Xtr[cc].append(X[m]); Ytr[cc].append(y[m])
            m = (c == cc) & ev_s1[s1]
            Xev[cc].append(X[m]); Kev[cc].append(d.filter(pl.Series(m)).select("s1", "r", "y"))
        del X, d
    for cc in Xtr:
        Xtr[cc] = np.concatenate(Xtr[cc]); Ytr[cc] = np.concatenate(Ytr[cc])
        Xev[cc] = np.concatenate(Xev[cc]); Kev[cc] = pl.concat(Kev[cc])
        print(f"{cc}: train rows {len(Ytr[cc])}, eval rows {len(Xev[cc])}", flush=True)
    summary = []
    for cfg in args.configs.split(";"):
        pc = parse_config(cfg)
        feats = [f for f in allfeats if f not in pc["drop"]]
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "device": "cuda", "tree_method": "hist",
                  "max_depth": pc["depth"], "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5}
        if pc["mono"]:
            params["monotone_constraints"] = "(" + ",".join(
                "1" if f in MONO_POS else ("-1" if f in MONO_NEG else "0") for f in feats) + ")"
        row = [cfg]
        cols = [allfeats.index(f) for f in feats]
        for tr_c, te_c in [("US", "India"), ("India", "US")]:
            y = Ytr[tr_c]
            w = np.where(y == 1, 1, 1 / args.neg_frac).astype(np.float32)
            if pc["rank"]:
                Xs_ = rank_apply(Xtr[tr_c][:, cols], rank_fit(Xtr[tr_c][:, cols]))
                Xt_ = rank_apply(Xev[te_c][:, cols], rank_fit(Xev[te_c][:, cols]))
                dtr = xgb.QuantileDMatrix(Xs_, y, weight=w, feature_names=feats)
                bst = xgb.train(params, dtr, num_boost_round=args.rounds)
                p = bst.inplace_predict(Xt_)
                res = Kev[te_c].with_columns(pl.Series("p", p))
                m = M3.decode_eval(res, keep & (country == te_c) & ev_s1, print, f"[{cfg}] {tr_c}->{te_c}",
                                   thrs=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98))
                row += [m["curve"]["assign@0.7"], m["best_f05"], m["best_thr"]]
                del dtr, bst, Xs_, Xt_
                continue
            if pc["iw"] > 0:
                w = w * importance_weights(Xtr[tr_c][:, cols], Xev[te_c][:, cols], pc["iw"], xgb)
            dtr = xgb.QuantileDMatrix(Xtr[tr_c][:, cols], y, weight=w, feature_names=feats)
            bst = xgb.train(params, dtr, num_boost_round=args.rounds)  # fixed rounds: no peeking at the target
            p = np.concatenate([bst.inplace_predict(Xev[te_c][i:i + 1_000_000][:, cols])     # chunks: 4 GB GPU
                                for i in range(0, len(Xev[te_c]), 1_000_000)])
            for it in range(pc["st"]):
                r0 = Kev[te_c].select("s1", "r").with_columns(pl.Series("p", p)).with_row_index("i")
                pos = r0.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")                         .filter(pl.col("p") > 0.98)["i"].to_numpy()
                neg = r0.filter(pl.col("p") < 0.02)["i"].to_numpy()
                neg = neg[np.random.default_rng(it).random(len(neg)) < args.neg_frac]
                idx = np.sort(np.concatenate([pos, neg]))
                yp = np.isin(idx, pos).astype(np.float32)
                print(f"  self-training round {it + 1}: pseudo pos {len(pos)}, neg {len(neg)} "
                      f"(true rate of pseudo-pos {Kev[te_c]['y'].to_numpy()[pos].mean():.4f})", flush=True)
                del dtr
                dtr = xgb.QuantileDMatrix(np.concatenate([Xtr[tr_c][:, cols], Xev[te_c][idx][:, cols]]),
                                          np.concatenate([y, yp]),
                                          weight=np.concatenate([w, np.where(yp == 1, 1, 1 / args.neg_frac).astype(np.float32)]),
                                          feature_names=feats)
                bst = xgb.train(params, dtr, num_boost_round=args.rounds)
                p = np.concatenate([bst.inplace_predict(Xev[te_c][i:i + 1_000_000][:, cols])
                                    for i in range(0, len(Xev[te_c]), 1_000_000)])
            if pc["qmap"]:
                for qn, qf in QMAPS.items():
                    ids = [feats.index(f) for f in qf if f in feats]
                    Xq = quantile_map(Xev[te_c][:, cols], Xtr[tr_c][:, cols], ids)
                    pq = np.concatenate([bst.inplace_predict(Xq[i:i + 1_000_000]) for i in range(0, len(Xq), 1_000_000)])
                    del Xq
                    M3.decode_eval(Kev[te_c].with_columns(pl.Series("p", pq)), keep & (country == te_c) & ev_s1, print,
                                   f"[{cfg}:{qn}] {tr_c}->{te_c}", thrs=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98))
            res = Kev[te_c].with_columns(pl.Series("p", p))
            if args.save_pred:
                os.makedirs(args.save_pred, exist_ok=True)
                res.write_parquet(os.path.join(args.save_pred, f"{cfg}_{tr_c}_{te_c}.parquet"))
            m = M3.decode_eval(res, keep & (country == te_c) & ev_s1, print, f"[{cfg}] {tr_c}->{te_c}",
                               thrs=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98))
            row += [m["curve"]["assign@0.7"], m["best_f05"], m["best_thr"]]
            del dtr, bst
        summary.append(row)
        print(f"SUMMARY {cfg}: US->IN @0.7 {row[1]:.5f} best {row[2]:.5f}@{row[3]} | IN->US @0.7 {row[4]:.5f} "
              f"best {row[5]:.5f}@{row[6]}", flush=True)


if __name__ == "__main__":
    main()
