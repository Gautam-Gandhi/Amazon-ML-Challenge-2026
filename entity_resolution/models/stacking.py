"""Stacking of the stage-2 probability with the cross-encoder scores (training countries only).

Logistic stack : cross-fitted logistic regression on [logit p, ce, logit p * ce] (+ per extra cross-encoder k:
                 [ce_k, logit p * ce_k, ce * ce_k] for pairs that have every score). It is the fallback of the GBDT stack.
GBDT stack     : XGBoost over [logit p, the three CE scores, 12 structural features, rival features], where a rival
                 feature is a score minus the best score among the record's other candidate S1 entities. Its dense
                 out-of-fold predictions choose the final decoding threshold.
"""
import os
import glob

import numpy as np
import polars as pl

from .. import config as C
from ..utils import LOG, s1_fold
from ..features.pair_features import layer_dir
from .cross_encoder import band_dir
from .gradient_boosting import model_dir, xgb_progress
from .validation import best_threshold, train_keep_mask

CE_RUNS = ["ce1_filled", "ce2", "ce3"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------------------------- logistic stack
def stack_features(df):
    lp = logit(df["p"].to_numpy())
    ce = df["ce"].to_numpy()
    return np.stack([lp, ce, lp * ce], 1)


def fit_stacker(df):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=1000).fit(stack_features(df), df["y"].to_numpy())


def _extra_cols(df):
    return sorted([c for c in df.columns if c.startswith("ce") and c[2:].isdigit()], key=lambda c: int(c[2:]))


def stack2_features(df):
    lp = logit(df["p"].to_numpy())
    c1 = df["ce"].to_numpy()
    cols = [lp, c1, lp * c1]
    for c in _extra_cols(df):
        ck = df[c].to_numpy()
        cols += [ck, lp * ck, c1 * ck]
    return np.stack(cols, 1)


def fit_stack2(df):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=3000).fit(stack2_features(df), df["y"].to_numpy())


def _with_ce(ws, res, split):
    d = res.join(pl.read_parquet(os.path.join(band_dir(ws, CE_RUNS[0]), f"{split}_ce.parquet")), on=["s1", "r"], how="left")
    for k, run in enumerate(CE_RUNS[1:]):
        e = pl.read_parquet(os.path.join(band_dir(ws, run), f"{split}_ce.parquet")).select("s1", "r", "ce")
        d = d.join(e.rename({"ce": f"ce{k + 2}"}), on=["s1", "r"], how="left")
    return d


def _all_extra():
    return pl.all_horizontal([pl.col(f"ce{k + 2}").is_not_null() for k in range(len(CE_RUNS) - 1)])


def logistic_stack(ws):
    """Stage: cross-fitted logistic stack (out-of-fold on the dense world) and its test predictions."""
    out = model_dir(ws, "logistic_stack")
    keep = train_keep_mask(ws)
    oof = pl.read_parquet(os.path.join(model_dir(ws, "stage2"), "oof.parquet"))
    d = _with_ce(ws, oof, "train")
    has = d.filter(pl.col("ce").is_not_null())
    both = _all_extra()
    parts = []
    for sub, fit, feat in [(has.filter(~both), fit_stacker, stack_features), (has.filter(both), fit_stack2, stack2_features)]:
        if sub.height == 0:
            continue
        fold = s1_fold(len(keep))[sub["s1"].to_numpy()]
        newp = np.zeros(sub.height, np.float32)
        for k in (0, 1):
            lr = fit(sub.filter(pl.Series(fold == k)))
            newp[fold != k] = lr.predict_proba(feat(sub.filter(pl.Series(fold != k))))[:, 1]
        parts.append(sub.select("s1", "r").with_columns(pl.Series("pn", newp)))
    new = oof.join(pl.concat(parts), on=["s1", "r"], how="left") \
             .with_columns(pl.coalesce(["pn", "p"]).alias("p")).drop("pn")
    new.select("s1", "r", "p").write_parquet(os.path.join(out, "oof_combined.parquet"))
    lr1 = fit_stacker(has.filter(~both))
    lr2 = fit_stack2(has.filter(both))
    # test: training-country pairs with a CE score are re-scored, everything else keeps the stage-2 probability
    tp = pl.read_parquet(os.path.join(model_dir(ws, "stage2"), "test_pred.parquet"))
    country = pl.read_parquet(ws.table("test", "s1"), columns=["country"])["country"].to_numpy()
    d = _with_ce(ws, tp, "test")
    has = d.filter(pl.col("ce").is_not_null() & pl.Series(np.isin(country[d["s1"].to_numpy()], list(C.TRAINING_COUNTRIES))))
    one = has.filter(~both)
    two = has.filter(both)
    parts = [one.select("s1", "r").with_columns(pl.Series("pn", lr1.predict_proba(stack_features(one))[:, 1].astype(np.float32))),
             two.select("s1", "r").with_columns(pl.Series("pn", lr2.predict_proba(stack2_features(two))[:, 1].astype(np.float32)))]
    res = tp.join(pl.concat(parts), on=["s1", "r"], how="left") \
            .with_columns(pl.coalesce(["pn", "p"]).alias("p")).drop("pn")
    res.write_parquet(os.path.join(out, "test_pred.parquet"))
    LOG.info("logistic stack: %s single-CE and %s multi-CE test pairs re-scored", f"{one.height:,}", f"{two.height:,}")


# ---------------------------------------------------------------------------------------------- GBDT stack
def _rival(t, col, key):
    """col minus the best col among the other pairs of the same key (null when the key has no other scored pair)."""
    g = t.filter(pl.col(col).is_not_null()).group_by(key).agg(
        pl.col(col).max().alias("_m1"), pl.col(col).top_k(2).min().alias("_m2"), pl.len().alias("_n"))
    t = t.join(g, on=key, how="left")
    other = pl.when(pl.col("_n") < 2).then(None).when(pl.col(col) == pl.col("_m1")).then(pl.col("_m2")).otherwise(pl.col("_m1"))
    return t.with_columns((pl.col(col) - other).alias(f"{col}_rv{key}")).drop("_m1", "_m2", "_n")


def _gbdt_inputs(ws, split):
    t = pl.read_parquet(os.path.join(model_dir(ws, "stage2"), "oof.parquet" if split == "train" else "test_pred.parquet"))
    t = t.select(["s1", "r", "p"] + (["y"] if "y" in t.columns else []))
    for k, run in enumerate(CE_RUNS):
        c = pl.read_parquet(os.path.join(band_dir(ws, run), f"{split}_ce.parquet")).select("s1", "r", pl.col("ce").alias(f"c{k}"))
        t = t.join(c.with_columns(pl.col("s1").cast(t["s1"].dtype), pl.col("r").cast(t["r"].dtype)), on=["s1", "r"], how="left")
    comp_cols = []
    t = t.with_columns(pl.col("p").clip(1e-6, 1 - 1e-6).pipe(lambda e: (e / (1 - e)).log()).alias("lp"))
    for col in ["lp"] + [f"c{k}" for k in range(len(CE_RUNS))]:
        t = _rival(t, col, "r")
        comp_cols.append(f"{col}_rvr")
    has = t.filter(pl.all_horizontal([pl.col(f"c{k}").is_not_null() for k in range(len(CE_RUNS))]))
    keys = has.select("s1", "r")
    st = pl.concat([pl.read_parquet(f, columns=["s1", "r"] + C.STACK_STRUCT_FEATURES).join(keys, on=["s1", "r"], how="semi")
                    for f in sorted(glob.glob(os.path.join(layer_dir(ws, "base", split), "part*.parquet")))])
    has = has.join(st, on=["s1", "r"], how="left")
    X = np.stack([logit(has["p"].to_numpy())] + [has[f"c{k}"].to_numpy() for k in range(len(CE_RUNS))]
                 + [has[c].cast(pl.Float32).to_numpy() for c in C.STACK_STRUCT_FEATURES]
                 + [has[c].cast(pl.Float32).fill_null(np.nan).to_numpy() for c in comp_cols], 1).astype(np.float32)
    return t, has, X


def gbdt_stack(ws):
    """Stage: GBDT stack. Cross-fitted on the dense world to choose the decoding threshold, then fitted on all
    training band pairs to score the test pairs of the training countries."""
    import xgboost as xgb
    out = model_dir(ws, "gbdt_stack")
    keep = train_keep_mask(ws)
    t, has, X = _gbdt_inputs(ws, "train")
    y = has["y"].to_numpy()
    fold = s1_fold(len(keep))[has["s1"].to_numpy()]
    newp = np.zeros(len(y), np.float32)
    for k in (0, 1):
        tr, te = fold == k, fold != k
        newp[te] = xgb.train(C.GBDT_STACK_PARAMS, xgb.DMatrix(X[tr], y[tr]), C.GBDT_STACK_ROUNDS,
                             callbacks=[xgb_progress(C.GBDT_STACK_ROUNDS, f"GBDT stack fold {k}")]).predict(xgb.DMatrix(X[te]))
        LOG.info("GBDT stack fold %d: trained on %s pairs, scored %s out-of-fold pairs", k, f"{int(tr.sum()):,}",
                 f"{int(te.sum()):,}")
    ref = pl.read_parquet(os.path.join(model_dir(ws, "logistic_stack"), "oof_combined.parquet")).select("s1", "r", pl.col("p").alias("pref"))
    new = t.join(has.select("s1", "r").with_columns(pl.Series("pn", newp)), on=["s1", "r"], how="left") \
           .join(ref, on=["s1", "r"], how="left").with_columns(pl.coalesce(["pn", "pref", "p"]).alias("p")).select("s1", "r", "p")
    thr, f05 = best_threshold(ws, new, keep, "GBDT stack")
    ws.save_json({"threshold": thr, "validation_f05": f05}, "models", "gbdt_stack", "threshold.json")
    del X
    _, has_tr, Xtr = _gbdt_inputs(ws, "train")
    bst = xgb.train(C.GBDT_STACK_PARAMS, xgb.DMatrix(Xtr, has_tr["y"].to_numpy()), C.GBDT_STACK_ROUNDS,
                    callbacks=[xgb_progress(C.GBDT_STACK_ROUNDS, "GBDT stack (all training pairs)")])
    del Xtr
    t, has, X = _gbdt_inputs(ws, "test")
    country = pl.read_parquet(ws.table("test", "s1"), columns=["country"])["country"].to_numpy()
    seen = np.isin(country[has["s1"].to_numpy()], list(C.TRAINING_COUNTRIES))
    pn = bst.predict(xgb.DMatrix(X)).astype(np.float32)
    upd = has.select("s1", "r").with_columns(pl.Series("pn", pn)).filter(pl.Series(seen))
    fb = pl.read_parquet(os.path.join(model_dir(ws, "logistic_stack"), "test_pred.parquet")).select("s1", "r", pl.col("p").alias("pref"))
    res = t.select("s1", "r", "p").join(upd, on=["s1", "r"], how="left").join(fb, on=["s1", "r"], how="left") \
           .with_columns(pl.coalesce(["pn", "pref", "p"]).cast(pl.Float32).alias("p")).select("s1", "r", "p")
    res.write_parquet(os.path.join(out, "test_pred.parquet"))
    LOG.info("GBDT stack: %s training-country test pairs re-scored", f"{upd.height:,}")
