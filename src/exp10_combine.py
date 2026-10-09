"""exp10_combine: exp09 stage-2 probabilities + exp08 cross-encoder scores (seen countries only).

The cross-encoder (exp08) scored the uncertain band of exp07. exp09 (consensus features) produces new stage-2
probabilities. For pairs that have a CE score, a logistic stacker on [logit(p_exp09), ce, product] is cross-fitted
on train (S1 folds) and replaces p for SEEN countries only (the CE does not transfer to unseen countries, exp06);
all other pairs keep p_exp09. Unseen countries (France) keep p_exp09 with --unseen_thr.
Optional --ce_run2 (exp18/exp23, 09-27): extra cross-encoders (comma-separated runs); pairs that have ALL scores get
a multi-CE stacker on [logit p, ce, logit p * ce] + [ce_k, logit p * ce_k, ce * ce_k] per extra CE k, the others keep
the single-CE stacker.
Optional --t_empty (seen countries): an S1 with no pair above the threshold accepts its best assigned pair if
p > t_empty (validated on dense OOF: t_main 0.75 / t_empty 0.6 = +0.00008).
Stages: eval (dense OOF F0.5) | predict (test submission)
"""
import os
import sys
import json
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, save_json, s1_fold, write_outputs, validate_outputs  # noqa: E402
import exp03_dense as M3  # noqa: E402
import exp06_crossenc as E6  # noqa: E402

SEEN = {"US", "India"}


def with_ce(res, ce, extras=None):
    d = res.join(ce, on=["s1", "r"], how="left")
    for k, e in enumerate(extras or []):
        d = d.join(e.rename({"ce": f"ce{k + 2}"}), on=["s1", "r"], how="left")
    return d


def extra_cols(df):
    return sorted([c for c in df.columns if c.startswith("ce") and c[2:].isdigit()], key=lambda c: int(c[2:]))


def stack2_features(df):
    lp = E6.logit(df["p"].to_numpy())
    c1 = df["ce"].to_numpy()
    cols = [lp, c1, lp * c1]
    for c in extra_cols(df):
        ck = df[c].to_numpy()
        cols += [ck, lp * ck, c1 * ck]
    return np.stack(cols, 1)


def fit_stack2(df):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=3000).fit(stack2_features(df), df["y"].to_numpy())


def load_ce2(args, split):
    if not args.ce_run2:
        return None
    return [pl.read_parquet(os.path.join(RUNS, r, f"{split}_ce.parquet")).select("s1", "r", "ce")
            for r in args.ce_run2.split(",")]


def all_extra(extras):
    if extras is None:
        return pl.lit(False)
    return pl.all_horizontal([pl.col(f"ce{k + 2}").is_not_null() for k in range(len(extras))])


def decode_two(a, t_empty):
    """a: assigned pairs with per-S1 threshold t and seen flag. Accept p > t; with t_empty, a seen-country S1 left
    without any accept takes its best assigned pair if p > t_empty."""
    acc = a.filter(pl.col("p") > pl.col("t"))
    if t_empty is None:
        return acc
    extra = a.filter((pl.col("p") > t_empty) & (pl.col("p") <= pl.col("t")) & pl.col("seen")
                     & ~pl.col("s1").is_in(acc["s1"].unique().implode())) \
             .sort("p", descending=True).unique(subset=["s1"], keep="first")
    return pl.concat([acc, extra.select(acc.columns)])


def stage_eval(args, log):
    keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
    oof = pl.read_parquet(os.path.join(RUNS, args.base, "oof.parquet"))
    ce = pl.read_parquet(os.path.join(RUNS, args.ce_run, "train_ce.parquet"))
    ce2 = load_ce2(args, "train")
    d = with_ce(oof, ce, ce2)
    has = d.filter(pl.col("ce").is_not_null())
    both = all_extra(ce2)
    parts = []
    for name, sub, fit, feat in [("single", has.filter(~both), E6.fit_stacker, E6.stack_features),
                                 ("double", has.filter(both), fit_stack2, stack2_features)]:
        if sub.height == 0:
            continue
        fold = s1_fold(len(keep))[sub["s1"].to_numpy()]
        newp = np.zeros(sub.height, np.float32)
        for k in (0, 1):
            lr = fit(sub.filter(pl.Series(fold == k)))
            newp[fold != k] = lr.predict_proba(feat(sub.filter(pl.Series(fold != k))))[:, 1]
            log(f"{name} stacker ({sub.height:,} pairs) fold {k}: coef {np.round(lr.coef_[0], 3)}")
        parts.append(sub.select("s1", "r").with_columns(pl.Series("pn", newp)))
    new = oof.join(pl.concat(parts), on=["s1", "r"], how="left") \
             .with_columns(pl.coalesce(["pn", "p"]).alias("p")).drop("pn")
    ref = M3.decode_eval(oof, keep, log, f"{args.base} (reference)")
    m = M3.decode_eval(new, keep, log, f"{args.base} + {args.ce_run} CE")
    new.select("s1", "r", "p").write_parquet(os.path.join(RUNS, args.run, "oof_combined.parquet"))
    import pickle
    with open(os.path.join(RUNS, args.run, "stacker.pkl"), "wb") as fh:
        pickle.dump(E6.fit_stacker(has.filter(~both)), fh)
    if ce2 is not None:
        with open(os.path.join(RUNS, args.run, "stacker2.pkl"), "wb") as fh:
            pickle.dump(fit_stack2(has.filter(both)), fh)
    save_json({"reference": ref, "combined": m}, os.path.join(RUNS, args.run, "metrics.json"))


def stage_predict(args, log):
    import pickle
    with open(os.path.join(RUNS, args.run, "stacker.pkl"), "rb") as fh:
        lr = pickle.load(fh)
    with open(os.path.join(RUNS, args.run, "metrics.json")) as fh:
        thr = args.thr if args.thr is not None else json.load(fh)["combined"]["best_thr"]
    tp = pl.read_parquet(os.path.join(RUNS, args.base, "test_pred.parquet"))
    ce = pl.read_parquet(os.path.join(RUNS, args.ce_run, "test_ce.parquet"))
    s1 = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["eid", "country"])
    country = s1["country"].to_numpy()
    ce2 = load_ce2(args, "test")
    d = with_ce(tp, ce, ce2)
    has = d.filter(pl.col("ce").is_not_null() & pl.Series(np.isin(country[d["s1"].to_numpy()], list(SEEN))))
    both = all_extra(ce2)
    one = has.filter(~both)
    parts = [one.select("s1", "r").with_columns(
        pl.Series("pn", lr.predict_proba(E6.stack_features(one))[:, 1].astype(np.float32)))]
    if ce2 is not None:
        with open(os.path.join(RUNS, args.run, "stacker2.pkl"), "rb") as fh:
            lr2 = pickle.load(fh)
        two = has.filter(both)
        parts.append(two.select("s1", "r").with_columns(
            pl.Series("pn", lr2.predict_proba(stack2_features(two))[:, 1].astype(np.float32))))
        log(f"test: single-CE stacker {one.height:,} pairs, double-CE stacker {two.height:,} pairs")
    res = tp.join(pl.concat(parts), on=["s1", "r"], how="left") \
            .with_columns(pl.coalesce(["pn", "p"]).alias("p")).drop("pn")
    res.write_parquet(os.path.join(RUNS, args.run, "test_pred.parquet"))
    seen = np.isin(country, list(SEEN))
    t = np.where(seen, thr, args.unseen_thr).astype(np.float32)
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.with_columns(pl.Series("t", t[a["s1"].to_numpy()]), pl.Series("seen", seen[a["s1"].to_numpy()]))
    a = decode_two(a, args.t_empty)
    r_ids = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(RUNS, args.run, "output")
    write_outputs(out_dir, s1["eid"].to_numpy(), a["s1"].to_numpy(), a["r"].to_numpy(),
                  res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    n = np.bincount(a["s1"].to_numpy(), minlength=len(country))
    for c in np.unique(country):
        m = country == c
        log(f"test {c}: thr {thr if c in SEEN else args.unseen_thr} mean matches {n[m].mean():.3f} empty {np.mean(n[m] == 0):.4f}")
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eval", "predict"])
    ap.add_argument("--run", default="exp10")
    ap.add_argument("--base", default="exp09")
    ap.add_argument("--ce_run", default="exp08")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    ap.add_argument("--ce_run2", default="", help="optional second cross-encoder run (train_ce/test_ce.parquet)")
    ap.add_argument("--t_empty", type=float, default=None, help="seen countries: rescue threshold for S1 with no accept")
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    {"eval": stage_eval, "predict": stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
