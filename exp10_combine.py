"""exp10_combine: exp09 stage-2 probabilities + exp08 cross-encoder scores (seen countries only).

The cross-encoder (exp08) scored the uncertain band of exp07. exp09 (consensus features) produces new stage-2
probabilities. For pairs that have a CE score, a logistic stacker on [logit(p_exp09), ce, product] is cross-fitted
on train (S1 folds) and replaces p for SEEN countries only (the CE does not transfer to unseen countries, exp06);
all other pairs keep p_exp09. Unseen countries (France) keep p_exp09 with --unseen_thr.
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


def with_ce(res, ce):
    return res.join(ce, on=["s1", "r"], how="left")


def stage_eval(args, log):
    keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
    oof = pl.read_parquet(os.path.join(RUNS, args.base, "oof.parquet"))
    ce = pl.read_parquet(os.path.join(RUNS, args.ce_run, "train_ce.parquet"))
    d = with_ce(oof, ce)
    has = d.filter(pl.col("ce").is_not_null())
    fold = s1_fold(len(keep))[has["s1"].to_numpy()]
    newp = np.zeros(has.height, np.float32)
    for k in (0, 1):
        lr = E6.fit_stacker(has.filter(pl.Series(fold == k)))
        newp[fold != k] = lr.predict_proba(E6.stack_features(has.filter(pl.Series(fold != k))))[:, 1]
        log(f"stacker fold {k}: coef {np.round(lr.coef_[0], 3)}")
    new = oof.join(has.select("s1", "r").with_columns(pl.Series("pn", newp)), on=["s1", "r"], how="left") \
             .with_columns(pl.coalesce(["pn", "p"]).alias("p")).drop("pn")
    ref = M3.decode_eval(oof, keep, log, f"{args.base} (reference)")
    m = M3.decode_eval(new, keep, log, f"{args.base} + {args.ce_run} CE")
    import pickle
    with open(os.path.join(RUNS, args.run, "stacker.pkl"), "wb") as fh:
        pickle.dump(E6.fit_stacker(has), fh)
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
    d = with_ce(tp, ce)
    has = d.filter(pl.col("ce").is_not_null() & pl.Series(np.isin(country[d["s1"].to_numpy()], list(SEEN))))
    has = has.with_columns(pl.Series("pn", lr.predict_proba(E6.stack_features(has))[:, 1].astype(np.float32)))
    res = tp.join(has.select("s1", "r", "pn"), on=["s1", "r"], how="left") \
            .with_columns(pl.coalesce(["pn", "p"]).alias("p")).drop("pn")
    res.write_parquet(os.path.join(RUNS, args.run, "test_pred.parquet"))
    t = np.where(np.isin(country, list(SEEN)), thr, args.unseen_thr).astype(np.float32)
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.with_columns(pl.Series("t", t[a["s1"].to_numpy()])).filter(pl.col("p") > pl.col("t"))
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
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    {"eval": stage_eval, "predict": stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
