"""exp17_cefill: extend cross-encoder (exp08) coverage to the uncertain band of the CURRENT stage-2 model (exp15).

Why (09-27): exp08 scored the band (0.01, 0.99) of the old exp07 model. On train that band covers exp15's uncertain
pairs almost fully (p in (0.05, 0.95): 97.4%), but on TEST only 78.7% (US/India) -> ~250K uncertain test pairs get no
CE score, so the test file gets less of the CE gain (+0.0017 dense) than validation says.
This run scores the missing pairs with the SAVED exp08 fold cross-encoders (no training):
  train: pairs of exp15 OOF with p in (lo, hi) and no CE score, scored out-of-fold (fold-k model scores S1 fold != k,
         exactly as exp08 did), so the stacker is refit on honest scores
  test : seen-country pairs of exp15 test_pred with p in (lo, hi) and no CE score, fold-average (as exp08)
Output: runs/exp17/{train,test}_ce.parquet = exp08 scores + the new ones (consumed by exp10_combine --ce_run exp17).
Stage: fill
"""
import os
import sys
import argparse

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, s1_fold  # noqa: E402
import exp06_crossenc as E6  # noqa: E402

SEEN = {"US", "India"}


def texts(pairs, split):
    s1 = pl.read_parquet(os.path.join(E6.PREP[split], f"{split}_s1.parquet"), columns=["name_raw", "addr_raw"])
    r = pl.read_parquet(os.path.join(E6.PREP[split], f"{split}_r.parquet"), columns=["name_raw", "addr_raw"])
    a, b = s1[pairs["s1"].to_numpy()], r[pairs["r"].to_numpy()]
    ta = (a["name_raw"].str.to_lowercase() + " ; " + a["addr_raw"].str.to_lowercase()).to_list()
    tb = (b["name_raw"].str.to_lowercase() + " ; " + b["addr_raw"].str.to_lowercase()).to_list()
    return ta, tb


def stage_fill(args, log):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = torch.device("cuda")
    out_dir = os.path.join(RUNS, args.run)
    ce_dir = os.path.join(RUNS, args.ce_run)
    tok = AutoTokenizer.from_pretrained(E6.MODEL, revision=E6.MODEL_REV)
    models = []
    for k in (0, 1):
        models.append(AutoModelForSequenceClassification.from_pretrained(
            os.path.join(ce_dir, f"ce_fold{k}"), attn_implementation="eager").to(dev).eval())
    for split, src in [("train", "oof.parquet"), ("test", "test_pred.parquet")]:
        base = pl.read_parquet(os.path.join(RUNS, args.base, src)).select("s1", "r", "p")
        have = pl.read_parquet(os.path.join(ce_dir, f"{split}_ce.parquet"))
        band = base.filter((pl.col("p") > args.lo) & (pl.col("p") < args.hi))
        if split == "test":
            country = pl.read_parquet(os.path.join(E6.PREP["test"], "test_s1.parquet"), columns=["country"])["country"].to_numpy()
            band = band.filter(pl.Series(np.isin(country[band["s1"].to_numpy()], list(SEEN))))
        miss = band.join(have.select("s1", "r"), on=["s1", "r"], how="anti").select("s1", "r")
        log(f"{split}: band ({args.lo},{args.hi}) {band.height:,} pairs, {miss.height:,} without a CE score")
        ta, tb = texts(miss, split)
        ids = E6.encode(tok, ta, tb, args.max_len)
        ce = np.zeros(len(ids), np.float32)
        with torch.no_grad():
            if split == "train":
                fold = s1_fold(pl.read_parquet(os.path.join(E6.PREP["train"], "train_s1.parquet"), columns=["country"]).height)[miss["s1"].to_numpy()]
                for k in (0, 1):                       # fold-k model was trained on S1 fold k -> scores fold != k
                    idx = np.where(fold != k)[0]
                    if len(idx):
                        ce[idx] = E6.score(tok, models[k], ids, idx, 128, dev)
            else:
                for k in (0, 1):
                    ce += E6.score(tok, models[k], ids, np.arange(len(ids)), 128, dev) / 2
        new = miss.with_columns(pl.Series("ce", ce))
        allce = pl.concat([have.select("s1", "r", "ce"), new.with_columns(pl.col("s1").cast(have["s1"].dtype), pl.col("r").cast(have["r"].dtype))])
        allce.write_parquet(os.path.join(out_dir, f"{split}_ce.parquet"))
        log(f"{split}: CE scores {have.height:,} -> {allce.height:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fill"])
    ap.add_argument("--run", default="exp17")
    ap.add_argument("--base", default="exp15")
    ap.add_argument("--ce_run", default="exp08")
    ap.add_argument("--lo", type=float, default=0.01)
    ap.add_argument("--hi", type=float, default=0.99)
    ap.add_argument("--max_len", type=int, default=96)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    from er_common import set_determinism
    set_determinism(0)
    stage_fill(args, log)


if __name__ == "__main__":
    main()
