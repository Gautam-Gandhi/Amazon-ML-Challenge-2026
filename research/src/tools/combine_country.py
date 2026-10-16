"""Assemble a test submission from different pipeline stages per country.

Seen countries (in training) use --seen_src probabilities with --thr; unseen countries (France) use --unseen_src with
--unseen_thr. Sources: "stage2" = runs/<run>/test_pred.parquet (or --seen_pred file), "stage1" = the p1 column of the
stage-2 feature parts (row-aligned with the stage-1 test feature parts).
Usage: python tools/combine_country.py --run exp07 --s2dir feat_v7s2/test --s1dir feat_v1/test \
           --seen_src stage2 --unseen_src stage1 --thr 0.7 --unseen_thr 0.9 --out exp07_fr_s1
"""
import os
import sys
import glob
import argparse
import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from er_common import CACHE, RUNS, write_outputs, validate_outputs  # noqa: E402

SEEN = {"US", "India"}


def stage1_probs(s1dir, s2dir):
    k = [pl.read_parquet(p, columns=["s1", "r"]) for p in sorted(glob.glob(os.path.join(CACHE, s1dir, "part*.parquet")))]
    p = [pl.read_parquet(p, columns=["p1"]) for p in sorted(glob.glob(os.path.join(CACHE, s2dir, "part*.parquet")))]
    assert len(k) == len(p)
    return pl.concat([pl.concat([a, b], how="horizontal") for a, b in zip(k, p)]).rename({"p1": "p"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seen_pred", default="", help="file under RUNS for seen-country stage-2 (e.g. exp08_exp05/test_pred.parquet)")
    ap.add_argument("--s1dir", default="feat_v1/test")
    ap.add_argument("--s2dir", default="feat_v7s2/test")
    ap.add_argument("--seen_src", default="stage2", choices=["stage1", "stage2"])
    ap.add_argument("--unseen_src", default="stage1", choices=["stage1", "stage2"])
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s1 = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["eid", "country"])
    country = s1["country"].to_numpy()
    st2 = pl.read_parquet(os.path.join(RUNS, args.seen_pred or os.path.join(args.run, "test_pred.parquet")), columns=["s1", "r", "p"])
    st1 = stage1_probs(args.s1dir, args.s2dir) if "stage1" in (args.seen_src, args.unseen_src) else None
    src = {"stage1": st1, "stage2": st2}
    seen_mask = lambda d: pl.Series(np.isin(country[d["s1"].to_numpy()], list(SEEN)))
    a = src[args.seen_src]
    b = src[args.unseen_src]
    res = pl.concat([a.filter(seen_mask(a)), b.filter(~seen_mask(b))]).sort(["s1", "r"])
    t = np.where(np.isin(country, list(SEEN)), args.thr, args.unseen_thr).astype(np.float32)
    best = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    best = best.with_columns(pl.Series("t", t[best["s1"].to_numpy()])).filter(pl.col("p") > pl.col("t"))
    r_ids = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(RUNS, args.out, "output")
    write_outputs(out_dir, s1["eid"].to_numpy(), best["s1"].to_numpy(), best["r"].to_numpy(),
                  res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    n = np.bincount(best["s1"].to_numpy(), minlength=len(country))
    for c in np.unique(country):
        m = country == c
        print(f"{c}: mean matches {n[m].mean():.3f} empty {np.mean(n[m] == 0):.4f}")
    with open(os.path.join(RUNS, args.out, "README.txt"), "w") as f:
        f.write(f"{vars(args)}\n")
    res.write_parquet(os.path.join(RUNS, args.out, "test_pred.parquet"))
    print("VALID" if validate_outputs(out_dir) else "INVALID")


if __name__ == "__main__":
    main()
