"""Per-S1 expected-F0.5-optimal decoding (Monte Carlo) vs global threshold.

After assignment (each R kept for its argmax-p S1), for every S1 consider the prefixes of its candidates sorted by
p (including the empty set) and pick the one that maximizes E[F0.5] under independent Bernoulli(p_i) labels
(+ `miss` expected true matches outside the candidate set). Optional calibration: p <- sigmoid(a*logit(p)+b).
Usage: python tools/decode_opt.py <run> [--samples 64] [--miss 0.05]   (dense world k80s0 evaluation)
Also importable: decode_mc(res, ...) -> (s1, r) arrays.
"""
import os
import sys
import argparse
import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from er_common import CACHE, RUNS, macro_f05  # noqa: E402

B2 = 0.25


def decode_mc(res, samples=64, miss=0.0, pmin=0.05, maxc=24, seed=0, chunk=200000):
    a = res.select("s1", "r", "p").filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.filter(pl.col("p") > pmin).sort(["s1", "p"], descending=[False, True])
    a = a.with_columns(pl.int_range(pl.len()).over("s1").alias("k")).filter(pl.col("k") < maxc)
    s1u, inv = np.unique(a["s1"].to_numpy(), return_inverse=True)
    k = a["k"].to_numpy()
    P = np.zeros((len(s1u), maxc), np.float32)
    P[inv, k] = a["p"].to_numpy()
    rng = np.random.default_rng(seed)
    best_j = np.zeros(len(s1u), np.int32)
    for st in range(0, len(s1u), chunk):
        Pc = P[st:st + chunk]
        n = Pc.shape[0]
        Y = rng.random((n, samples, maxc), dtype=np.float32) < Pc[:, None, :]          # (n, S, C)
        M = rng.poisson(miss, (n, samples)) if miss > 0 else 0
        ntrue = Y.sum(2) + M                                                            # (n, S)
        tp = np.cumsum(Y, 2)                                                            # TP of prefix j+1
        j = np.arange(1, maxc + 1)
        F = (1 + B2) * tp / (j[None, None, :] + B2 * ntrue[:, :, None])
        EF = F.mean(1)                                                                  # (n, C)
        EF[Pc == 0] = -1                                                                # padded slots
        e0 = (ntrue == 0).mean(1)                                                       # empty prediction
        jb = EF.argmax(1)
        best_j[st:st + n] = np.where(EF[np.arange(n), jb] > e0, jb + 1, 0)
    keep = k < best_j[inv]
    return a["s1"].to_numpy()[keep], a["r"].to_numpy()[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--miss", type=float, default=0.0)
    ap.add_argument("--oof", default="oof.parquet")
    args = ap.parse_args()
    import exp03_dense as M3
    keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
    gt = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_gt.parquet"))
    gt = gt.filter(pl.Series(keep[gt["s1_idx"].to_numpy()]))
    res = pl.read_parquet(os.path.join(RUNS, args.run, args.oof))
    n1 = len(keep)

    def f(s, r):
        per = macro_f05(s, r, gt["s1_idx"].to_numpy(), gt["r_idx"].to_numpy(), n1, return_per_entity=True)
        return per[keep].mean()
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    for thr in (0.6, 0.7, 0.8):
        d = a.filter(pl.col("p") > thr)
        print(f"global thr {thr}: F0.5 {f(d['s1'].to_numpy(), d['r'].to_numpy()):.5f}", flush=True)
    for miss in sorted({0.0, args.miss, 0.05, 0.1}):
        s, r = decode_mc(res, samples=args.samples, miss=miss)
        print(f"MC expected-F0.5 decoding (samples {args.samples}, miss {miss}): F0.5 {f(s, r):.5f}", flush=True)


if __name__ == "__main__":
    main()
