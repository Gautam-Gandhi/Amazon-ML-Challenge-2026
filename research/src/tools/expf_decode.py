"""Expected-F0.5 decoding per Source 1 entity (plug-in, independence approximation), validated on the dense OOF.

For an S1 with assigned candidates sorted by p (p_1 >= p_2 >= ...), accepting the top k gives expected F0.5
    E[F(k)] ~= 1.25 * sum_{i<=k} p_i / (0.25 * (N + m) + k),   N = sum_i p_i (expected true matches among candidates)
where m is the expected number of true matches missing from the candidate set (a per-S1 constant, tuned);
accepting nothing gives P(S1 is a singleton) ~= prod_i (1 - p_i) * s0 (s0 tuned: missing matches make true
singletons rarer than the product suggests). The k with the highest expected F is chosen per S1.
Usage: ER_WORK_DIR=work_v3 python tools/expf_decode.py work_v3/runs/exp20c/oof_combined.parquet
"""
import os
import sys
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp03_dense as M3  # noqa: E402


def expf_select(a, m=0.05, s0=1.0, cap=0.999):
    """a: assigned pairs (s1, r, p). Returns the accepted pairs."""
    a = a.with_columns(pl.col("p").clip(1e-6, cap)).sort(["s1", "p"], descending=[False, True])
    s1 = a["s1"].to_numpy()
    p = a["p"].to_numpy().astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(s1)) + 1]
    ends = np.r_[starts[1:], len(s1)]
    keep = np.zeros(len(p), bool)
    for st, en in zip(starts, ends):
        q = p[st:en]
        cs = np.cumsum(q)
        N = cs[-1] + m
        k = np.arange(1, len(q) + 1)
        ef = 1.25 * cs / (0.25 * N + k)
        e0 = s0 * np.prod(1 - q)
        best = int(np.argmax(ef))
        if ef[best] > e0:
            keep[st:st + best + 1] = True
    return a.filter(pl.Series(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof")
    args = ap.parse_args()
    keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
    gt = M3.labels_kept(keep)
    res = pl.read_parquet(args.oof).select("s1", "r", "p")
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    F = lambda d: M3.f05_world(d["s1"].to_numpy(), d["r"].to_numpy(), gt, keep)
    base = a.filter(pl.col("p") > 0.7)
    print(f"threshold 0.7: {F(base):.5f}")
    for m in [0.0, 0.05, 0.15]:
        for s0 in [0.8, 1.0, 1.3]:
            print(f"expected-F decode m={m} s0={s0}: {F(expf_select(a, m, s0)):.5f}", flush=True)


if __name__ == "__main__":
    main()
