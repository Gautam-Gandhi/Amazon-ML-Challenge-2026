"""exp33_descswap: reject descriptor-for-descriptor swaps at the same house number in unseen countries (France).

Why (09-27): in France, "X Club SARL" and "X Ecole SARL" at the same address are sibling organisations (distractors).
LB evidence: accepting the rejected ones (exp14rc/rcd, ~26K flips) cost -0.0036, i.e. ~100% false. But the model still
ACCEPTS 2,081 such pairs (p > 0.9; 93% descriptor -> descriptor), e.g. "Pessac Club SARL" -> "Pessac Ecole SARL",
"Ecuries Ecole SASU" -> "Ecuries College SASU": it has no French vocabulary, so its p inside this type carries no truth.
Descriptor vocabulary, label-free, per unseen country: a word is a descriptor if it is in >= --min_s1 S1 names, is NOT
over-represented in S2/S3 names relative to S1 names (ratio < --max_over; injected noise words are over-represented:
associes 77x, developpement 11x, groupe 4.7x, services 2.4x, fils 1.6x, france 1.2x; every descriptor 0.80-0.91x), and is
the added word of >= --min_swaps one-word swaps at the same house number.
Rule: unseen-country pair, name_core differs by exactly one word swapped (S1 word a -> R word b), a and b both
descriptors, first house numbers equal and present -> p := p * --demote.
Decode = exp31.decode_write (unchanged thresholds).
"""
import os
import sys
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger  # noqa: E402
import exp31_sibprior as E31  # noqa: E402

SEEN = {"US", "India"}


def descriptor_vocab(s1, r, country, pairs_swapped_in, args, log):
    """pairs_swapped_in: Series of R words added in same-number one-word swaps (this country)."""
    s = s1.filter(pl.col("country") == country)
    rr = r.filter(pl.col("country") == country)
    cnt = lambda d: d.select(pl.col("name_core").str.split(" ").list.unique().alias("w")).explode("w").group_by("w").len()
    a = cnt(s).rename({"len": "n_s1"})
    b = cnt(rr).rename({"len": "n_r"})
    m = a.join(b, on="w", how="left").fill_null(0).with_columns(
        ((pl.col("n_r") / rr.height) / (pl.col("n_s1") / s.height)).alias("over"))
    sw = pairs_swapped_in.value_counts().rename({pairs_swapped_in.name: "w", "count": "n_swap"})
    m = m.join(sw, on="w", how="left").fill_null(0)
    v = m.filter((pl.col("n_s1") >= args.min_s1) & (pl.col("over") < args.max_over) & (pl.col("n_swap") >= args.min_swaps)
                 & (pl.col("w").str.len_chars() >= 3) & ~pl.col("w").is_in(args.exclude.split(",")))
    log(f"{country}: descriptor vocabulary {v.height} words: {sorted(v['w'].to_list())}")
    return set(v["w"].to_list())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp33")
    ap.add_argument("--base", default="exp32")
    ap.add_argument("--metrics_run", default="exp29g")
    ap.add_argument("--min_s1", type=int, default=200)
    ap.add_argument("--max_over", type=float, default=1.0)
    ap.add_argument("--min_swaps", type=int, default=200)
    ap.add_argument("--exclude", default="compagnie",
                    help="words kept out of the vocabulary: 'compagnie' is the spelled-out 'Cie' filler (records add 'cie' to "
                         "the legal suffix, and the model accepts 51%% of descriptor->compagnie swaps vs 5-20%% for descriptors)")
    ap.add_argument("--demote", type=float, default=0.5)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    ap.add_argument("--t_empty", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, "log_predict.txt"))
    log("args:", vars(args))
    tp = pl.read_parquet(os.path.join(RUNS, args.base, "test_pred.parquet")).select("s1", "r", "p")
    P = os.path.join(CACHE, "prep_v2")
    cols = ["country", "name_core", "addr_nums"]
    s1 = pl.read_parquet(os.path.join(P, "test_s1.parquet"), columns=cols)
    r = pl.read_parquet(os.path.join(P, "test_r.parquet"), columns=cols)
    si, ri = tp["s1"].to_numpy(), tp["r"].to_numpy()
    A = s1["name_core"].gather(si).str.split(" ").list.unique()
    B = r["name_core"].gather(ri).str.split(" ").list.unique()
    t = tp.with_row_index("i").with_columns(
        pl.Series("country", s1["country"].to_numpy()[si]),
        A.list.set_difference(B).alias("miss"), B.list.set_difference(A).alias("extra"),
        s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null("").alias("h1"),
        r["addr_nums"].gather(ri).str.split(" ").list.first().fill_null("").alias("h2"))
    t = t.filter(~pl.col("country").is_in(list(SEEN)) & (pl.col("miss").list.len() == 1) & (pl.col("extra").list.len() == 1)
                 & (pl.col("h1") != "") & (pl.col("h1") == pl.col("h2")))
    t = t.with_columns(pl.col("miss").list.first().alias("a"), pl.col("extra").list.first().alias("b"))
    idx = []
    for c in sorted(t["country"].unique().to_list()):
        tc = t.filter(pl.col("country") == c)
        vocab = descriptor_vocab(s1, r, c, tc["b"].rename("b"), args, log)
        hit = tc.filter(pl.col("a").is_in(list(vocab)) & pl.col("b").is_in(list(vocab)))
        log(f"{c}: same-number descriptor swaps {hit.height:,}; with p > {args.unseen_thr}: {(hit['p'] > args.unseen_thr).sum():,} "
            f"(p hist >0.9: {np.histogram(hit['p'].to_numpy(), bins=[0.9, 0.95, 0.98, 0.99, 1.01])[0].tolist()})")
        idx.append(hit["i"].to_numpy())
    idx = np.concatenate(idx) if idx else np.array([], dtype=np.int64)
    p = tp["p"].to_numpy().copy()
    p[idx] = p[idx] * args.demote
    E31.decode_write(tp.with_columns(pl.Series("p", p.astype(np.float32))), args, log)


if __name__ == "__main__":
    main()
