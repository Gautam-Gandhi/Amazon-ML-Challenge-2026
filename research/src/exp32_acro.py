"""exp32_acro: acronym rule for unseen countries (France).

Why (09-27): the pipeline has no acronym feature, so an acronym record ("VC" for "Volley Comite EURL", "CLC" for
"Combattants Liger Club SARL") only gets address evidence. The French generator uses acronyms ~17x more often than the
US one (accepted per 1K S1: France 48.8, India 9.5, US 2.8), and French acronym pairs sit in the uncertain band
(p hist of same-number unique-acronym pairs: 145 / 257 / 366 / 484 / 1,509 in (0,.1] (.1,.3] (.3,.5] (.5,.7] (.7,.9]).
Evidence the type is near-certain, including OUT of country (unlike random brand names, domains or identical-name
types, whose held-out-country true rate tracks p):
  in-country (dense OOF): same house number, unique acronym-matching S1: 99.86% true (8,577 pairs); p (0.1,0.3] 21/21
  LOCO US->India (same type, R's best S1): p (0.1,0.3] 0.93, (0.3,0.5] 1.00, (0.5,0.7] 1.00, (0.7,0.8] 1.00, (0.8,0.9] 1.00
  LOCO India->US: 1.00 in every bin; only p <= 0.05 is mostly false (US->India 0.025)
Rule (unseen countries only, like exp11): R name_core is one 2-5 letter token equal to the initials of the S1
name_core words (all words, or without articles/'and'/'et'); the first house numbers are equal; exactly one of the
R's candidate S1s has matching initials and it is the R's best S1; p in (--p_floor, --unseen_thr] -> p := --p_set.
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
STOP = {"la", "le", "les", "de", "du", "des", "d", "l", "et", "and", "of", "the", "a", "en", "au", "aux"}


def initials(core, drop_stop):
    toks = [t for t in core.split() if t]
    if drop_stop:
        toks = [t for t in toks if t not in STOP]
    return "".join(t[0] for t in toks)


def acro_flips(pred, split, countries_excluded, p_floor, p_hi, log):
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    s1 = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=["country", "name_core", "addr_nums"])
    r = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=["name_core", "addr_nums"])
    rc = r["name_core"].to_numpy()
    short = np.array([(2 <= len(x) <= 5 and " " not in x and x.isalpha()) for x in rc])
    t = pred.with_row_index("i").filter(pl.Series(short[pred["r"].to_numpy()]))
    t = t.with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("rk"))
    si, ri = t["s1"].to_numpy(), t["r"].to_numpy()
    s1c = s1["name_core"].to_numpy()
    rn = rc[ri]
    acro = np.array([(initials(s1c[s], False) == x) or (initials(s1c[s], True) == x) for s, x in zip(si, rn)])
    t = t.with_columns(pl.Series("acro", acro))
    t = t.with_columns(pl.col("acro").sum().over("r").alias("n_acro"))
    t = t.filter(pl.col("acro"))
    si, ri = t["s1"].to_numpy(), t["r"].to_numpy()
    h1 = s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null("")
    h2 = r["addr_nums"].gather(ri).str.split(" ").list.first().fill_null("")
    t = t.with_columns(pl.Series("country", s1["country"].to_numpy()[si]), h1.alias("h1"), h2.alias("h2"))
    f = t.filter(~pl.col("country").is_in(list(countries_excluded)) & (pl.col("h1") != "") & (pl.col("h1") == pl.col("h2"))
                 & (pl.col("n_acro") == 1) & (pl.col("rk") == 1) & (pl.col("p") > p_floor) & (pl.col("p") <= p_hi))
    log(f"[{split}] acronym pairs (same number, unique, best S1) to flip: {f.height:,}; by country {f.group_by('country').len().rows()}")
    return f["i"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp32")
    ap.add_argument("--base", default="exp31")
    ap.add_argument("--metrics_run", default="exp29g")
    ap.add_argument("--p_floor", type=float, default=0.1)
    ap.add_argument("--p_set", type=float, default=0.95)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    ap.add_argument("--t_empty", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, "log_predict.txt"))
    log("args:", vars(args))
    tp = pl.read_parquet(os.path.join(RUNS, args.base, "test_pred.parquet")).select("s1", "r", "p")
    idx = acro_flips(tp, "test", SEEN, args.p_floor, args.unseen_thr, log)
    p = tp["p"].to_numpy().copy()
    p[idx] = np.maximum(p[idx], args.p_set)
    E31.decode_write(tp.with_columns(pl.Series("p", p.astype(np.float32))), args, log)


if __name__ == "__main__":
    main()
