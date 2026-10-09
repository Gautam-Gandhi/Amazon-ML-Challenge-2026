"""exp35_acrocand: acronym candidate channel for unseen countries (France) - fixes a blocking leak.

Why (09-27, Step 2 audit): pruning keeps a pair only if it is in the S1's top-15 or the record's top-2 by JOINT
(name + address) similarity. An acronym record ("VC" for "Volley Comite EURL") has ~0 name similarity, so on dense
French streets it is out-ranked by siblings and neighbours. French acronym records whose initials match exactly one S1
at the same house number and street (a near-certain match by construction): 18,018 pairs, 94.6% found by the search but
only 79.3% survive pruning -> 3,724 never reach the matcher (US/India: ~98.5% survive). The same join on labeled data is
99.9% (US) / 98.9% (India) true; sampled French lost pairs are clear matches ("FA" <- "Federation des Anges", 52 BIS
Route de la Courance). exp32 already accepts such pairs when they ARE candidates.
Rule (unseen countries): add the join pairs that are missing from the candidate set with p := --p_set, unless the record
is already accepted elsewhere (any p > --unseen_thr). They are added to candidate_pairs.tsv too (+0.014 per French S1).
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
from exp32_acro import initials, SEEN  # noqa: E402


def street(df):
    return df.with_columns(
        pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("h"),
        pl.col("addr_clean").str.split(" ").list.eval(
            pl.element().filter(~pl.element().str.contains(r"\d") & (pl.element().str.len_chars() > 1)
                                & ~pl.element().str.starts_with("fr"))).alias("w"))


def acro_join(countries, log):
    P = os.path.join(CACHE, "prep_v2")
    cols = ["country", "name_core", "addr_clean", "addr_nums"]
    s1 = pl.read_parquet(os.path.join(P, "test_s1.parquet"), columns=cols).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32))
    r = pl.read_parquet(os.path.join(P, "test_r.parquet"), columns=cols).with_row_index("r").with_columns(pl.col("r").cast(pl.Int32))
    s1 = street(s1.filter(pl.col("country").is_in(countries)))
    r = street(r.filter(pl.col("country").is_in(countries)))
    s1 = s1.with_columns(pl.col("name_core").map_elements(lambda c: initials(c, False), return_dtype=pl.Utf8).alias("i1"),
                         pl.col("name_core").map_elements(lambda c: initials(c, True), return_dtype=pl.Utf8).alias("i2"))
    ra = r.filter(pl.col("name_core").str.contains(r"^[a-z]{2,5}$") & (pl.col("h") != ""))
    pa = pl.concat([ra.join(s1.select("s1", "country", pl.col(i).alias("name_core"), "h", pl.col("w").alias("w1")),
                            on=["country", "name_core", "h"]) for i in ("i1", "i2")]).unique(subset=["s1", "r"])
    pa = pa.with_columns((pl.col("w").list.set_intersection("w1").list.len()
                          / pl.col("w").list.set_union("w1").list.len().clip(1, None)).alias("j")).filter(pl.col("j") >= 0.5)
    pa = pa.filter(pl.len().over("r") == 1).select("s1", "r", "country")
    log(f"acronym join pairs (unique S1, same number + street): {pa.height:,} by country {sorted(pa.group_by('country').len().rows())}")
    return pa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp35")
    ap.add_argument("--base", default="exp33")
    ap.add_argument("--metrics_run", default="exp29g")
    ap.add_argument("--p_set", type=float, default=0.95)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    ap.add_argument("--t_empty", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, "log_predict.txt"))
    log("args:", vars(args))
    tp = pl.read_parquet(os.path.join(RUNS, args.base, "test_pred.parquet")).select("s1", "r", "p")
    ctry = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"]
    unseen = sorted(set(ctry.unique().to_list()) - SEEN)
    pa = acro_join(unseen, log)
    new = pa.join(tp.select("s1", "r"), on=["s1", "r"], how="anti")
    busy = tp.filter(pl.col("p") > args.unseen_thr).select("r").unique()
    add = new.join(busy, on="r", how="anti")
    log(f"join pairs missing from the candidate set: {new.height:,}; skipped (record accepted elsewhere): {new.height - add.height:,}; "
        f"added with p={args.p_set}: {add.height:,}")
    res = pl.concat([tp, add.select(pl.col("s1").cast(tp["s1"].dtype), pl.col("r").cast(tp["r"].dtype),
                                    pl.lit(args.p_set, dtype=pl.Float32).alias("p"))])
    E31.decode_write(res, args, log)


if __name__ == "__main__":
    main()
