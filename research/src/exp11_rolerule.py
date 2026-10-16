"""exp11_rolerule: word-role rule for unseen countries (France), on top of exp10 test predictions.

Why (09-26 analysis, EXPERIMENTS.md "exp11"): the generator adds words to S2/S3 names in two roles, and each country
has its own vocabulary for both:
  noise  : the word REPLACES one descriptor of the S1 name, same address   ("Obrien Lion LLC" -> "Obrien LLC Services")
  family : the full S1 name + the word, usually another house number = a sister-company distractor ("Falcon Group")
Train labels: the noise pattern (swap + equal first house number, word with a noise role) is a true match 99.9% (US)
/ 98.5% (India) of the time, and the in-country model already accepts ~99% of it. In France the model rejects 76%
of that pattern, because the French noise words (fils, associes, and the dual-use groupe/developpement/france, like
US "partners") never occur in training.
Role statistics of a word t are computed per split and country from the split's own records, no labels:
  role_swap(t) = among assigned pairs whose names differ by exactly the single extra R token t (+ <= 1 missing S1
                 token), the share that are swaps (one S1 token missing);  nsc(t) = R-vs-S1 over-representation
Rule: an assigned pair of an UNSEEN country is accepted (p := max(p, --p_set)) when it is a swap of a noise-role
word (role_swap >= 0.1, nsc > 0.1, >= 200 occurrences) with an equal first house number and p > --p_floor.
Seen countries (US/India) are unchanged: their model already knows these words.
--pattern desc (exp11c probe): instead, frequent DESCRIPTOR words swapped at the same house number (nsc <= 0.1, i.e.
not over-represented on the R side: club, ecole, amicale...). Descriptor swaps sit at the same number 40% of the time
vs ~80% for noise words and ~2% for sister-company words -> the same-number subset should be mostly noise.
--pattern dualappend: a DUAL-USE word (0.15 <= role_swap < 0.5, like US "partners") APPENDED (nothing missing) at the
same house number: 97.2% true in train (3,607 pairs); sister companies sit at another number. France: developpement/
groupe/france appended at the same address get p ~0.15.
Stages: predict (writes <run>/output, validated)
"""
import os
import sys
import json
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, write_outputs, validate_outputs  # noqa: E402

SEEN = {"US", "India"}


def assigned(pred):
    return pred.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")


def role_pairs(a, s1, r):
    """Assigned pairs whose names differ by one extra R token (+ <= 1 missing S1 token), with role statistics."""
    c = s1["country"].to_numpy()
    si, ri = a["s1"].to_numpy(), a["r"].to_numpy()
    A = s1["name_core"].gather(si).str.split(" ").list.unique()
    B = r["name_core"].gather(ri).str.split(" ").list.unique()
    h1 = s1["addr_nums"].gather(si).str.split(" ").list.first()
    h2 = r["addr_nums"].gather(ri).str.split(" ").list.first()
    a = a.with_columns(B.list.set_difference(A).alias("extra"), A.list.set_difference(B).alias("miss"),
                       pl.Series("country", c[si]), ((h1 == h2) & (h1 != "")).alias("hn_eq"))
    d = a.filter((pl.col("extra").list.len() == 1) & (pl.col("miss").list.len() <= 1)).with_columns(
        pl.col("extra").list.first().alias("w"), (pl.col("miss").list.len() == 1).alias("swap"))
    role = d.group_by(["country", "w"]).agg(pl.len().alias("wn"), pl.col("swap").mean().alias("role_swap"))
    ns = []
    for cc in np.unique(c):
        ta = s1.filter(pl.col("country") == cc)["name_core"].str.split(" ").list.unique().explode().value_counts()
        rc = r.filter(pl.col("country") == cc)
        tb = rc["name_core"].str.split(" ").list.unique().explode().value_counts()
        n1 = int((c == cc).sum())
        t = tb.rename({tb.columns[0]: "w", "count": "nr"}).join(
            ta.rename({ta.columns[0]: "w", "count": "ns1"}), on="w", how="left").fill_null(0)
        ns.append(t.with_columns(pl.lit(cc).alias("country"),
                                 ((pl.col("nr") + 1) / (pl.col("ns1") + 1)).log().sub(np.log(rc.height / n1)).alias("nsc"))
                  .select("country", "w", "nsc"))
    return d.join(role, on=["country", "w"]).join(pl.concat(ns), on=["country", "w"], how="left")


def stage_predict(args, log):
    base = os.path.join(RUNS, args.base)
    tp = pl.read_parquet(os.path.join(base, "test_pred.parquet")).select("s1", "r", "p")
    s1 = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["eid", "country", "name_core", "addr_nums"])
    r = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_r.parquet"), columns=["eid", "country", "name_core", "addr_nums"])
    country = s1["country"].to_numpy()
    with open(os.path.join(RUNS, args.metrics_run or args.base, "metrics.json")) as fh:
        thr = args.thr if args.thr is not None else json.load(fh)["combined"]["best_thr"]
    a = assigned(tp)
    d = role_pairs(a, s1, r)
    words = set(args.words.split(",")) if args.words else None
    if args.pattern == "noise":
        rule = d.filter(pl.col("swap") & pl.col("hn_eq") & (pl.col("role_swap") >= args.min_role) & (pl.col("wn") >= 200)
                        & (pl.col("nsc") > 0.1) & ~pl.col("country").is_in(list(SEEN)))
    elif args.pattern == "desc":
        rule = d.filter(pl.col("swap") & pl.col("hn_eq") & (pl.col("wn") >= 200) & (pl.col("nsc") <= 0.1)
                        & ~pl.col("country").is_in(list(SEEN)))
    else:
        rule = d.filter(~pl.col("swap") & pl.col("hn_eq") & (pl.col("role_swap") >= 0.15) & (pl.col("role_swap") < 0.5)
                        & (pl.col("wn") >= 200) & (pl.col("nsc") > 0.1) & ~pl.col("country").is_in(list(SEEN)))
    if words is not None:
        rule = rule.filter(pl.col("w").is_in(list(words)))
    flip = rule.filter((pl.col("p") <= args.p_set) & (pl.col("p") > args.p_floor))
    log("rule words (unseen countries):", rule.group_by(["country", "w"]).agg(
        pl.len().alias("n"), (pl.col("p") <= args.p_set).sum().alias("below"), pl.col("role_swap").first().round(2),
        pl.col("nsc").first().round(1)).sort("n", descending=True).rows())
    # S1 whose flipped pair would be their only accepted match (highest-variance case: one-match S1 vs singleton)
    t_s1 = np.where(np.isin(country, list(SEEN)), thr, args.unseen_thr)
    acc = a.with_columns(pl.Series("t", t_s1[a["s1"].to_numpy()])).filter(pl.col("p") > pl.col("t"))
    n_acc = np.bincount(acc["s1"].to_numpy(), minlength=len(country))
    fs = flip["s1"].to_numpy()
    log(f"flips {flip.height} in {len(np.unique(fs))} S1 ({len(np.unique(fs)) / (country == 'France').sum():.3f} of France); "
        f"S1 with no other accepted match: {int((n_acc[np.unique(fs)] == 0).sum())}")
    res = tp.join(flip.select("s1", "r", pl.lit(args.p_set, pl.Float32).alias("pn")), on=["s1", "r"], how="left") \
            .with_columns(pl.max_horizontal("p", pl.col("pn").fill_null(0)).alias("p")).drop("pn")
    out_run = os.path.join(RUNS, args.run)
    res.write_parquet(os.path.join(out_run, "test_pred.parquet"))
    a2 = assigned(res)
    a2 = a2.with_columns(pl.Series("t", t_s1[a2["s1"].to_numpy()])).filter(pl.col("p") > pl.col("t"))
    out_dir = os.path.join(out_run, "output")
    write_outputs(out_dir, s1["eid"].to_numpy(), a2["s1"].to_numpy(), a2["r"].to_numpy(),
                  res["s1"].to_numpy(), res["r"].to_numpy(), r["eid"].to_numpy())
    n = np.bincount(a2["s1"].to_numpy(), minlength=len(country))
    for cc in np.unique(country):
        m = country == cc
        log(f"test {cc}: mean matches {n[m].mean():.3f} empty {np.mean(n[m] == 0):.4f}")
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["predict"])
    ap.add_argument("--run", default="exp11")
    ap.add_argument("--base", default="exp10")
    ap.add_argument("--metrics_run", default="", help="run whose metrics.json gives the seen-country threshold (default --base)")
    ap.add_argument("--words", default="", help="optional comma-separated restriction of the rule words (probes)")
    ap.add_argument("--min_role", type=float, default=0.1)
    ap.add_argument("--pattern", default="noise", choices=["noise", "desc", "dualappend"])
    ap.add_argument("--p_floor", type=float, default=0.01)
    ap.add_argument("--p_set", type=float, default=0.95)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    stage_predict(args, log)


if __name__ == "__main__":
    main()
