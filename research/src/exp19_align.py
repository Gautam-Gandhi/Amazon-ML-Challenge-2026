"""exp19_align: France (unseen-country) acceptance alignment for structural pair types that are near-certain
matches in the training countries (H5, EXPERIMENTS.md 09-27).

Why: 96K French pairs are uncertain although no other candidate ties with the chosen S1 (France 0.345 no-tie
uncertain pairs per S1 vs US 0.176 / India 0.052). For the same structural type, France accepts far less than the
in-country model does (brand/acronym at the same house number: France 86.6% vs US 98.8% / India 97.5%; identical name
at a far house number, same legal form: 75-81% vs 99%). The model's confidence does not transfer (small embedding
margins in France, unfamiliar vocabulary). France-specific distractor types are excluded: descriptor swaps (LB:
mostly false, exp14rcd) and legal-form siblings (different legal form / nearby number, sibling pattern in US test).
Types (assigned pair = best S1 of its R; NO TIE = no other candidate S1 of the R has the same name_core or the same
addr_clean as the chosen S1):
  brand    : R name shares no word with the S1 name, R has an address, same first house number and the same street
             words (non-numeric address tokens, Jaccard >= 0.5)
  id_far   : identical name_core, both house numbers present, |difference| > 12, same legal form or one missing
  id_near  : identical name_core, both present, 0 < |difference| <= 12, SAME legal form (siblings differ in legal form)
  id_noaddr: identical name_core, R has no address, name unique among the country's S1 (no same-name S1 at all)
  id_street: identical name_core, same first house number, different street words (Jaccard < 0.5): street noise
  typo_swap: one R word replaces one S1 word within edit distance 2 (typo), same first house number
Stage eval   : training-country true rate of each type on the dense OOF (labels) -> the rule is used only for types
               with true rate >= --min_true
Stage predict: unseen-country pairs of those types with p in (--p_floor, unseen_thr] get p := --p_set
"""
import os
import sys
import json
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, save_json, write_outputs, validate_outputs  # noqa: E402

SEEN = {"US", "India"}
TYPES = ["brand", "id_far", "id_near", "id_noaddr", "id_street", "typo_swap"]


def typed_pairs(pred, split):
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    cols = ["country", "name_core", "addr_clean", "addr_nums", "name_sfx"]
    s1 = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=cols)
    r = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=cols[1:])
    name_cnt = s1.group_by(["country", "name_core"]).len("n_name")
    t = pred.select(pl.col("s1").cast(pl.Int64), pl.col("r").cast(pl.Int64), "p")
    sid = t["s1"].to_numpy()
    t = t.with_columns(s1["name_core"].gather(sid).alias("n1"), s1["addr_clean"].gather(sid).alias("a1"))
    best = t.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    oth = t.join(best.select("r", pl.col("s1").alias("sb"), pl.col("n1").alias("nb"), pl.col("a1").alias("ab")), on="r") \
           .filter(pl.col("s1") != pl.col("sb"))
    tie = oth.group_by("r").agg(((pl.col("n1") == pl.col("nb")) | (pl.col("a1") == pl.col("ab"))).any().alias("tie"))
    b = best.join(tie, on="r", how="left").with_columns(pl.col("tie").fill_null(False))
    si, ri = b["s1"].to_numpy(), b["r"].to_numpy()
    A = s1["name_core"].gather(si).str.split(" ").list.unique()
    B = r["name_core"].gather(ri).str.split(" ").list.unique()
    h1 = s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null("")
    h2 = r["addr_nums"].gather(ri).str.split(" ").list.first().fill_null("")
    w1 = s1["addr_clean"].gather(si).str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"\d")))
    w2 = r["addr_clean"].gather(ri).str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"\d")))
    f1 = s1["name_sfx"].gather(si).fill_null("")
    f2 = r["name_sfx"].gather(ri).fill_null("")
    b = b.with_columns(pl.Series("country", s1["country"].to_numpy()[si]),
                       B.list.set_intersection(A).list.len().alias("common"),
                       (B.list.set_difference(A).list.len() + A.list.set_difference(B).list.len()).alias("ndiff"),
                       B.list.set_difference(A).list.len().alias("ne"), A.list.set_difference(B).list.len().alias("nm"),
                       B.list.set_difference(A).list.first().alias("e1"), A.list.set_difference(B).list.first().alias("m1"),
                       h1.alias("h1"), h2.alias("h2"), (r["addr_clean"].gather(ri) == "").alias("r_noaddr"),
                       (w1.list.set_intersection(w2).list.len() / w1.list.set_union(w2).list.len().clip(1, None)).alias("street_j"),
                       f1.alias("f1"), f2.alias("f2"))
    b = b.join(name_cnt.rename({"name_core": "n1"}), on=["country", "n1"], how="left")
    from rapidfuzz.distance import Levenshtein
    b = b.with_columns(pl.struct(["e1", "m1"]).map_elements(
        lambda d: Levenshtein.distance(d["e1"], d["m1"]) if d["e1"] and d["m1"] else 99, return_dtype=pl.Int64).alias("ed"))
    dh = (pl.col("h1").str.extract(r"(\d+)").cast(pl.Int64, strict=False)
          - pl.col("h2").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).abs()
    sfx_ok = (pl.col("f1") == "") | (pl.col("f2") == "") | (pl.col("f1") == pl.col("f2"))
    both_hn = (pl.col("h1") != "") & (pl.col("h2") != "")
    kind = (pl.when(pl.col("tie")).then(pl.lit(None))
            .when((pl.col("common") == 0) & ~pl.col("r_noaddr") & (pl.col("h1") == pl.col("h2")) & (pl.col("h1") != "")
                  & (pl.col("street_j") >= 0.5)).then(pl.lit("brand"))
            .when((pl.col("ndiff") == 0) & both_hn & (dh > 12) & sfx_ok).then(pl.lit("id_far"))
            .when((pl.col("ndiff") == 0) & both_hn & (dh > 0) & (dh <= 12) & (pl.col("f1") == pl.col("f2"))
                  & (pl.col("f1") != "")).then(pl.lit("id_near"))
            .when((pl.col("ndiff") == 0) & pl.col("r_noaddr") & (pl.col("n_name") == 1)).then(pl.lit("id_noaddr"))
            .when((pl.col("ndiff") == 0) & ~pl.col("r_noaddr") & (pl.col("h1") == pl.col("h2")) & (pl.col("h1") != "")
                  & (pl.col("street_j") < 0.5)).then(pl.lit("id_street"))
            .when((pl.col("ne") == 1) & (pl.col("nm") == 1) & (pl.col("ed") <= 2) & (pl.col("h1") == pl.col("h2"))
                  & (pl.col("h1") != "")).then(pl.lit("typo_swap"))
            .otherwise(pl.lit(None)))
    return b.with_columns(kind.alias("kind")).select("s1", "r", "p", "country", "kind")


def stage_eval(args, log):
    oof = pl.read_parquet(os.path.join(RUNS, args.base_oof, "oof_combined.parquet"))
    gt = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_gt.parquet")).select(
        pl.col("s1_idx").cast(pl.Int64).alias("s1"), pl.col("r_idx").cast(pl.Int64).alias("r"), pl.lit(1).alias("y"))
    b = typed_pairs(oof, "train").join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
    stats = b.filter(pl.col("kind").is_not_null() & (pl.col("p") > args.p_floor)).group_by("kind").agg(
        pl.len().alias("n"), pl.col("y").mean().alias("true_rate"), (pl.col("p") > 0.7).mean().alias("accepted"))
    log("training countries (dense OOF, labels):", stats.sort("kind").rows())
    use = [k for k, tr in zip(stats["kind"], stats["true_rate"]) if tr >= args.min_true]
    log(f"types used (true rate >= {args.min_true}): {use}")
    save_json({"types": use, "stats": stats.rows()}, os.path.join(RUNS, args.run, "align_types.json"))


def stage_predict(args, log):
    if args.types is not None:          # explicit list ("none" = no alignment: final decoding only)
        use = [] if args.types == "none" else args.types.split(",")
    else:
        with open(os.path.join(RUNS, args.run, "align_types.json")) as fh:
            use = json.load(fh)["types"]
    base = os.path.join(RUNS, args.base)
    tp = pl.read_parquet(os.path.join(base, "test_pred.parquet")).select("s1", "r", "p")
    b = typed_pairs(tp, "test")
    fr = b.filter(~pl.col("country").is_in(list(SEEN)) & pl.col("kind").is_in(use))
    log("unseen-country typed pairs:", fr.group_by("kind").agg(pl.len().alias("n"), pl.col("p").mean().round(3).alias("mean_p"),
        (pl.col("p") > args.unseen_thr).mean().round(3).alias("accepted")).sort("kind").rows())
    flip = fr.filter((pl.col("p") > args.p_floor) & (pl.col("p") <= args.unseen_thr))
    log(f"flips: {flip.height:,} ({flip.group_by('kind').len().sort('kind').rows()})")
    res = tp.join(flip.select("s1", "r", pl.lit(args.p_set, pl.Float32).alias("pn")), on=["s1", "r"], how="left") \
            .with_columns(pl.max_horizontal("p", pl.col("pn").fill_null(0)).alias("p")).drop("pn")
    out_run = os.path.join(RUNS, args.run)
    res.write_parquet(os.path.join(out_run, "test_pred.parquet"))
    s1 = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["eid", "country"])
    country = s1["country"].to_numpy()
    with open(os.path.join(RUNS, args.metrics_run, "metrics.json")) as fh:
        thr = json.load(fh)["combined"]["best_thr"]
    seen = np.isin(country, list(SEEN))
    t = np.where(seen, thr, args.unseen_thr)
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.with_columns(pl.Series("t", t[a["s1"].to_numpy()]), pl.Series("seen", seen[a["s1"].to_numpy()]))
    acc = a.filter(pl.col("p") > pl.col("t"))
    if args.t_empty is not None:   # training countries: an S1 without any accept takes its best pair if p > t_empty
        extra = a.filter((pl.col("p") > args.t_empty) & (pl.col("p") <= pl.col("t")) & pl.col("seen")
                         & ~pl.col("s1").is_in(acc["s1"].unique().implode()))                  .sort("p", descending=True).unique(subset=["s1"], keep="first")
        log(f"two-threshold decoding (training countries, t_empty {args.t_empty}): +{extra.height:,} pairs")
        acc = pl.concat([acc, extra.select(acc.columns)])
    a = acc
    r_ids = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(out_run, "output")
    write_outputs(out_dir, s1["eid"].to_numpy(), a["s1"].to_numpy(), a["r"].to_numpy(),
                  res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    n = np.bincount(a["s1"].to_numpy(), minlength=len(country))
    for c in np.unique(country):
        m = country == c
        log(f"test {c}: mean matches {n[m].mean():.3f} empty {np.mean(n[m] == 0):.4f}")
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eval", "predict"])
    ap.add_argument("--run", default="exp19")
    ap.add_argument("--base", default="exp17cr")
    ap.add_argument("--base_oof", default="exp17c", help="run with oof_combined.parquet (dense OOF) for type stats")
    ap.add_argument("--metrics_run", default="exp17c")
    ap.add_argument("--min_true", type=float, default=0.9)
    ap.add_argument("--p_floor", type=float, default=0.05)
    ap.add_argument("--p_set", type=float, default=0.95)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    ap.add_argument("--t_empty", type=float, default=None, help="training countries: rescue threshold for S1 with no accept")
    ap.add_argument("--types", default=None, help="override the aligned types: comma list, or 'none' (decode only)")
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    {"eval": stage_eval, "predict": stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
