"""exp31_sibprior: prior-shift correction for test-only legal-form sibling distractors (training countries).

Why (09-27 afternoon, label-free count matching test vs dense OOF): the generator produces TRUE records with the same
noise mix in train and test (accepted pairs per 1K S1 agree within +-1..3 in every fine pair type), but the test adds
far more "legal-form sibling" distractors: same name_core, the S1 has NO legal form, the record adds one, and the first
house number differs. US candidates of this group per 1K S1: dense OOF 29.6 (9-12% true) vs test 159 at |dnum|<=12.
The model (trained in the dense world, where these siblings are rare) accepts many of them in the test:
accepted per 1K US S1 at p>0.7: <=2: OOF TP 1.74 vs test 6.30; 3-12: 0.63 vs 4.27; 13-100: 2.58 vs 3.72;
>100: 21.8 vs 24.0.  The excess sits at p <= 0.98 (near) / p <= 0.95 (far); above that test == OOF.
India's test shows no such excess (the model already rejects its siblings); France has no OOF reference.
Fix: for training-country pairs of this group (--countries, default US), demote the pair (p := p * --demote) when
p <= --t_near (|dnum| <= 12) or p <= --t_far (|dnum| > 12). Everything else is unchanged.
Stages:
  eval    : the same demotion on the dense OOF (labels) -> the true-pair cost of the rule (F0.5 before/after)
  predict : --base test_pred -> demote -> the exp19_align decode (thr from --metrics_run, unseen 0.9, t_empty) -> outputs
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


def sibling_mask(pred, split, countries):
    """True for pairs of the sibling group: identical name_core, S1 without legal form, R with one, both first house
    numbers present and different. Returns (mask, |first-number difference|)."""
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    cols = ["country", "name_core", "addr_nums", "name_sfx"]
    s1 = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=cols)
    r = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=cols[1:])
    si, ri = pred["s1"].to_numpy(), pred["r"].to_numpy()
    d = pl.DataFrame({
        "ctry": s1["country"].to_numpy()[si],
        "same_name": (s1["name_core"].gather(si) == r["name_core"].gather(ri)).to_numpy(),
        "f1": s1["name_sfx"].gather(si).fill_null(""), "f2": r["name_sfx"].gather(ri).fill_null(""),
        "h1": s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null(""),
        "h2": r["addr_nums"].gather(ri).str.split(" ").list.first().fill_null("")})
    d = d.with_columns((pl.col("h1").str.extract(r"(\d+)").cast(pl.Int64, strict=False)
                        - pl.col("h2").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).abs().alias("dnum"))
    m = (pl.col("ctry").is_in(list(countries)) & pl.col("same_name") & (pl.col("f1") == "") & (pl.col("f2") != "")
         & (pl.col("h1") != "") & (pl.col("h2") != "") & (pl.col("h1") != pl.col("h2")))
    d = d.with_columns(m.alias("m"))
    return d["m"].to_numpy(), d["dnum"].fill_null(10**9).to_numpy()


def demote(pred, split, args, log):
    m, dnum = sibling_mask(pred, split, args.countries.split(","))
    p = pred["p"].to_numpy().astype(np.float32)
    thr = np.where(dnum <= 12, args.t_near, args.t_far)
    hit = m & (p <= thr)
    log(f"[{split}] sibling-group pairs {m.sum():,}; demoted (p <= {args.t_near} near / {args.t_far} far): {hit.sum():,} "
        f"(of them p > 0.7: {(hit & (p > 0.7)).sum():,})")
    p2 = np.where(hit, p * args.demote, p).astype(np.float32)
    return pred.with_columns(pl.Series("p", p2)), hit


def stage_eval(args, log):
    import exp03_dense as M3
    keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
    oof = pl.read_parquet(os.path.join(RUNS, args.base_oof, "oof_combined.parquet"))
    m_ref = M3.decode_eval(oof, keep, log, f"{args.base_oof} (reference)", thrs=(0.7,))
    new, hit = demote(oof, "train", args, log)
    gt = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_gt.parquet")).select(
        pl.col("s1_idx").cast(pl.Int32).alias("s1"), pl.col("r_idx").cast(pl.Int32).alias("r"), pl.lit(1).alias("y"))
    h = oof.filter(pl.Series(hit)).join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
    log(f"demoted OOF pairs: {h.height:,}, true {h['y'].sum():,}; with p > 0.7: {h.filter(pl.col('p') > 0.7).height:,} "
        f"(true {h.filter(pl.col('p') > 0.7)['y'].sum():,})")
    m = M3.decode_eval(new, keep, log, "sibling demotion", thrs=(0.7,))
    save_json({"reference": m_ref, "demoted": m, "args": vars(args)}, os.path.join(RUNS, args.run, "metrics_eval.json"))


def stage_predict(args, log):
    tp = pl.read_parquet(os.path.join(RUNS, args.base, "test_pred.parquet")).select("s1", "r", "p")
    res, hit = demote(tp, "test", args, log)
    decode_write(res, args, log)


def decode_write(res, args, log):
    """Write test_pred + outputs of run args.run: the exp19_align decode (seen thr from args.metrics_run, unseen
    args.unseen_thr, two-threshold rescue args.t_empty for training countries), then validate."""
    out_run = os.path.join(RUNS, args.run)
    res.write_parquet(os.path.join(out_run, "test_pred.parquet"))
    s1 = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["eid", "country"])
    country = s1["country"].to_numpy()
    with open(os.path.join(RUNS, args.metrics_run, "metrics.json")) as fh:
        thr = json.load(fh)["combined"]["best_thr"]
    seen = np.isin(country, list(SEEN))
    t = np.where(seen, thr, args.unseen_thr)
    # decode exactly as exp19_align.stage_predict
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.with_columns(pl.Series("t", t[a["s1"].to_numpy()]), pl.Series("seen", seen[a["s1"].to_numpy()]))
    acc = a.filter(pl.col("p") > pl.col("t"))
    if args.t_empty is not None:
        extra = a.filter((pl.col("p") > args.t_empty) & (pl.col("p") <= pl.col("t")) & pl.col("seen")
                         & ~pl.col("s1").is_in(acc["s1"].unique().implode())).sort("p", descending=True).unique(subset=["s1"], keep="first")
        log(f"two-threshold decoding (training countries, t_empty {args.t_empty}): +{extra.height:,} pairs")
        acc = pl.concat([acc, extra.select(acc.columns)])
    r_ids = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(out_run, "output")
    write_outputs(out_dir, s1["eid"].to_numpy(), acc["s1"].to_numpy(), acc["r"].to_numpy(),
                  res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    n = np.bincount(acc["s1"].to_numpy(), minlength=len(country))
    for c in np.unique(country):
        mm = country == c
        log(f"test {c}: mean matches {n[mm].mean():.3f} empty {np.mean(n[mm] == 0):.4f}")
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eval", "predict"])
    ap.add_argument("--run", default="exp31")
    ap.add_argument("--base", default="exp29gf", help="run whose test_pred.parquet is decoded (after the France rule)")
    ap.add_argument("--base_oof", default="exp29g", help="run with the dense OOF oof_combined.parquet")
    ap.add_argument("--metrics_run", default="exp29g")
    ap.add_argument("--countries", default="US")
    ap.add_argument("--t_near", type=float, default=0.98)
    ap.add_argument("--t_far", type=float, default=0.95)
    ap.add_argument("--demote", type=float, default=0.5)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    ap.add_argument("--t_empty", type=float, default=0.6)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    {"eval": stage_eval, "predict": stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
