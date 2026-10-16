"""exp12_roletrans: role-based translation of unseen-country name words into training vocabulary (test side only).

Why: exp11 showed (LB +0.0014) that France loses true matches because its NOISE words (fils, associes, and the
dual-use groupe/developpement/france) never occur in training. exp11 patched one pattern with a rule (swap + equal
house number). exp12 instead translates each unseen-country word, by its inferred role, into the training word with
the same role, then recomputes the whole test side (blocking, features, stages 1/2) with the unchanged models, so the
model uses all its other evidence on these pairs.
Role of a word t in a split/country (no labels, see exp11_rolerule.role_pairs), for words with >= 200 single-extra-
word occurrences and R over-representation nsc > 0.1:
  role_swap >= 0.5          -> "services"  (pure noise word; US/India: center, services, service ~0.75-0.84)
  0.15 <= role_swap < 0.5   -> "partners"  (dual-use: swap = noise, append = sister company; US partners 0.18)
  role_swap < 0.05, hn_eq <= 0.3 -> "holdings"  (family / sister-company word, at another house number; US/India
                               holdings, group: role_swap ~0.01, hn_eq 0.01-0.21. "5as" (SAS typo) is appended at the
                               SAME address and stays unchanged)
  otherwise unchanged. Only countries absent from training are translated (seen countries keep their vocabulary).
Tokens are replaced in name_core, name_clean and name_alias of both S1 and S2/S3 records.

Stages:
  prep    (ER_WORK_DIR = base world, e.g. work_v3): role map from --base test predictions; writes --out_world with the
          translated test prep, train-side prep/features hard-linked, blocker models and --model_run model files copied
  combine (ER_WORK_DIR = base world): unseen-country predictions from --out_world/runs/<model_run>, seen countries
          from --base; decode (seen thr from --base metrics, unseen --unseen_thr); optional exp11 rule on top (--rule)
Test side in between: scripts/run_exp12.sh (mode all, work_v4), scripts/run_exp12s.sh (mode noise, work_v5)
"""
import os
import sys
import json
import shutil
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, write_outputs, validate_outputs  # noqa: E402
import exp11_rolerule as R11  # noqa: E402

SEEN = {"US", "India"}
TARGET = [(0.5, 1.01, "services"), (0.15, 0.5, "partners"), (-1.0, 0.05, "holdings")]
NAME_COLS = ["name_core", "name_clean", "name_alias"]
NOISE_TARGETS = ["center", "service", "services"]   # pure noise words of both training countries (role_swap 0.73-0.84)


def role_map(tp, s1, r, log, mode="all"):
    """mode all  : every role, one target token per role (exp12; merged distinct family words into one token ->
                   sister-company distractors became identical records and were accepted, see EXPERIMENTS.md)
       mode noise: pure noise words only (role_swap >= 0.5), each onto a DISTINCT training noise word; words that are
                   already training noise words stay; family and dual-use words untouched (exp12s)"""
    d = R11.role_pairs(R11.assigned(tp), s1, r)
    words = d.group_by(["country", "w"]).agg(pl.col("wn").first(), pl.col("role_swap").first(), pl.col("nsc").first(),
                                             pl.col("hn_eq").mean().alias("hn_eq")) \
             .filter(~pl.col("country").is_in(list(SEEN)) & (pl.col("wn") >= 200) & (pl.col("nsc") > 0.1))
    m, free = {}, {}
    for c, w, wn, rs, nsc, hn in words.sort("wn", descending=True).rows():
        tgt = next((t for lo, hi, t in TARGET if lo <= rs < hi), None)
        if tgt == "holdings" and hn > 0.3:   # family words sit at another address; "5as" (SAS typo) does not
            tgt = None
        if mode == "noise":
            if tgt != "services" or w in NOISE_TARGETS:
                tgt = None
            else:
                pool = free.setdefault(c, [t for t in NOISE_TARGETS if t not in set(words.filter(pl.col("country") == c)["w"])])
                tgt = pool.pop(0) if pool else None
        log(f"  {c} {w:16s} n {wn:6d} role_swap {rs:.2f} hn_eq {hn:.2f} nsc {nsc:+.1f} -> {tgt}")
        if tgt is not None and tgt != w:
            m.setdefault(c, {})[w] = tgt
    return m


def translate(df, m):
    """Replace tokens of the name columns for rows of translated countries."""
    out = []
    for c, sub in df.with_row_index("_i").group_by("country"):
        c = c[0]
        if c in m:
            mp = m[c]
            sub = sub.with_columns([
                pl.col(col).str.split(" ").list.eval(pl.element().replace(mp)).list.join(" ").alias(col)
                for col in NAME_COLS if col in sub.columns])
        out.append(sub)
    return pl.concat(out).sort("_i").drop("_i")


def link_tree(src, dst):
    """Hard-link every file of src into dst (same drive; no extra space)."""
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            t = os.path.join(dst, rel, f)
            if not os.path.exists(t):
                os.link(os.path.join(root, f), t)


def stage_prep(args, log):
    base = os.path.join(RUNS, args.base)
    tp = pl.read_parquet(os.path.join(base, "test_pred.parquet")).select("s1", "r", "p")
    P = os.path.join(CACHE, "prep_v1")
    s1 = pl.read_parquet(os.path.join(P, "test_s1.parquet"))
    r = pl.read_parquet(os.path.join(P, "test_r.parquet"))
    log("role map (unseen countries):")
    m = role_map(tp, s1, r, log, args.mode)
    log("translation:", m)
    ow = args.out_world
    oc = os.path.join(ow, "data", "cache")
    os.makedirs(os.path.join(oc, "prep_v1"), exist_ok=True)
    with open(os.path.join(ow, "role_map.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    for f in os.listdir(P):
        if not f.startswith("test_"):
            t = os.path.join(oc, "prep_v1", f)
            if not os.path.exists(t):
                os.link(os.path.join(P, f), t)
    for tag, df in [("s1", s1), ("r", r)]:
        new = translate(df, m)
        changed = sum((new[c] != df[c]).sum() for c in NAME_COLS if c in df.columns)
        log(f"test_{tag}: {changed} name cells translated")
        new.write_parquet(os.path.join(oc, "prep_v1", f"test_{tag}.parquet"))
    link_tree(os.path.join(oc, "prep_v1"), os.path.join(oc, "prep_v2"))
    # train-side layers (unchanged) and models
    for d in ["feat_v3", os.path.join("feat_v7", "train"), os.path.join("feat_v9", "train")]:
        link_tree(os.path.join(CACHE, d), os.path.join(oc, d))
    os.makedirs(os.path.join(oc, "emb_v1"), exist_ok=True)
    for f in ["model_fold0.pt", "model_fold1.pt"]:
        shutil.copy2(os.path.join(CACHE, "emb_v1", f), os.path.join(oc, "emb_v1", f))
    rd = os.path.join(ow, "runs", args.model_run)
    os.makedirs(rd, exist_ok=True)
    for f in os.listdir(os.path.join(RUNS, args.model_run)):
        if f.endswith(".json") or f in ("s1_oof.parquet", "oof.parquet"):
            t = os.path.join(rd, f)
            if not os.path.exists(t):
                os.link(os.path.join(RUNS, args.model_run, f), t)
    log(f"world ready: {ow}")


def stage_combine(args, log):
    base = os.path.join(RUNS, args.base)
    s1 = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["eid", "country", "name_core", "addr_nums"])
    r = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_r.parquet"), columns=["eid", "country", "name_core", "addr_nums"])
    country = s1["country"].to_numpy()
    seen_s1 = pl.Series(np.isin(country, list(SEEN)))
    tp = pl.read_parquet(os.path.join(base, "test_pred.parquet")).select("s1", "r", "p")
    tp = tp.filter(seen_s1.gather(tp["s1"].to_numpy()))
    un = pl.read_parquet(os.path.join(args.out_world, "runs", args.model_run, "test_pred.parquet")).select("s1", "r", "p")
    un = un.filter(~seen_s1.gather(un["s1"].to_numpy()))
    res = pl.concat([tp, un.with_columns(pl.col("p").cast(tp["p"].dtype))])
    if args.rule:
        # exp11 rule on top, with role statistics of the ORIGINAL (untranslated) names
        d = R11.role_pairs(R11.assigned(res), s1, r)
        rule = d.filter(pl.col("swap") & pl.col("hn_eq") & (pl.col("role_swap") >= 0.1) & (pl.col("wn") >= 200)
                        & (pl.col("nsc") > 0.1) & ~pl.col("country").is_in(list(SEEN)))
        flip = rule.filter((pl.col("p") <= 0.95) & (pl.col("p") > 0.01))
        log(f"rule on top: {rule.height} rule pairs, {flip.height} flips")
        res = res.join(flip.select("s1", "r", pl.lit(0.95, pl.Float32).alias("pn")), on=["s1", "r"], how="left") \
                 .with_columns(pl.max_horizontal("p", pl.col("pn").fill_null(0)).alias("p")).drop("pn")
    out_run = os.path.join(RUNS, args.run)
    res.write_parquet(os.path.join(out_run, "test_pred.parquet"))
    with open(os.path.join(base, "metrics.json")) as fh:
        thr = json.load(fh)["combined"]["best_thr"]
    t_s1 = np.where(np.isin(country, list(SEEN)), thr, args.unseen_thr)
    a = R11.assigned(res)
    a = a.with_columns(pl.Series("t", t_s1[a["s1"].to_numpy()])).filter(pl.col("p") > pl.col("t"))
    out_dir = os.path.join(out_run, "output")
    write_outputs(out_dir, s1["eid"].to_numpy(), a["s1"].to_numpy(), a["r"].to_numpy(),
                  res["s1"].to_numpy(), res["r"].to_numpy(), r["eid"].to_numpy())
    n = np.bincount(a["s1"].to_numpy(), minlength=len(country))
    for cc in np.unique(country):
        mm = country == cc
        log(f"test {cc}: mean matches {n[mm].mean():.3f} empty {np.mean(n[mm] == 0):.4f}")
    # France uncertainty after translation (vs exp10: 10% of accepted pairs in (0.3, 0.98])
    au = R11.assigned(res)
    fr = au.filter(~seen_s1.gather(au["s1"].to_numpy()))
    log(f"unseen: share of pairs p>0.3 that are in (0.3,0.98]: "
        f"{((fr['p'] > 0.3) & (fr['p'] <= 0.98)).sum() / max((fr['p'] > 0.3).sum(), 1):.4f}")
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "combine"])
    ap.add_argument("--run", default="exp12")
    ap.add_argument("--base", default="exp10")
    ap.add_argument("--model_run", default="exp09")
    ap.add_argument("--out_world", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_v4"))
    ap.add_argument("--rule", action="store_true")
    ap.add_argument("--mode", default="all", choices=["all", "noise"])
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    {"prep": stage_prep, "combine": stage_combine}[args.stage](args, log)


if __name__ == "__main__":
    main()
