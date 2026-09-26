"""exp13_rolefeat: word-ROLE features (per split and country, no labels) as a 4th feature layer on top of exp09.

Why (exp11/exp12, 09-26): the generator adds words in roles with a per-country vocabulary. NOISE words replace a
descriptor at the same address ("Obrien Lion LLC" -> "Obrien LLC Services"); FAMILY words are appended to the full name
of a sister company at another address ("Falcon Corp" -> "Falcon Group"). The model only knows the US/India words, so
an unseen country (France, and India/US in the LOCO benchmark) loses ~0.02 F0.5: in-country 0.9896 vs France ~0.969,
LOCO ~0.970. The top teams (~0.99) must score ~0.985-0.99 on France. exp11 patched French words with rules (+0.0014 LB);
here the model gets the role information itself, so it can learn "noise-role word swapped at the same address -> match"
from US/India and apply it to any vocabulary.

Role table of a word t in (split, country), from NEAR-DUPLICATE candidate pairs (names differ by one extra R token t,
<= 1 missing S1 token) where the S1 is the R's top candidate by embedding (r_rank_j == 1; model-free, same for train/test):
  e_n (count), e_swap (share with one S1 token missing), e_hneq (share with equal first house number)
and symmetric statistics for a word as the single MISSING S1 token (m_n, m_swap, m_hneq); plus document frequencies
of t in S1 and R names of the country (f1, nsc = R over-representation).
Pair features (min/max over the pair's extra tokens E = R-S1 and missing tokens M = S1-R):
  rl_e_swap_min/max, rl_e_hneq_min/max, rl_e_lrate_min/max, rl_e_nsc_min/max, rl_e_f1_max,
  rl_m_swap_min, rl_m_hneq_min, rl_m_lrate_min, rl_m_f1_max
Stages: feats | train1 | feats2 | train2 | predict (model stages = exp05 code over 4 layers: base, exp07, exp09, exp13)
"""
import os
import sys
import glob
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger  # noqa: E402
import exp05_tokfeat as E5  # noqa: E402
import exp07_tokfeat2 as E7  # noqa: E402
import exp09_consensus as E9  # noqa: E402

FEAT13 = os.path.join(CACHE, "feat_v13")
S2DIR13 = os.path.join(CACHE, "feat_v13s2")
MIN_N = 5
FEATS = ["rl_e_swap_min", "rl_e_swap_max", "rl_e_hneq_min", "rl_e_hneq_max", "rl_e_lrate_min", "rl_e_lrate_max",
         "rl_e_nsc_min", "rl_e_nsc_max", "rl_e_f1_max", "rl_m_swap_min", "rl_m_hneq_min", "rl_m_lrate_min", "rl_m_f1_max"]


def tables(split):
    def prep(tag):
        cols = ["name_core", "addr_nums"] + (["country"] if tag == "s1" else [])
        d = pl.read_parquet(os.path.join(E5.PREP[split], f"{split}_{tag}.parquet"), columns=cols)
        return d.with_columns(pl.col("name_core").str.split(" ").list.unique().alias("tok"),
                              pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("hn1"))
    return prep("s1"), prep("r")


def pair_sets(k, s1, r):
    si, ri = k["s1"].to_numpy(), k["r"].to_numpy()
    A, B = s1["tok"].gather(si), r["tok"].gather(ri)
    h1, h2 = s1["hn1"].gather(si), r["hn1"].gather(ri)
    return k.with_columns(pl.Series("country", s1["country"].to_numpy()[si]),
                          B.list.set_difference(A).list.eval(pl.element().filter(pl.element() != "")).alias("E"),
                          A.list.set_difference(B).list.eval(pl.element().filter(pl.element() != "")).alias("M"),
                          ((h1 == h2) & (h1 != "")).alias("hn_eq"))


def role_tables(split, src, s1, r, log):
    parts = sorted(glob.glob(os.path.join(src, "part*.parquet")))
    acc_e, acc_m = [], []
    for p in parts:
        k = pair_sets(pl.read_parquet(p, columns=["s1", "r", "r_rank_j"]).filter(pl.col("r_rank_j") == 1), s1, r)
        ne, nm = pl.col("E").list.len(), pl.col("M").list.len()
        e = k.filter((ne == 1) & (nm <= 1)).select("country", pl.col("E").list.first().alias("t"), (nm == 1).alias("sw"), "hn_eq")
        m = k.filter((nm == 1) & (ne <= 1)).select("country", pl.col("M").list.first().alias("t"), (ne == 1).alias("sw"), "hn_eq")
        acc_e.append(e.group_by(["country", "t"]).agg(pl.len().alias("n"), pl.col("sw").sum().alias("ns"), pl.col("hn_eq").sum().alias("nh")))
        acc_m.append(m.group_by(["country", "t"]).agg(pl.len().alias("n"), pl.col("sw").sum().alias("ns"), pl.col("hn_eq").sum().alias("nh")))
    n1 = s1.group_by("country").len("N1")

    def fin(acc, pre):
        t = pl.concat(acc).group_by(["country", "t"]).agg(pl.col("n").sum(), pl.col("ns").sum(), pl.col("nh").sum()).join(n1, on="country")
        return t.select("country", "t",
                        pl.when(pl.col("n") >= MIN_N).then(pl.col("ns") / pl.col("n")).alias(f"{pre}_swap"),
                        pl.when(pl.col("n") >= MIN_N).then(pl.col("nh") / pl.col("n")).alias(f"{pre}_hneq"),
                        ((pl.col("n") + 1) / pl.col("N1")).log10().alias(f"{pre}_lrate"))
    te, tm = fin(acc_e, "e"), fin(acc_m, "m")
    # document frequencies per country
    rcountry = pl.read_parquet(os.path.join(E5.PREP[split], f"{split}_r.parquet"), columns=["country"])["country"]
    d1 = s1.select("country", pl.col("tok").alias("t")).explode("t").group_by(["country", "t"]).len("c1")
    dr = r.select(rcountry.alias("country"), pl.col("tok").alias("t")).explode("t").group_by(["country", "t"]).len("cr")
    nr = r.select(rcountry.alias("country")).group_by("country").len("NR")
    df = dr.join(d1, on=["country", "t"], how="full", coalesce=True).fill_null(0).join(n1, on="country").join(nr, on="country")
    df = df.select("country", "t", (pl.col("c1") / pl.col("N1")).alias("f1"),
                   (((pl.col("cr") + 1) / (pl.col("c1") + 1)).log() - (pl.col("NR") / pl.col("N1")).log()).alias("nsc"))
    for c in te["country"].unique().sort():
        x = te.filter(pl.col("country") == c).join(df, on=["country", "t"], how="left")
        top = x.sort("e_lrate", descending=True).head(12)
        log(f"  {split} {c}: {x.height} extra-word roles; most frequent: " + ", ".join(
            f"{t}(sw {sw if sw is None else round(sw, 2)}, hn {h if h is None else round(h, 2)})" for t, sw, h in top.select("t", "e_swap", "e_hneq").rows()))
    return te, tm, df


def stage_feats(args, log):
    for split, src in [("train", E5.FEAT3), ("test", E5.FEAT4T)]:
        out_dir = os.path.join(FEAT13, split)
        os.makedirs(out_dir, exist_ok=True)
        s1, r = tables(split)
        te, tm, df = role_tables(split, src, s1, r, log)
        te = te.join(df, on=["country", "t"], how="left")
        tm = tm.join(df.select("country", "t", "f1"), on=["country", "t"], how="left")
        for p in sorted(glob.glob(os.path.join(src, "part*.parquet"))):
            dst = os.path.join(out_dir, os.path.basename(p))
            if os.path.exists(dst):
                continue
            k = pair_sets(pl.read_parquet(p, columns=["s1", "r"]).with_row_index("row"), s1, r)
            fe = k.select("row", "country", pl.col("E").alias("t")).explode("t").drop_nulls("t").join(te, on=["country", "t"], how="left") \
                  .group_by("row").agg(pl.col("e_swap").min().alias("rl_e_swap_min"), pl.col("e_swap").max().alias("rl_e_swap_max"),
                                       pl.col("e_hneq").min().alias("rl_e_hneq_min"), pl.col("e_hneq").max().alias("rl_e_hneq_max"),
                                       pl.col("e_lrate").min().alias("rl_e_lrate_min"), pl.col("e_lrate").max().alias("rl_e_lrate_max"),
                                       pl.col("nsc").min().alias("rl_e_nsc_min"), pl.col("nsc").max().alias("rl_e_nsc_max"),
                                       pl.col("f1").max().alias("rl_e_f1_max"))
            fm = k.select("row", "country", pl.col("M").alias("t")).explode("t").drop_nulls("t").join(tm, on=["country", "t"], how="left") \
                  .group_by("row").agg(pl.col("m_swap").min().alias("rl_m_swap_min"), pl.col("m_hneq").min().alias("rl_m_hneq_min"),
                                       pl.col("m_lrate").min().alias("rl_m_lrate_min"), pl.col("f1").max().alias("rl_m_f1_max"))
            f = k.select("row").join(fe, on="row", how="left").join(fm, on="row", how="left").sort("row")
            assert f.height == k.height
            f.select([pl.col(c).cast(pl.Float32) for c in FEATS]).write_parquet(dst)
            log(f"{split} {os.path.basename(p)}: {k.height} rows")


def use_layers(args):
    """exp05 model stages over 4 layers: base (feat_v3 / test) + exp07 (feat_v7) + exp09 (feat_v9) + exp13 (feat_v13)."""
    E5.FEAT5 = E7.FEAT7
    E5.S2DIR = S2DIR13
    if args.test_feat:
        E5.FEAT4T = os.path.join(CACHE, args.test_feat)

    def iter_s1(split):
        src = E5.FEAT3 if split == "train" else E5.FEAT4T
        layers = [sorted(glob.glob(os.path.join(d, "part*.parquet"))) for d in
                  [src, os.path.join(E7.FEAT7, split), os.path.join(E9.FEAT9, split), os.path.join(FEAT13, split)]]
        assert len({len(x) for x in layers}) == 1, [len(x) for x in layers]
        return lambda: (pl.concat([pl.read_parquet(f) for f in fs], how="horizontal") for fs in zip(*layers))
    E5.iter_s1 = iter_s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "train1", "feats2", "train2", "predict"])
    ap.add_argument("--run", default="exp13")
    ap.add_argument("--test_feat", default="")
    ap.add_argument("--neg_frac", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    from er_common import set_determinism
    set_determinism(0)
    use_layers(args)
    if args.stage == "feats":
        stage_feats(args, log)
    else:
        {"train1": E5.stage_train1, "feats2": E5.stage_feats2, "train2": E5.stage_train2,
         "predict": E5.stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
