"""Word-role features (13 columns), learned per split and country from the data itself (no labels).

The data generator adds words to names in two roles: noise words REPLACE a descriptor at the same address (a match),
family words are APPENDED to a sister company's name at another address (a different business). For every word, the
role table measures, over "near-duplicate" pairs (a record and its closest S1 entity whose names differ by one extra
record word and at most one missing S1 word): how often the word replaced another word (swap share), how often the
house number stayed the same, how often it occurs in this role, and its frequency / over-representation in names.
Pair features: min / max of these statistics over the record's extra words and the S1 entity's missing words.
"""
import os
import glob

import numpy as np
import polars as pl

from ..utils import LOG, progress
from .pair_features import layer_dir

MIN_N = 5
FEATS = ["rl_e_swap_min", "rl_e_swap_max", "rl_e_hneq_min", "rl_e_hneq_max", "rl_e_lrate_min", "rl_e_lrate_max",
         "rl_e_nsc_min", "rl_e_nsc_max", "rl_e_f1_max", "rl_m_swap_min", "rl_m_hneq_min", "rl_m_lrate_min", "rl_m_f1_max"]


def tables(ws, split):
    def prep(tag):
        cols = ["name_core", "addr_nums"] + (["country"] if tag == "s1" else [])
        d = pl.read_parquet(ws.table(split, tag), columns=cols)
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


def role_tables(ws, split, src, s1, r):
    parts = sorted(glob.glob(os.path.join(src, "part*.parquet")))
    acc_e, acc_m = [], []
    for p in progress(parts, desc=f"word-role statistics {split}", unit="part"):
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
    rcountry = pl.read_parquet(ws.table(split, "r"), columns=["country"])["country"]
    d1 = s1.select("country", pl.col("tok").alias("t")).explode("t").group_by(["country", "t"]).len("c1")
    dr = r.select(rcountry.alias("country"), pl.col("tok").alias("t")).explode("t").group_by(["country", "t"]).len("cr")
    nr = r.select(rcountry.alias("country")).group_by("country").len("NR")
    df = dr.join(d1, on=["country", "t"], how="full", coalesce=True).fill_null(0).join(n1, on="country").join(nr, on="country")
    df = df.select("country", "t", (pl.col("c1") / pl.col("N1")).alias("f1"),
                   (((pl.col("cr") + 1) / (pl.col("c1") + 1)).log() - (pl.col("NR") / pl.col("N1")).log()).alias("nsc"))
    return te, tm, df


def build(ws):
    """Stage: word-role features for every base-feature part (row-aligned)."""
    for split in ["train", "test"]:
        src = layer_dir(ws, "base", split)
        out_dir = layer_dir(ws, "roles", split)
        s1, r = tables(ws, split)
        te, tm, df = role_tables(ws, split, src, s1, r)
        te = te.join(df, on=["country", "t"], how="left")
        tm = tm.join(df.select("country", "t", "f1"), on=["country", "t"], how="left")
        parts = sorted(glob.glob(os.path.join(src, "part*.parquet")))
        for p in progress(parts, desc=f"word-role features {split}", unit="part"):
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
            f.select([pl.col(c).cast(pl.Float32) for c in FEATS]).write_parquet(os.path.join(out_dir, os.path.basename(p)))
            LOG.info("word-role features %s %s: %s pairs", split, os.path.basename(p), f"{k.height:,}")
