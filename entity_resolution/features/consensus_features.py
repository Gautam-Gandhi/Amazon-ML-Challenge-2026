"""Consensus features (9 columns): how a candidate fits among the other candidates of the same S1 entity.

A distractor business has its own group of records that all carry the same differing word or house number, while noise
changes records independently. Counted inside each S1 entity's candidate list:
  nx_sup_max / nx_sup_xsrc / nx_n2 : other candidates sharing a record-only name word (all / other source only),
                                     number of record-only words shared by >= 2 others
  ns_miss_max                       : candidates containing an S1 name word this record lacks
  nm_same_core                      : other candidates with exactly this record's core name
  hn_sup_r / hn_sup_s1 / hn_r_eq_s1 : candidates with the record's / the S1's first house number, and equality
  ax_sup_max                        : other candidates sharing a record-only address word
"""
import os
import glob

import numpy as np
import polars as pl

from ..utils import LOG, progress
from .pair_features import layer_dir

NEW_COLS = ["nx_sup_max", "nx_sup_xsrc", "nx_n2", "ns_miss_max", "nm_same_core", "hn_sup_r", "hn_sup_s1",
            "hn_r_eq_s1", "ax_sup_max"]


def _tok_long(keys, tok_list, col):
    """keys (row, s1, key) + list column per key -> long (row, s1, tok)."""
    return keys.with_columns(tok_list.alias(col)).explode(col).filter(pl.col(col).is_not_null() & (pl.col(col) != ""))


def consensus_chunk(P, s1_tab, r_tab):
    """P: (row, s1, r) for complete S1 groups. Returns (row, NEW_COLS...)."""
    si, ri = P["s1"].to_numpy(), P["r"].to_numpy()
    src = r_tab["src"].to_numpy()[ri]
    P = P.with_columns(pl.Series("src", src))
    # ---- name tokens
    rtok = _tok_long(P.select("row", "s1", "r", "src"), r_tab["tokn"].gather(ri), "tok").unique(["row", "tok"])
    cnt = rtok.group_by(["s1", "tok"]).agg(pl.len().alias("cnt"),
                                          (pl.col("src") == 2).sum().alias("c2"), (pl.col("src") == 3).sum().alias("c3"))
    s1u = np.unique(si)
    stok = _tok_long(pl.DataFrame({"row": np.zeros(len(s1u), np.int64), "s1": s1u}),
                     s1_tab["tokn"].gather(s1u), "tok").select("s1", "tok").unique().with_columns(pl.lit(1).alias("in_s1"))
    rt = rtok.join(cnt, on=["s1", "tok"]).join(stok, on=["s1", "tok"], how="left")
    extra = rt.filter(pl.col("in_s1").is_null()).with_columns(
        (pl.col("cnt") - 1).alias("sup"),
        pl.when(pl.col("src") == 2).then(pl.col("c3")).otherwise(pl.col("c2")).alias("xsup"))
    f_extra = extra.group_by("row").agg(pl.col("sup").max().alias("nx_sup_max"), pl.col("xsup").max().alias("nx_sup_xsrc"),
                                        (pl.col("sup") >= 2).sum().alias("nx_n2"))
    # S1 tokens missing in R: candidates containing them
    pm = P.select("row", "s1").join(stok.select("s1", "tok"), on="s1")
    pm = pm.join(rtok.select("row", "tok").with_columns(pl.lit(1).alias("has")), on=["row", "tok"], how="left") \
           .filter(pl.col("has").is_null()).join(cnt.select("s1", "tok", "cnt"), on=["s1", "tok"], how="left")
    f_miss = pm.group_by("row").agg(pl.col("cnt").fill_null(0).max().alias("ns_miss_max"))
    # identical core among candidates
    core = r_tab["name_core"].gather(ri)
    same = P.select("row", "s1").with_columns(core.alias("core"))
    same = same.with_columns((pl.len().over(["s1", "core"]) - 1).alias("nm_same_core")).select("row", "nm_same_core")
    # ---- house numbers (first number)
    rn = r_tab["hn1"].gather(ri)
    hn = P.select("row", "s1").with_columns(rn.alias("hn"), s1_tab["hn1"].gather(si).alias("hs"))
    hn = hn.with_columns(
        pl.when(pl.col("hn") != "").then(pl.len().over(["s1", "hn"]) - 1).otherwise(-1).alias("hn_sup_r"),
        (pl.col("hn") == pl.col("hs")).cast(pl.Int8).alias("hn_r_eq_s1"),
    )
    s1hn = hn.filter(pl.col("hn") == pl.col("hs")).group_by("s1").len("hn_sup_s1")
    hn = hn.join(s1hn, on="s1", how="left").with_columns(pl.col("hn_sup_s1").fill_null(0)) \
           .select("row", "hn_sup_r", "hn_sup_s1", "hn_r_eq_s1")
    # ---- address tokens (non-numeric)
    atok = _tok_long(P.select("row", "s1", "r"), r_tab["toka"].gather(ri), "tok").unique(["row", "tok"])
    acnt = atok.group_by(["s1", "tok"]).len("cnt")
    satok = _tok_long(pl.DataFrame({"row": np.zeros(len(s1u), np.int64), "s1": s1u}),
                      s1_tab["toka"].gather(s1u), "tok").select("s1", "tok").unique().with_columns(pl.lit(1).alias("in_s1"))
    ax = atok.join(acnt, on=["s1", "tok"]).join(satok, on=["s1", "tok"], how="left").filter(pl.col("in_s1").is_null()) \
             .group_by("row").agg((pl.col("cnt") - 1).max().alias("ax_sup_max"))
    out = P.select("row")
    for f in [f_extra, f_miss, same, hn, ax]:
        out = out.join(f, on="row", how="left")
    return out.with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in NEW_COLS if c in out.columns]).sort("row")


def tables(ws, split):
    def prep(tag):
        cols = ["name_core", "addr_clean", "addr_nums"] + (["src"] if tag == "r" else [])
        d = pl.read_parquet(ws.table(split, tag), columns=cols)
        return d.with_columns(
            pl.col("name_core").str.split(" ").alias("tokn"),
            pl.col("addr_clean").str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).alias("toka"),
            pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("hn1"))
    return prep("s1"), prep("r")


def build(ws):
    """Stage: consensus features for every base-feature part (row-aligned)."""
    for split in ["train", "test"]:
        src = layer_dir(ws, "base", split)
        out_dir = layer_dir(ws, "consensus", split)
        s1_tab, r_tab = tables(ws, split)
        for p in sorted(glob.glob(os.path.join(src, "part*.parquet"))):
            k = pl.read_parquet(p, columns=["s1", "r"]).with_row_index("row")
            # parts are sorted by (s1, r) within a country; an S1 group can straddle two parts only at a part
            # boundary (about one S1 entity per 3M rows)
            outs = []
            s1v = k["s1"].to_numpy()
            bounds = np.unique(s1v)
            chunk = 150000
            for i in progress(range(0, len(bounds), chunk), desc=f"consensus features {split} {os.path.basename(p)}",
                              unit="chunk"):
                lo, hi = bounds[i], bounds[min(i + chunk, len(bounds)) - 1]
                P = k.filter((pl.col("s1") >= lo) & (pl.col("s1") <= hi))
                outs.append(consensus_chunk(P, s1_tab, r_tab))
            f = pl.concat(outs).sort("row")
            assert f.height == k.height
            f.drop("row").write_parquet(os.path.join(out_dir, os.path.basename(p)))
            LOG.info("consensus features %s %s: %s pairs", split, os.path.basename(p), f"{k.height:,}")
