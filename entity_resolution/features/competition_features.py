"""Stage-2 features built from a first-stage probability p (17 columns):

  competition: rank / gap / sum / number of confident candidates within the S1 entity, rank / margin / runner-up within
               the record, expected number of other matches of the S1 entity
  consistency: similarity (name, address, numbers, source) of the record to the S1 entity's most confident other
               candidates ("anchors"); the true records of one business resemble each other
"""
import numpy as np
import polars as pl

from ..utils import LOG, progress


def group_feats(t):
    """t: (row, s1, r, p) -> competition features."""
    return t.with_columns([
        pl.col("p").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("p_rank_s1"),
        (pl.col("p") - pl.col("p").max().over("s1")).alias("p_gap_s1"),
        pl.col("p").sum().over("s1").alias("p_sum_s1"),
        (pl.col("p") > 0.5).sum().over("s1").cast(pl.Int16).alias("p_nhi_s1"),
        pl.col("p").rank("ordinal", descending=True).over("r").cast(pl.Int16).alias("p_rank_r"),
        (pl.col("p") - pl.col("p").sort(descending=True).slice(1, 1).first().over("r")).alias("p_margin_r"),
        pl.col("p").sort(descending=True).slice(1, 1).first().over("r").alias("p_2nd_r"),
    ]).with_columns([
        (pl.col("p_sum_s1") - pl.col("p")).alias("p_sum_other_s1"),
    ])


def consistency_feats(t, r_tab, n_anchor, chunk_s1=400000):
    """For each candidate (s1, r): similarity of r to the top-n_anchor other confident candidates of s1."""
    from rapidfuzz import process, fuzz
    anchors = (t.filter(pl.col("p") > 0.5)
                .with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("ar"))
                .filter(pl.col("ar") <= n_anchor + 1)
                .select("s1", pl.col("r").alias("a"), pl.col("p").alias("pa")))
    s1_vals = t["s1"].unique().sort().to_numpy()
    out = []
    names = r_tab["name_core"]
    addrs = r_tab["addr_clean"]
    nums = r_tab["addr_nums"]
    srcs = r_tab["src"].to_numpy()
    for i in progress(range(0, len(s1_vals), chunk_s1), desc="consistency features", unit="chunk"):
        lo, hi = s1_vals[i], s1_vals[min(i + chunk_s1, len(s1_vals)) - 1]
        tc = t.filter((pl.col("s1") >= lo) & (pl.col("s1") <= hi)).select("row", "s1", "r")
        ac = anchors.filter((pl.col("s1") >= lo) & (pl.col("s1") <= hi))
        j = tc.join(ac, on="s1", how="inner").filter(pl.col("r") != pl.col("a"))
        ri = j["r"].to_numpy()
        ai = j["a"].to_numpy()
        nm = process.cpdist(names.gather(ri).to_list(), names.gather(ai).to_list(), scorer=fuzz.token_sort_ratio,
                            workers=-1, dtype=np.uint8)
        ad = process.cpdist(addrs.gather(ri).to_list(), addrs.gather(ai).to_list(), scorer=fuzz.token_set_ratio,
                            workers=-1, dtype=np.uint8)
        nu = process.cpdist(nums.gather(ri).to_list(), nums.gather(ai).to_list(), scorer=fuzz.token_set_ratio,
                            workers=-1, dtype=np.uint8)
        a_empty = (addrs.gather(ai) == "").to_numpy()
        ad = np.where(a_empty, 255, ad)  # 255 = anchor has no address
        same_src = (srcs[ri] == srcs[ai]).astype(np.int8)
        j = j.with_columns(pl.Series("an_name", nm), pl.Series("an_addr", ad), pl.Series("an_num", nu),
                           pl.Series("an_same_src", same_src))
        j = j.with_columns([
            pl.when(pl.col("an_addr") == 255).then(None).otherwise(pl.col("an_addr")).alias("an_addr_v"),
        ])
        agg = j.group_by("row").agg([
            pl.col("an_name").max().alias("cons_name_max"),
            pl.col("an_name").mean().alias("cons_name_mean"),
            pl.col("an_addr_v").max().alias("cons_addr_max"),
            pl.col("an_num").max().alias("cons_num_max"),
            (pl.col("an_name") * pl.col("pa")).max().alias("cons_name_pw"),
            pl.col("pa").filter(pl.col("an_name") == pl.col("an_name").max()).first().alias("cons_name_pa"),
            ((pl.col("an_name") >= 90) & (pl.col("an_same_src") == 0)).sum().cast(pl.Int8).alias("cons_n_name90_xsrc"),
            ((pl.col("an_name") >= 90) & (pl.col("an_same_src") == 1)).sum().cast(pl.Int8).alias("cons_n_name90_ssrc"),
            pl.len().cast(pl.Int8).alias("cons_n_anchor"),
        ])
        out.append(agg)
    LOG.info("consistency features: %s candidate pairs", f"{t.height:,}")
    agg = pl.concat(out)
    return t.select("row").join(agg, on="row", how="left").sort("row")
