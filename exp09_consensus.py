"""exp09_consensus: cluster-consensus features (structural, vocabulary-free) on top of the v3 world (exp07).

Why (two-stage LOCO, tools/loco2.py): out-of-country the loss is dominated by FALSE MATCHES on singletons
(0.91-0.96) and one-match S1 (0.87-0.94): the model accepts look-alike *distractor businesses*. A distractor
("Recherche Pharmacie" next to S1 "Recherche Club", same street) has its OWN ~3-4 S2/S3 records, and all of them
carry the differing word / the same other house number. Noise alters words in individual records independently.
So: for a candidate R of S1, is R's differing token SHARED by other candidates of the same S1 (-> separate entity)
or unique to R (-> noise)? This does not depend on vocabulary, so it should transfer to France.
Features (per pair, computed inside the S1's candidate list):
  nx_sup_max / nx_sup_xsrc / nx_n2 : max #other candidates sharing an R-only name token (all / other source only),
                                     #R-only tokens shared by >= 2 others
  ns_miss_max                       : max #candidates containing an S1 name token that R lacks
  nm_same_core                      : #other candidates with exactly R's name_core
  hn_sup_r / hn_sup_s1 / hn_r_eq_s1 : #other candidates with R's first house number / with S1's first house number
  ax_sup_max                        : max #other candidates sharing an R-only (non-numeric) address token
Stages: feats | train1 | feats2 | train2 | predict   (model stages = exp05 code with 3 feature layers)
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

FEAT9 = os.path.join(CACHE, "feat_v9")
S2DIR9 = os.path.join(CACHE, "feat_v9s2")
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


def tables(split):
    def prep(tag):
        cols = ["name_core", "addr_clean", "addr_nums"] + (["src"] if tag == "r" else [])
        d = pl.read_parquet(os.path.join(E5.PREP[split], f"{split}_{tag}.parquet"), columns=cols)
        return d.with_columns(
            pl.col("name_core").str.split(" ").alias("tokn"),
            pl.col("addr_clean").str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).alias("toka"),
            pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("hn1"))
    return prep("s1"), prep("r")


def stage_feats(args, log):
    for split, src in [("train", E5.FEAT3), ("test", E5.FEAT4T)]:
        out_dir = os.path.join(FEAT9, split)
        os.makedirs(out_dir, exist_ok=True)
        s1_tab, r_tab = tables(split)
        log(f"{split}: tables ready")
        for p in sorted(glob.glob(os.path.join(src, "part*.parquet"))):
            dst = os.path.join(out_dir, os.path.basename(p))
            if os.path.exists(dst):
                continue
            k = pl.read_parquet(p, columns=["s1", "r"]).with_row_index("row")
            # parts are sorted by (s1, r) within country; an S1 group can straddle two parts only at a part
            # boundary (~1 S1 per 3M rows) -> its consensus counts see part of the group (negligible)
            outs = []
            s1v = k["s1"].to_numpy()
            bounds = np.unique(s1v)
            chunk = 150000
            for i in range(0, len(bounds), chunk):
                lo, hi = bounds[i], bounds[min(i + chunk, len(bounds)) - 1]
                P = k.filter((pl.col("s1") >= lo) & (pl.col("s1") <= hi))
                outs.append(consensus_chunk(P, s1_tab, r_tab))
            f = pl.concat(outs).sort("row")
            assert f.height == k.height
            f.drop("row").write_parquet(dst)
            log(f"{split} {os.path.basename(p)}: {k.height} rows")


def use_layers(args):
    """exp05 model stages over 3 layers: stage-1 base (feat_v3 / test) + exp07 tokens (feat_v7) + consensus (feat_v9)."""
    E5.FEAT5 = E7.FEAT7
    E5.S2DIR = S2DIR9
    if args.test_feat:
        E5.FEAT4T = os.path.join(CACHE, args.test_feat)

    def iter_s1(split):
        src = E5.FEAT3 if split == "train" else E5.FEAT4T
        p1 = sorted(glob.glob(os.path.join(src, "part*.parquet")))
        p7 = sorted(glob.glob(os.path.join(E7.FEAT7, split, "part*.parquet")))
        p9 = sorted(glob.glob(os.path.join(FEAT9, split, "part*.parquet")))
        assert len(p1) == len(p7) == len(p9), (len(p1), len(p7), len(p9))
        return lambda: (pl.concat([pl.read_parquet(a), pl.read_parquet(b), pl.read_parquet(c)], how="horizontal")
                        for a, b, c in zip(p1, p7, p9))
    E5.iter_s1 = iter_s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "train1", "feats2", "train2", "predict"])
    ap.add_argument("--run", default="exp09")
    ap.add_argument("--test_feat", default="")
    ap.add_argument("--neg_frac", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    args = ap.parse_args()
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
