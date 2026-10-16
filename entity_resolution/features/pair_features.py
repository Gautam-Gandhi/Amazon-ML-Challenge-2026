"""Base pair features (65 columns): blocking ranks, cosines and competition, fuzzy name and address scores, number and
legal-form agreement, record flags, address structure, name frequency and length.

Test candidates are the pruned search output. Training candidates are built in the dense world: the S1 entities that
are dropped (20%) lose their candidates, and ranks, competition and name counts are recomputed without them.
"""
import os
import time

import numpy as np
import polars as pl

from .. import config as C
from ..utils import LOG, format_duration, progress
from ..blocking.candidate_generation import (add_ranks, candidate_countries, competition_features,
                                             dense_world_mask, load_candidates, prune)

S1_COLS = ["name_core", "name_clean", "name_sfx", "addr_clean", "addr_nums", "addr_ncomp", "name_flags", "country"]
R_COLS = S1_COLS + ["name_alias", "src"]


def layer_dir(ws, layer, split):
    return ws.dir("features", layer, split)


def _with_name_counts(s1, r, s1_for_counts):
    s1_cnt = s1_for_counts.group_by(["country", "name_core"]).len("s1_core_cnt")
    r_cnt = r.group_by(["country", "name_core"]).len("r_core_cnt")
    s1 = s1.join(s1_cnt, on=["country", "name_core"], how="left").join(r_cnt, on=["country", "name_core"], how="left").sort("i")
    r = r.join(s1_cnt, on=["country", "name_core"], how="left").join(r_cnt, on=["country", "name_core"], how="left").sort("i")
    fill = [pl.col("s1_core_cnt").fill_null(0).cast(pl.Int32), pl.col("r_core_cnt").fill_null(0).cast(pl.Int32)]
    return s1.with_columns(fill), r.with_columns(fill)


def load_tables(ws, split, keep=None):
    """S1 and record tables with name-frequency counts (S1 counts only over kept S1 entities in the dense world)."""
    s1 = pl.read_parquet(ws.table(split, "s1"), columns=S1_COLS).with_row_index("i")
    r = pl.read_parquet(ws.table(split, "r"), columns=R_COLS).with_row_index("i")
    return _with_name_counts(s1, r, s1 if keep is None else s1.filter(pl.Series(keep)))


def fuzz_feats(a, b, prefix):
    """rapidfuzz pairwise scores (multithreaded C++) -> dict of uint8 arrays."""
    from rapidfuzz import process, fuzz
    from rapidfuzz.distance import JaroWinkler
    out = {}
    for nm, sc in [("ratio", fuzz.ratio), ("tsort", fuzz.token_sort_ratio), ("tset", fuzz.token_set_ratio),
                   ("partial", fuzz.partial_ratio)]:
        out[f"{prefix}_{nm}"] = process.cpdist(a, b, scorer=sc, workers=-1, dtype=np.uint8)
    if prefix.startswith("n"):
        out[f"{prefix}_jw"] = (process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=-1,
                                              dtype=np.float32) * 100).astype(np.uint8)
    return out


def _toks(c):
    return pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _inter(a, b):
    return _toks(a).list.set_intersection(_toks(b)).list.len().cast(pl.Int8)


def _jacc(a, b):
    i = _toks(a).list.set_intersection(_toks(b)).list.len()
    u = _toks(a).list.set_union(_toks(b)).list.len()
    return pl.when(u > 0).then(i / u).otherwise(0.0).cast(pl.Float32)


def write_feature_parts(cand, s1, r, out_dir, pi, desc="base features"):
    """Features of one country's candidates, written in parts of FEATURE_CHUNK_ROWS rows. Returns the next part id."""
    from rapidfuzz import process, fuzz
    chunk = C.FEATURE_CHUNK_ROWS
    for st in progress(range(0, cand.height, chunk), desc=desc, unit="part"):
        t0 = time.time()
        c = cand.slice(st, chunk)
        A = s1[c["s1"].to_numpy()]
        B = r[c["r"].to_numpy()]
        f = {}
        f.update(fuzz_feats(A["name_core"].to_list(), B["name_core"].to_list(), "nc"))
        f.update(fuzz_feats(A["name_clean"].to_list(), B["name_clean"].to_list(), "ncl"))
        a_ns = A["name_core"].str.replace_all(" ", "").to_list()
        b_ns = B["name_core"].str.replace_all(" ", "").to_list()
        f["nns_ratio"] = process.cpdist(a_ns, b_ns, scorer=fuzz.ratio, workers=-1, dtype=np.uint8)
        f["nns_partial"] = process.cpdist(a_ns, b_ns, scorer=fuzz.partial_ratio, workers=-1, dtype=np.uint8)
        f.update(fuzz_feats(A["addr_clean"].to_list(), B["addr_clean"].to_list(), "ad"))
        an = A["addr_nums"].to_list()
        bn = B["addr_nums"].to_list()
        f["num_tset"] = process.cpdist(an, bn, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.uint8)
        a1 = A["addr_nums"].str.split(" ").list.first().fill_null("")
        b1 = B["addr_nums"].str.split(" ").list.first().fill_null("")
        f["num1_ratio"] = process.cpdist(a1.to_list(), b1.to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.uint8)
        df = pl.DataFrame({
            "an": A["addr_nums"], "bn": B["addr_nums"], "a1": a1, "b1": b1,
            "at": A["addr_clean"], "bt": B["addr_clean"], "nt": A["name_core"], "mt": B["name_core"],
            "asf": A["name_sfx"], "bsf": B["name_sfx"],
        })
        sets = df.select(
            pl.when((pl.col("a1") == "") | (pl.col("b1") == "")).then(-1)
            .otherwise((pl.col("a1") == pl.col("b1")).cast(pl.Int8)).cast(pl.Int8).alias("num1_eq"),
            _jacc("an", "bn").alias("num_jacc"),
            _inter("an", "bn").alias("num_inter"),
            _jacc("at", "bt").alias("ad_jacc"),
            _jacc("nt", "mt").alias("nc_jacc"),
            (pl.col("nt").str.split(" ").list.first() == pl.col("mt").str.split(" ").list.first())
            .cast(pl.Int8).alias("nc_first_eq"),
            (pl.col("asf") == pl.col("bsf")).cast(pl.Int8).alias("sfx_eq"),
            _jacc("asf", "bsf").alias("sfx_jacc"),
            pl.col("an").str.count_matches(r"\S+").cast(pl.Int8).alias("a_nnum"),
            pl.col("bn").str.count_matches(r"\S+").cast(pl.Int8).alias("b_nnum"),
            pl.col("nt").str.count_matches(r"\S+").cast(pl.Int8).alias("a_ntok"),
            pl.col("mt").str.count_matches(r"\S+").cast(pl.Int8).alias("b_ntok"),
            pl.col("at").str.count_matches(r"\S+").cast(pl.Int16).alias("a_natok"),
            pl.col("bt").str.count_matches(r"\S+").cast(pl.Int16).alias("b_natok"),
            (pl.col("asf") != "").cast(pl.Int8).alias("a_hassfx"),
            (pl.col("bsf") != "").cast(pl.Int8).alias("b_hassfx"),
        )
        fl = B["name_flags"].to_numpy()
        extra = pl.DataFrame({
            "b_native": (fl & 1).astype(np.int8), "b_domain": ((fl >> 1) & 1).astype(np.int8),
            "b_hashtag": ((fl >> 2) & 1).astype(np.int8), "b_alias": ((fl >> 3) & 1).astype(np.int8),
            "b_junk": ((fl >> 4) & 1).astype(np.int8),
            "src": B["src"].to_numpy(),
            "b_addr_empty": (B["addr_clean"] == "").cast(pl.Int8).to_numpy(),
            "a_ncomp": A["addr_ncomp"].to_numpy(), "b_ncomp": B["addr_ncomp"].to_numpy(),
            "a_s1cnt": A["s1_core_cnt"].to_numpy(), "a_rcnt": A["r_core_cnt"].to_numpy(),
            "b_s1cnt": B["s1_core_cnt"].to_numpy(), "b_rcnt": B["r_core_cnt"].to_numpy(),
            "a_nlen": A["name_core"].str.len_chars().cast(pl.Int16).to_numpy(),
            "b_nlen": B["name_core"].str.len_chars().cast(pl.Int16).to_numpy(),
        })
        feat = pl.concat([c, pl.DataFrame(f), sets, extra], how="horizontal")
        feat.write_parquet(os.path.join(out_dir, f"part{pi:03d}.parquet"))
        LOG.info("base features part %d: %s pairs, %d columns (%s)", pi, f"{feat.height:,}", feat.width,
                 format_duration(time.time() - t0))
        pi += 1
    return pi


def build_test_features(ws):
    """Stage: pruned test candidates + base features."""
    out_dir = layer_dir(ws, "base", "test")
    s1, r = load_tables(ws, "test")
    pi = 0
    for country in candidate_countries(ws, "test"):
        cand = add_ranks(load_candidates(ws, "test", country))
        n0 = cand.height
        cand = competition_features(prune(cand)).sort(["s1", "r"])
        LOG.info("test %s: %s candidate pairs -> %s after pruning", country, f"{n0:,}", f"{cand.height:,}")
        pi = write_feature_parts(cand, s1, r, out_dir, pi, desc=f"base features test {country}")


def build_train_features(ws):
    """Stage: pruned training candidates of the dense world + base features."""
    n1 = pl.read_parquet(ws.table("train", "s1"), columns=["country"]).height
    keep = dense_world_mask(n1)
    out_dir = layer_dir(ws, "base", "train")
    s1, r = load_tables(ws, "train", keep)
    keep_idx = pl.Series("s1", np.where(keep)[0].astype(np.int32))
    LOG.info("dense world: %s of %s training S1 entities kept", f"{keep.sum():,}", f"{len(keep):,}")
    pi = 0
    for country in ["India", "US"]:
        cand = load_candidates(ws, "train", country)
        cand = cand.filter(pl.col("s1").is_in(keep_idx))
        cand = add_ranks(cand)
        n0 = cand.height
        cand = competition_features(prune(cand)).sort(["s1", "r"])
        LOG.info("train %s: %s candidate pairs -> %s after pruning", country, f"{n0:,}", f"{cand.height:,}")
        pi = write_feature_parts(cand, s1, r, out_dir, pi, desc=f"base features train {country}")
