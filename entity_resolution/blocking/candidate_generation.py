"""Candidate generation: search channels per country, then ranking and pruning to ~19 candidates per S1 entity.

Channels: name top-20, address top-20, joint (name + address) top-30, and a reverse channel where every S2/S3 record
retrieves its own top-3 S1 entities. A pair is kept if it is among the S1 entity's top 15 by joint similarity or the
S1 entity is among the record's top 2.
"""
import glob
import time

import numpy as np
import polars as pl

from .. import config as C
from ..utils import LOG, format_duration, s1_fold


def candidates_path(ws, split, country, tag):
    return ws.path("candidates", split, f"{country}_{tag}.parquet")


def search_candidates(ws, split):
    """Stage: raw candidate pairs with channel ranks and cosines. Training S1 entities of fold f are searched with the
    model trained on the other fold (out-of-fold); test uses the fold-0 model."""
    import torch                                    # imported here: the pruning helpers below are used without a GPU
    from .embedding_model import Encoder, Towers, model_path
    from .vector_search import ivf_search, pair_cosines
    dev = torch.device("cuda")
    s1c = pl.read_parquet(ws.table(split, "s1"), columns=["country"])["country"].to_numpy()
    rc = pl.read_parquet(ws.table(split, "r"), columns=["country"])["country"].to_numpy()
    n1 = len(s1c)
    enc = Encoder(ws, split, dev)
    if split == "train":
        fold = s1_fold(n1)
        jobs = [("fold1", np.where(fold == 0)[0]), ("fold0", np.where(fold == 1)[0])]
    else:
        jobs = [("fold0", np.arange(n1))]
    for tag, qrows_all in jobs:
        model = Towers(C.EMBED_DIM).to(dev)
        model.load_state_dict(torch.load(model_path(ws, tag)))
        model.eval()
        for c in np.unique(s1c):
            t0 = time.time()
            parts = []
            qs = qrows_all[s1c[qrows_all] == c]
            rr = np.where(rc == c)[0]
            if len(qs) == 0 or len(rr) == 0:
                continue
            res = ivf_search(enc, model, "s1", qs, "r", rr, C.SEARCH_K, C.EMBED_DIM, nprobe=C.NPROBE,
                             desc=f"{split} {c}: ")
            for ch, (sc, ix) in res.items():
                k = ix.shape[1]
                valid = ix >= 0
                parts.append(pl.DataFrame({
                    "s1": np.repeat(qs, k)[valid.ravel()].astype(np.int32),
                    "r": rr[ix[valid]].astype(np.int32),
                    "ch": np.full(valid.sum(), {"name": 0, "addr": 1, "joint": 2}[ch], np.int8),
                    "rank": np.tile(np.arange(k, dtype=np.int16), len(qs))[valid.ravel()],
                }))
            del res
            rres = ivf_search(enc, model, "r", rr, "s1", qs, {"joint": C.REVERSE_K}, C.EMBED_DIM, nprobe=C.NPROBE,
                              desc=f"{split} {c}: reverse ")
            sc, ix = rres["joint"]
            valid = ix >= 0
            parts.append(pl.DataFrame({
                "s1": qs[ix[valid]].astype(np.int32),
                "r": np.repeat(rr, C.REVERSE_K)[valid.ravel()].astype(np.int32),
                "ch": np.full(valid.sum(), 3, np.int8),
                "rank": np.tile(np.arange(C.REVERSE_K, dtype=np.int16), len(rr))[valid.ravel()],
            }))
            del rres
            job = (pl.concat(parts).group_by(["s1", "r"]).agg([
                pl.col("rank").filter(pl.col("ch") == i).min().fill_null(999).cast(pl.Int16).alias(f"rk_{nm}")
                for i, nm in enumerate(["name", "addr", "joint", "rev"])]))
            del parts
            cn, ca = pair_cosines(enc, model, job["s1"].to_numpy(), job["r"].to_numpy())
            job = job.with_columns(pl.Series("cos_n", cn), pl.Series("cos_a", ca)).sort(["s1", "r"])
            job.write_parquet(candidates_path(ws, split, c, tag))
            torch.cuda.empty_cache()
            LOG.info("search %s %s (model %s): %s S1 entities against %s records -> %s candidate pairs (%s)", split, c,
                     tag, f"{len(qs):,}", f"{len(rr):,}", f"{job.height:,}", format_duration(time.time() - t0))
        del model
    del enc
    torch.cuda.empty_cache()


def candidate_countries(ws, split):
    return sorted({p.replace("\\", "/").split("/")[-1].split("_")[0] for p in glob.glob(ws.path("candidates", split, "*.parquet"))})


def load_candidates(ws, split, country):
    parts = sorted(glob.glob(ws.path("candidates", split, f"{country}_*.parquet")))
    return pl.concat([pl.read_parquet(p) for p in parts])


def add_ranks(cand):
    """Joint similarity and the rank of each pair within its S1 entity and within its record."""
    return cand.with_columns((pl.col("cos_n") + pl.col("cos_a")).alias("cos_j")).with_columns([
        pl.col("cos_j").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("s1_rank_j"),
        pl.col("cos_j").rank("ordinal", descending=True).over("r").cast(pl.Int16).alias("r_rank_j"),
    ])


def prune(cand):
    return cand.filter((pl.col("s1_rank_j") <= C.PRUNE_TOP_S1) | (pl.col("r_rank_j") <= C.PRUNE_TOP_RECORD))


def competition_features(cand):
    """Competition over the pruned candidates, per S1 entity and per record."""
    return cand.with_columns([
        pl.len().over("s1").cast(pl.Int16).alias("s1_ncand"),
        (pl.col("cos_j") - pl.col("cos_j").max().over("s1")).alias("s1_gap_j"),
        (pl.col("cos_a") - pl.col("cos_a").max().over("s1")).alias("s1_gap_a"),
        (pl.col("cos_n") - pl.col("cos_n").max().over("s1")).alias("s1_gap_n"),
        pl.len().over("r").cast(pl.Int16).alias("r_ns1"),
        (pl.col("cos_j") - pl.col("cos_j").max().over("r")).alias("r_gap_j"),
        # margin to the best *other* S1 entity competing for this record
        (pl.col("cos_j") - pl.col("cos_j").sort(descending=True).slice(1, 1).first().over("r")).alias("r_margin2"),
    ])


def dense_world_mask(n_train_s1):
    """S1 entities kept in the dense training world (the other 20% become sources of distractor records)."""
    rng = np.random.default_rng(1000 + C.WORLD_SEED)
    return rng.random(n_train_s1) < C.KEEP_FRACTION
