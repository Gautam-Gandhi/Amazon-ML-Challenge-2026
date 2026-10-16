"""Validation in the dense training world: labels of the kept S1 entities and assignment-decoding F0.5."""
import numpy as np
import polars as pl

from .. import config as C
from ..utils import LOG, macro_f05
from ..blocking.candidate_generation import dense_world_mask


def train_keep_mask(ws):
    n1 = pl.read_parquet(ws.table("train", "s1"), columns=["country"]).height
    return dense_world_mask(n1)


def add_labels(ws, df):
    """Join the ground truth: y = 1 for true (s1, r) pairs."""
    gt = pl.read_parquet(ws.table("train", "gt")).rename({"s1_idx": "s1", "r_idx": "r"})
    gt = gt.with_columns(pl.lit(1, pl.Int8).alias("y"))
    return df.join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))


def labels_kept(ws, keep):
    gt = pl.read_parquet(ws.table("train", "gt"))
    return gt.filter(pl.Series(keep[gt["s1_idx"].to_numpy()]))


def f05_world(s, r, gt, keep):
    per = macro_f05(s, r, gt["s1_idx"].to_numpy(), gt["r_idx"].to_numpy(), len(keep), return_per_entity=True)
    return float(per[keep].mean())


def best_threshold(ws, res, keep, tag):
    """Assignment decoding (each record -> its best S1 entity, accept if p > thr) on out-of-fold predictions;
    returns the F0.5-optimal threshold of DECODE_GRID."""
    gt = labels_kept(ws, keep)
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    best = (None, -1)
    curve = []
    for thr in C.DECODE_GRID:
        d = a.filter(pl.col("p") > thr)
        f = f05_world(d["s1"].to_numpy(), d["r"].to_numpy(), gt, keep)
        curve.append(f"{thr}:{f:.5f}")
        if f > best[1]:
            best = (thr, f)
    LOG.info("[%s] validation F0.5 by threshold: %s", tag, " ".join(curve))
    LOG.info("[%s] best threshold %s, F0.5 %.5f", tag, best[0], best[1])
    return best[0], best[1]
