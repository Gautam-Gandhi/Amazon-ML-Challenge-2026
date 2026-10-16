"""Final decoding: rules -> assignment decoding -> candidate gate -> the two output files.

Assignment decoding: each S2/S3 record goes only to its best S1 entity, and the pair is accepted if p > the threshold
of its country (validated threshold for training countries, 0.9 for unseen countries). A training-country S1 entity
left without any match may take its best pair if p > 0.6.
Candidate gate: the final candidate set keeps the pairs that the post-matcher stages can still accept (stage-2
p > 0.01) plus the acronym candidates; every accepted pair is in it.
"""
import os

import numpy as np
import polars as pl

from .. import config as C
from ..utils import LOG
from ..data_io import write_outputs, validate_outputs
from ..models.gradient_boosting import model_dir
from . import rules


def decode(pred, country, threshold):
    seen = np.isin(country, list(C.TRAINING_COUNTRIES))
    t = np.where(seen, threshold, C.UNSEEN_THRESHOLD)
    a = pred.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.with_columns(pl.Series("t", t[a["s1"].to_numpy()]), pl.Series("seen", seen[a["s1"].to_numpy()]))
    acc = a.filter(pl.col("p") > pl.col("t"))
    extra = a.filter((pl.col("p") > C.EMPTY_THRESHOLD) & (pl.col("p") <= pl.col("t")) & pl.col("seen")
                     & ~pl.col("s1").is_in(acc["s1"].unique().implode())).sort("p", descending=True).unique(subset=["s1"], keep="first")
    return pl.concat([acc, extra.select(acc.columns)])


def candidate_gate(ws, pred, accepted):
    stage2 = pl.read_parquet(os.path.join(model_dir(ws, "stage2"), "test_pred.parquet")).select("s1", "r", pl.col("p").alias("p2"))
    cand = pred.select("s1", "r").join(stage2, on=["s1", "r"], how="left")
    kept = cand.filter(pl.col("p2").is_null() | (pl.col("p2") > C.CANDIDATE_GATE))      # null = acronym candidates
    lost = accepted.select(pl.col("s1").cast(pl.Int64), pl.col("r").cast(pl.Int64)).join(
        kept.select(pl.col("s1").cast(pl.Int64), pl.col("r").cast(pl.Int64)), on=["s1", "r"], how="anti")
    assert lost.height == 0, "an accepted pair would leave the candidate set"
    return kept


def finalize(ws, output_dir):
    """Stage: rules, decoding, candidate gate, output files, validation."""
    threshold = ws.load_json("models", "gbdt_stack", "threshold.json")["threshold"]
    pred = pl.read_parquet(os.path.join(model_dir(ws, "gbdt_stack"), "test_pred.parquet")).select("s1", "r", "p")
    pred = rules.france_word_roles(ws, pred)
    pred = rules.us_legal_form_siblings(ws, pred)
    pred = rules.french_acronyms(ws, pred)
    pred = rules.french_descriptor_swaps(ws, pred)
    pred = rules.french_acronym_candidates(ws, pred)
    s1 = pl.read_parquet(ws.table("test", "s1"), columns=["eid", "country"])
    country = s1["country"].to_numpy()
    acc = decode(pred, country, threshold)
    kept = candidate_gate(ws, pred, acc)
    r_ids = pl.read_parquet(ws.table("test", "r"), columns=["eid"])["eid"].to_numpy()
    write_outputs(output_dir, s1["eid"].to_numpy(), acc["s1"].to_numpy(), acc["r"].to_numpy(),
                  kept["s1"].to_numpy(), kept["r"].to_numpy(), r_ids)
    n = np.bincount(acc["s1"].to_numpy(), minlength=len(country))
    for c in np.unique(country):
        m = country == c
        LOG.info("%s: %.3f matches per S1 entity, %.4f without a match", c, n[m].mean(), np.mean(n[m] == 0))
    LOG.info("candidates: %s pairs (%.2f per S1 entity); matches: %s pairs", f"{kept.height:,}", kept.height / len(country),
             f"{acc.height:,}")
    if not validate_outputs(output_dir, ws.data_dir):
        raise RuntimeError("the output files did not pass the validator")
