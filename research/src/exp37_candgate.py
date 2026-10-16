"""exp37_candgate: final candidate set = pairs that can still be accepted after the stage-2 matcher.

Why (09-27): candidate_pairs.tsv counts in the final ranking, and a smaller set per Source 1 entity ranks higher.
Every stage after stage 2 only acts on pairs with stage-2 p > 0.01:
  - the cross-encoders and stackers re-score the band 0.01 < p < 0.99;
  - the unseen-country word-role rule needs p > 0.01, the acronym rule p > 0.1;
  - decoding thresholds are 0.6 / 0.7 / 0.9.
So a pair with stage-2 p <= 0.01 can never be accepted, and the cascade gates it out before the cross-encoder stage.
The acronym candidate channel (exp35) is kept as is.
Measured on test: 6,845,820 -> 6,591,700 candidates (3.95 -> 3.80 per S1), 0 accepted pairs lost.
matching_results.tsv is byte-identical to the base run's.

Stage: predict (reads --base test_pred.parquet + output/matching_results.tsv and --stage2 test_pred.parquet)
"""
import os
import sys
import shutil
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, write_outputs, validate_outputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp37")
    ap.add_argument("--base", default="exp35", help="final decoded run (test_pred.parquet + output/)")
    ap.add_argument("--stage2", default="exp15", help="stage-2 run whose test_pred.parquet holds the stage-2 p")
    ap.add_argument("--gate", type=float, default=0.01)
    args = ap.parse_args()
    run_dir = os.path.join(RUNS, args.run)
    os.makedirs(run_dir, exist_ok=True)
    log = Logger(os.path.join(run_dir, "log_predict.txt"))
    log("args:", vars(args))
    P = os.path.join(CACHE, "prep_v2")
    s1_ids = pl.read_parquet(os.path.join(P, "test_s1.parquet"), columns=["eid"])["eid"].to_numpy()
    r_ids = pl.read_parquet(os.path.join(P, "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    cand = pl.read_parquet(os.path.join(RUNS, args.base, "test_pred.parquet")).select("s1", "r")
    s2 = pl.read_parquet(os.path.join(RUNS, args.stage2, "test_pred.parquet")).select("s1", "r", pl.col("p").alias("p2"))
    cand = cand.join(s2, on=["s1", "r"], how="left")
    kept = cand.filter(pl.col("p2").is_null() | (pl.col("p2") > args.gate))    # null = pairs of the acronym channel
    # accepted pairs of the base run
    base_out = os.path.join(RUNS, args.base, "output")
    m = pl.read_csv(os.path.join(base_out, "matching_results.tsv"), separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    m = m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids") \
         .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    s1_idx = pl.DataFrame({"source1_entity_id": s1_ids, "s1": np.arange(len(s1_ids), dtype=np.int64)})
    r_idx = pl.DataFrame({"matched_entity_ids": r_ids, "r": np.arange(len(r_ids), dtype=np.int64)})
    acc = m.join(s1_idx, on="source1_entity_id").join(r_idx, on="matched_entity_ids").select("s1", "r")
    lost = acc.join(kept.select(pl.col("s1").cast(pl.Int64), pl.col("r").cast(pl.Int64)), on=["s1", "r"], how="anti")
    log(f"candidates {cand.height:,} -> {kept.height:,} ({cand.height / len(s1_ids):.3f} -> {kept.height / len(s1_ids):.3f} per S1); "
        f"accepted pairs {acc.height:,}, of them outside the gated set: {lost.height}")
    assert lost.height == 0, "an accepted pair would leave the candidate set"
    out_dir = os.path.join(run_dir, "output")
    write_outputs(out_dir, s1_ids, acc["s1"].to_numpy(), acc["r"].to_numpy(), kept["s1"].to_numpy(), kept["r"].to_numpy(), r_ids)
    same = open(os.path.join(out_dir, "matching_results.tsv"), "rb").read() == open(os.path.join(base_out, "matching_results.tsv"), "rb").read()
    log(f"matching_results.tsv byte-identical to {args.base}: {same}")
    assert same
    kept.select("s1", "r").write_parquet(os.path.join(run_dir, "candidates.parquet"))
    shutil.copy2(os.path.join(RUNS, args.base, "test_pred.parquet"), os.path.join(run_dir, "test_pred.parquet"))
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


if __name__ == "__main__":
    main()
