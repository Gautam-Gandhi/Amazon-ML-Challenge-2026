"""Recall vs candidate-set size for pruning rules (s1_rank_j<=K or r_rank_j<=R).  Usage: python tools/prune_curve.py [India,US]"""
import os, sys
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
import exp01_match as M

gt = pl.read_parquet(os.path.join(M.PREP, "train_gt.parquet")).rename({"s1_idx": "s1", "r_idx": "r"})
gt = gt.with_columns(pl.lit(1, pl.Int8).alias("y"))
cs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["India", "US"]
s1c = pl.read_parquet(os.path.join(M.PREP, "train_s1.parquet"), columns=["country"]).with_row_index("s1")
gt = gt.join(s1c.with_columns(pl.col("s1").cast(pl.Int32)), on="s1").filter(pl.col("country").is_in(cs)).drop("country")
ngt = gt.height
nS1 = s1c.filter(pl.col("country").is_in(cs)).height
rules = [(k, rr) for k in [5, 8, 10, 12, 15, 20, 25, 30, 999] for rr in [0, 1, 2, 3]]
acc = {x: [0, 0] for x in rules}
for c in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["India", "US"]):
    cand = M.add_ranks(M.load_cand("train", c)).join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
    cand = cand.select("s1_rank_j", "r_rank_j", "y")
    for k, rr in rules:
        m = cand.filter((pl.col("s1_rank_j") <= k) | (pl.col("r_rank_j") <= rr))
        acc[(k, rr)][0] += m.height
        acc[(k, rr)][1] += int(m["y"].sum())
    del cand
for (k, rr), (n, y) in acc.items():
    print(f"s1_rank_j<={k:3d} or r_rank_j<={rr}: pairs={n} ({n / nS1:.1f}/S1) recall={y / ngt:.5f}")
