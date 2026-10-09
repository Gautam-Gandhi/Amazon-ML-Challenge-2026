"""Evaluate the exp11 word-role rule out-of-country with labels, on LOCO predictions saved by tools/loco.py --save_pred.
Usage: ER_WORK_DIR=work_v3 python tools/loco_rule_eval.py work_v3/runs/loco_pred/base_US_India.parquet India
"""
import os
import sys
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from er_common import CACHE  # noqa: E402
import exp03_dense as M3  # noqa: E402
import exp11_rolerule as R  # noqa: E402

pred_path, target = sys.argv[1], sys.argv[2]
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v2", "train_s1.parquet"), columns=["country", "name_core", "addr_nums"])
r = pl.read_parquet(os.path.join(CACHE, "prep_v2", "train_r.parquet"), columns=["country", "name_core", "addr_nums"])
country = s1["country"].to_numpy()
h = (np.arange(len(country), dtype=np.uint64) * np.uint64(0x9E3779B1) % np.uint64(1000)).astype(np.int32)
mask = keep & (country == target) & (h >= 600) & (h < 950)
res = pl.read_parquet(pred_path)
thrs = (0.7, 0.8, 0.85, 0.9, 0.93, 0.95)
M3.decode_eval(res, mask, print, "base", thrs=thrs)
d = R.role_pairs(R.assigned(res.select("s1", "r", "p")), s1, r).join(res.select("s1", "r", "y"), on=["s1", "r"])
rule = d.filter(pl.col("swap") & pl.col("hn_eq") & (pl.col("role_swap") >= 0.1) & (pl.col("wn") >= 200) & (pl.col("nsc") > 0.1))
flip = rule.filter((pl.col("p") <= 0.95) & (pl.col("p") > 0.01))
print(f"rule pairs {rule.height}, true {rule['y'].mean():.4f}; flips {flip.height}, flip TRUE rate {flip['y'].mean():.4f}")
print("  per word:", flip.group_by("w").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true"),
      pl.col("role_swap").first().round(2)).sort("n", descending=True).head(12).rows())
new = res.join(flip.select("s1", "r", pl.lit(0.95, pl.Float32).alias("pn")), on=["s1", "r"], how="left") \
         .with_columns(pl.max_horizontal("p", pl.col("pn").fill_null(0)).alias("p")).drop("pn")
M3.decode_eval(new, mask, print, "with rule", thrs=thrs)
