"""exp39 fallback (09-27 22:25): exp36 + two measured micro-gains.
(1) France pocket: identical name_core, record with an address but without a house number, record's best S1,
    p in (0.7, 0.9] -> p = 0.95. Out-of-country (LOCO, both directions) true rate 0.86 / 0.89 in that band; ~400 pairs.
(2) seen-country thresholds t_main 0.75 / t_empty 0.7 (dense OOF of exp30g: +0.00001 US, +0.00002 India) via a
    metrics run exp30g_t75 whose combined.best_thr = 0.75; rules exp31-35 re-run on top with --t_empty 0.7.
This script writes runs/exp39p/test_pred.parquet (= exp30gf test_pred with the pocket flipped) and exp30g_t75/metrics.json."""
import os, sys, json, shutil
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS
P = os.path.join(CACHE, "prep_v2")
s1 = pl.read_parquet(os.path.join(P, "test_s1.parquet"), columns=["country", "name_core"])
r = pl.read_parquet(os.path.join(P, "test_r.parquet"), columns=["name_core", "addr_clean", "addr_nums"])
tp = pl.read_parquet(os.path.join(RUNS, "exp30gf", "test_pred.parquet"))
si, ri = tp["s1"].to_numpy(), tp["r"].to_numpy()
t = tp.with_row_index("i").with_columns(
    pl.Series("country", s1["country"].to_numpy()[si]),
    pl.Series("same", s1["name_core"].to_numpy()[si] == r["name_core"].to_numpy()[ri]),
    pl.Series("has_addr", (r["addr_clean"] != "").to_numpy()[ri]),
    pl.Series("no_num", (r["addr_nums"].fill_null("") == "").to_numpy()[ri]))
best = t.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
hit = best.filter((pl.col("country") == "France") & pl.col("same") & pl.col("has_addr") & pl.col("no_num")
                  & (pl.col("p") > 0.7) & (pl.col("p") <= 0.9))
print("pocket flips:", hit.height)
p = tp["p"].to_numpy().copy(); p[hit["i"].to_numpy()] = 0.95
os.makedirs(os.path.join(RUNS, "exp39p"), exist_ok=True)
tp.with_columns(pl.Series("p", p.astype(tp["p"].dtype.to_python() if False else np.float32))).write_parquet(os.path.join(RUNS, "exp39p", "test_pred.parquet"))
m = json.load(open(os.path.join(RUNS, "exp30g", "metrics.json")))
m["combined"]["best_thr"] = 0.75
os.makedirs(os.path.join(RUNS, "exp30g_t75"), exist_ok=True)
json.dump(m, open(os.path.join(RUNS, "exp30g_t75", "metrics.json"), "w"))
print("ok")
