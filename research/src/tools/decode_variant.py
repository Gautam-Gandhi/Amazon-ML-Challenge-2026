"""Re-decode an existing run's test predictions with per-country thresholds -> new submission TSVs.

Countries not listed in --thr_map use --thr. Useful for unseen-country (France) threshold probes.
Usage: python tools/decode_variant.py --src exp04 --out exp04_fr90 --thr 0.7 --thr_map France=0.9 [--prep prep_v2]
"""
import os
import sys
import argparse
import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from er_common import CACHE, RUNS, write_outputs, validate_outputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--thr_map", default="")
    ap.add_argument("--prep", default="prep_v2")
    args = ap.parse_args()
    tmap = {k: float(v) for k, v in (kv.split("=") for kv in args.thr_map.split(",") if kv)}
    prep = os.path.join(CACHE, args.prep)
    s1 = pl.read_parquet(os.path.join(prep, "test_s1.parquet"), columns=["eid", "country"])
    r_ids = pl.read_parquet(os.path.join(prep, "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    country = s1["country"].to_numpy()
    res = pl.read_parquet(os.path.join(RUNS, args.src, "test_pred.parquet"))
    thr = np.full(len(country), args.thr, np.float32)
    for c, t in tmap.items():
        thr[country == c] = t
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.with_columns(pl.Series("t", thr[a["s1"].to_numpy()])).filter(pl.col("p") > pl.col("t"))
    out_dir = os.path.join(RUNS, args.out, "output")
    write_outputs(out_dir, s1["eid"].to_numpy(), a["s1"].to_numpy(), a["r"].to_numpy(),
                  res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    n = np.bincount(a["s1"].to_numpy(), minlength=len(country))
    for c in np.unique(country):
        m = country == c
        print(f"{c}: thr {tmap.get(c, args.thr)} mean matches {n[m].mean():.3f} empty {np.mean(n[m] == 0):.4f}")
    with open(os.path.join(RUNS, args.out, "README.txt"), "w") as f:
        f.write(f"decode of runs/{args.src}/test_pred.parquet with thr={args.thr} thr_map={tmap}\n")
    print("VALID" if validate_outputs(out_dir) else "INVALID")


if __name__ == "__main__":
    main()
