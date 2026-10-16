"""Mine token substitutions between S1 and matched S2/S3 records (after normalization).
For each pair, unmatched tokens on both sides; if exactly one unmatched token on each side -> substitution pair.
Frequent substitutions that are *not* typos reveal normalization gaps (abbreviations, variants).
Usage: python tools/subst_mining.py test France 0.98      (confident predicted pairs of the current best run)
       python tools/subst_mining.py train US              (ground-truth pairs)"""
import os, sys
from collections import Counter
import numpy as np, polars as pl
from rapidfuzz.distance import JaroWinkler
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from er_common import CACHE, RUNS
split, country = sys.argv[1], sys.argv[2]
prep = os.path.join(CACHE, "prep_v2" if split == "test" else "prep_v1")
s1 = pl.read_parquet(os.path.join(prep, f"{split}_s1.parquet"), columns=["name_core", "addr_clean", "country"])
r = pl.read_parquet(os.path.join(prep, f"{split}_r.parquet"), columns=["name_core", "addr_clean"])
if split == "test":
    thr = float(sys.argv[3])
    res = pl.read_parquet(os.path.join(RUNS, "exp05", "test_pred.parquet"))
    pairs = res.filter(pl.col("p") > thr).select("s1", "r")
else:
    pairs = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_gt.parquet")).rename({"s1_idx": "s1", "r_idx": "r"})
c = s1["country"].to_numpy()
pairs = pairs.filter(pl.Series(c[pairs["s1"].to_numpy()] == country))
pairs = pairs.sample(min(400000, pairs.height), seed=0)
A = s1[pairs["s1"].to_numpy()]; B = r[pairs["r"].to_numpy()]
for field in ["addr_clean", "name_core"]:
    sub, dele, ins = Counter(), Counter(), Counter()
    for a, b in zip(A[field].to_list(), B[field].to_list()):
        if not a or not b:
            continue
        ta, tb = a.split(), b.split()
        sa, sb = set(ta), set(tb)
        ua = [t for t in ta if t not in sb and not t.isdigit()]
        ub = [t for t in tb if t not in sa and not t.isdigit()]
        # drop typo-like pairs (high Jaro-Winkler)
        ua2 = [t for t in ua if not any(JaroWinkler.similarity(t, u) > 0.85 for u in ub)]
        ub2 = [u for u in ub if not any(JaroWinkler.similarity(t, u) > 0.85 for t in ua)]
        if len(ua2) == 1 and len(ub2) == 1:
            sub[(ua2[0], ub2[0])] += 1
        for t in ua2:
            dele[t] += 1
        for u in ub2:
            ins[u] += 1
    n = pairs.height
    print(f"=== {split} {country} {field}: {n} pairs")
    print("  top substitutions S1->R:", [(f"{a}->{b}", k) for (a, b), k in sub.most_common(40)])
    print("  top S1 tokens missing in R:", [(t, k) for t, k in dele.most_common(25)])
    print("  top R tokens not in S1:", [(t, k) for t, k in ins.most_common(25)])
