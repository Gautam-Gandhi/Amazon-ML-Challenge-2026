"""Inspect GT pairs missed by the exp01 embedding blocker on held-out queries.  Usage: python tools/blocker_misses.py fold0"""
import os, sys
import numpy as np
import polars as pl
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
import exp01_block as B
from er_common import s1_fold

dev = torch.device("cuda")
tag = sys.argv[1] if len(sys.argv) > 1 else "fold0"
nq = 5000
s1 = pl.read_parquet(os.path.join(B.PREP, "train_s1.parquet"))
r = pl.read_parquet(os.path.join(B.PREP, "train_r.parquet"))
gt = pl.read_parquet(os.path.join(B.PREP, "train_gt.parquet"))
fold = s1_fold(s1.height)
enc = B.Encoder("train", dev)
model = B.Towers(128).to(dev)
model.load_state_dict(torch.load(os.path.join(B.EMB, f"model_{tag}.pt")))
model.eval()
rng = np.random.default_rng(5)
country = s1["country"].to_numpy()
rc = r["country"].to_numpy()
gs, gr = gt["s1_idx"].to_numpy(), gt["r_idx"].to_numpy()
cats = {}
for c in ["US", "India"]:
    qs = rng.choice(np.where((country == c) & (fold == 1 - int(tag[-1])))[0], nq, replace=False)
    rr = np.where(rc == c)[0]
    res = B.search(enc, model, "s1", qs, "r", rr, {"name": 20, "addr": 20, "joint": 30}, 128)
    u = np.concatenate([res[ch][1] for ch in res], 1)
    qpos = {s: i for i, s in enumerate(qs)}
    rloc = np.full(len(rc), -1, np.int64); rloc[rr] = np.arange(len(rr))
    m = np.isin(gs, qs)
    miss = [(s, t) for s, t in zip(gs[m], gr[m]) if not (u[qpos[s]] == rloc[t]).any()]
    print(f"== {c}: {len(miss)} missed of {m.sum()}")
    cn, ca = B.pair_cos(enc, model, np.array([a for a, b in miss]), np.array([b for a, b in miss]))
    for (a, b), x, y in list(zip(miss, cn, ca))[:40]:
        A = s1.row(a, named=True); R = r.row(b, named=True)
        print(f"  cos_n={x:.2f} cos_a={y:.2f} | S1: {A['name_raw']} | {A['addr_raw']}\n{'':25}R : {R['name_raw']} | {R['addr_raw']}")
    # categories
    for (a, b), x, y in zip(miss, cn, ca):
        R = r.row(b, named=True)
        k = ("empty_addr" if R["addr_clean"] == "" else "has_addr") + ("|alias" if R["name_flags"] & 8 else "") + \
            ("|native" if R["name_flags"] & 1 else "") + ("|lowname" if x < 0.5 else "")
        cats[k] = cats.get(k, 0) + 1
print(sorted(cats.items(), key=lambda t: -t[1]))
