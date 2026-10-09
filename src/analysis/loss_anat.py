"""Loss anatomy of the current best dense OOF (exp29g oof_combined, assign + thr 0.7 / t_empty 0.6)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
sys.path.insert(0, _REPO)
os.environ["ER_WORK_DIR"] = _WORK
import importlib
import er_common
importlib.reload(er_common)
from er_common import RUNS, CACHE, macro_f05
import exp03_dense as M3

keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
n1 = len(keep)
gt = M3.labels_kept(keep)
res = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
t_main, t_empty = 0.7, 0.6
acc = a.filter(pl.col("p") > t_main)
has = np.zeros(n1, bool); has[acc["s1"].to_numpy()] = True
resc = a.filter((pl.col("p") > t_empty) & (pl.col("p") <= t_main)).filter(~pl.Series(has[a.filter((pl.col("p") > t_empty) & (pl.col("p") <= t_main))["s1"].to_numpy()]))
resc = resc.sort("p", descending=True).unique(subset=["s1"], keep="first")
d = pl.concat([acc, resc])
ps, pr = d["s1"].to_numpy(), d["r"].to_numpy()
gs, gr = gt["s1_idx"].to_numpy(), gt["r_idx"].to_numpy()
f = macro_f05(ps, pr, gs, gr, n1, return_per_entity=True)
print("F0.5 kept:", f[keep].mean(), "n kept", keep.sum())
s1c = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country", "name_core", "addr_clean"])
rr = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["name_core", "addr_clean"])
ctry = s1c["country"].to_numpy()
for c in ["US", "India"]:
    m = keep & (ctry == c)
    print(c, "F", f[m].mean(), "loss", (1 - f[m]).sum() / keep.sum())
# pair-level categories
kp = set(zip(ps.tolist(), pr.tolist()))
cand = set(zip(res["s1"].to_numpy().tolist(), res["r"].to_numpy().tolist()))
r_noaddr = (rr["addr_clean"] == "").to_numpy()
# name multiplicity among kept S1 of same country
s1k = s1c.with_row_index("i").filter(pl.Series(keep))
cnt = s1k.group_by(["country", "name_core"]).len("nn")
s1n = s1c.with_row_index("i").join(cnt, on=["country", "name_core"], how="left").sort("i")["nn"].fill_null(0).to_numpy()
gk = keep[gs]
gs, gr = gs[gk], gr[gk]
fn_in = []; fn_out = []
best_r = dict(zip(a["r"].to_numpy().tolist(), a["s1"].to_numpy().tolist()))
cats = {}
for s, r in zip(gs.tolist(), gr.tolist()):
    if (s, r) in kp:
        continue
    incand = (s, r) in cand
    other = best_r.get(r, -1)
    wrong_s1 = incand and other != s
    k = ("noaddr" if r_noaddr[r] else "addr") + ("_shared" if s1n[s] >= 2 else "_uniq") + ("|incand" if incand else "|notcand") + ("|lost2other" if wrong_s1 else "")
    cats[k] = cats.get(k, 0) + 1
print("FN categories (pairs):")
for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"   {k:40s} {v:8d}")
# FP categories
gset = set(zip(gs.tolist(), gr.tolist()))
r_owner = dict(zip(gr.tolist(), gs.tolist()))
all_gt_r_owner = dict(zip(gt["r_idx"].to_numpy().tolist(), gt["s1_idx"].to_numpy().tolist()))
fpc = {}
for s, r in zip(ps.tolist(), pr.tolist()):
    if not keep[s] or (s, r) in gset:
        continue
    o = all_gt_r_owner.get(r, -1)
    k = "pure_distractor" if o < 0 else ("orphan(owner dropped)" if not keep[o] else "belongs_other_kept")
    k += "|noaddr" if r_noaddr[r] else "|addr"
    fpc[k] = fpc.get(k, 0) + 1
print("FP categories (pairs):")
for k, v in sorted(fpc.items(), key=lambda x: -x[1]):
    print(f"   {k:40s} {v:8d}")
np.save(_os.path.join(_HERE, 'f_per_s1.npy'), f)
