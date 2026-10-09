"""Sample US/India S1s with loss (dense OOF, exp29g) and print all candidates with p and label."""
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
from er_common import RUNS, CACHE
import exp03_dense as M3
country = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3])
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
f = np.load(_os.path.join(_HERE, 'f_per_s1.npy'))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32))
res = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
res = res.with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("rk"))
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country", "name_raw", "addr_raw"])
r = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["name_raw", "addr_raw", "addr_clean"])
ctry = s1["country"].to_numpy()
radr = (r["addr_clean"] != "").to_numpy()
# S1s with loss where at least one error pair involves an R with an address
bad = np.where(keep & (ctry == country) & (f < 0.999))[0]
print(country, "S1 with loss:", len(bad), "total loss share", (1 - f[bad]).sum() / keep.sum())
rng = np.random.default_rng(seed)
gset = gt.group_by("s1").agg(pl.col("r"))
gmap = dict(zip(gset["s1"].to_list(), gset["r"].to_list()))
shown = 0
for s in rng.permutation(bad):
    trs = set(gmap.get(int(s), []))
    sub = res.filter(pl.col("s1") == int(s)).sort("p", descending=True)
    cand_r = set(sub["r"].to_list())
    # skip cases where all errors are address-less
    err_r = [x for x in trs if x not in cand_r] + [x for x in sub.filter(pl.col("p") > 0.7)["r"].to_list() if x not in trs] + \
            [x for x in sub.filter(pl.col("p") <= 0.7)["r"].to_list() if x in trs]
    if all(not radr[x] for x in err_r):
        continue
    print("=" * 100)
    print(f"S1 F={f[s]:.3f} ntrue={len(trs)} | {s1['name_raw'][int(s)]} | {s1['addr_raw'][int(s)]}")
    for row in sub.iter_rows(named=True):
        lab = "T" if row["r"] in trs else "-"
        print(f"   {lab} p={row['p']:.3f} rk={row['rk']} | {r['name_raw'][row['r']]} | {r['addr_raw'][row['r']]}")
    for x in trs:
        if x not in cand_r:
            print(f"   T MISSING | {r['name_raw'][x]} | {r['addr_raw'][x]}")
    shown += 1
    if shown >= n:
        break
