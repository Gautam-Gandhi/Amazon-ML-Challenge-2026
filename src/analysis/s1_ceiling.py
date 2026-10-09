"""Ceilings with stage 1 fixed (dense world, labels): perfect matcher on the candidates that survive p1 > thr,
vs on all candidates; the current final pipeline (exp29g OOF, assign @0.7); and where its remaining loss is (FP vs FN)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
import exp03_dense as M3
from er_common import CACHE, RUNS, macro_f05
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0)); n1 = len(keep)
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32))
gs, gr = gt["s1"].to_numpy(), gt["r"].to_numpy()
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country", "name_core"]).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32))
ctry = s1["country"].to_numpy()
noaddr = (pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["addr_clean"])["addr_clean"] == "").to_numpy()
cnt = s1.filter(pl.Series(keep)).group_by(["country", "name_core"]).len("nsame")
nsame = s1.join(cnt, on=["country", "name_core"], how="left").sort("s1")["nsame"].fill_null(0).to_numpy()
amb = lambda d: noaddr[d["r"].to_numpy()] & (nsame[d["s1"].to_numpy()] >= 2)
def F(d):
    f = macro_f05(d["s1"].to_numpy(), d["r"].to_numpy(), gs, gr, n1, return_per_entity=True)
    return f"ALL {f[keep].mean():.5f} | US {f[keep & (ctry == 'US')].mean():.5f} | India {f[keep & (ctry == 'India')].mean():.5f}"
oof = pl.read_parquet(os.path.join(RUNS, "exp13", "s1_oof.parquet"), columns=["s1", "r", "y", "p"]).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32))
tp = oof.filter(pl.col("y") == 1)
print(f"{'perfect matcher, all pruned candidates (no filter)':62s}", F(tp))
for t in [0.001, 0.003, 0.01, 0.03]:
    print(f"{f'perfect matcher, candidates with p1 > {t}':62s}", F(tp.filter(pl.col("p") > t)))
fin = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
print("exp29g OOF pairs:", fin.height, " of which true:", fin.join(gt, on=["s1", "r"], how="semi").height)
a = fin.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first").filter(pl.col("p") > 0.7).select("s1", "r")
cand_true = gt.join(fin.select("s1", "r"), on=["s1", "r"], how="semi")
print(f"{'current final pipeline (exp29g, assign @0.7)':62s}", F(a))
atrue = a.join(gt, on=["s1", "r"], how="semi")
print(f"{'  ... with every false positive removed':62s}", F(atrue))
print(f"{'  ... with every missed true candidate added (FPs kept)':62s}", F(pl.concat([a, cand_true]).unique()))
dec = pl.concat([cand_true.filter(pl.Series(~amb(cand_true))), a.filter(pl.Series(amb(a)))]).unique()
print(f"{'perfect except ambiguous address-less records':62s}", F(dec))
fp = a.height - atrue.height; fn = cand_true.height - atrue.height
print(f"final accepted {a.height:,}: FP {fp:,}, missed true candidates {fn:,}, true pairs never in candidates {gt.height - cand_true.height:,}")
# where are the final errors in terms of p1?
p1 = oof.select("s1", "r", "p")
fpd = a.join(gt, on=["s1", "r"], how="anti").join(p1, on=["s1", "r"], how="left")
fnd = cand_true.join(atrue, on=["s1", "r"], how="anti").join(p1, on=["s1", "r"], how="left")
for nm, d in [("final FP", fpd), ("final FN", fnd)]:
    q = d["p"].to_numpy()
    print(f"{nm}: p1 quantiles 10/50/90% {np.quantile(q, [0.1, 0.5, 0.9]).round(3)}; share with p1 > 0.5: {(q > 0.5).mean():.2f}")
