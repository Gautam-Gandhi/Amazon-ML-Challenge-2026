"""Score ceilings on the dense training world (labels), current candidate set (exp29g OOF pairs):
 A: perfect matcher on the candidates (accept exactly the true candidate pairs)
 B: perfect everywhere EXCEPT address-less records whose true S1 name is shared by >= 2 S1 (keep the model's actual
    decisions there: unresolvable ambiguity) -> the most a better matcher could reach
 C: the current pipeline (exp29g decode, thr 0.7)."""
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
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
n1 = len(keep)
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32))
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country", "name_core"]).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32))
r = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["addr_clean"])
noaddr = (r["addr_clean"] == "").to_numpy()
cnt = s1.filter(pl.Series(keep)).group_by(["country", "name_core"]).len("nsame")
nsame = s1.join(cnt, on=["country", "name_core"], how="left").sort("s1")["nsame"].fill_null(0).to_numpy()
ctry = s1["country"].to_numpy()
oof = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
cand_true = gt.join(oof.select("s1", "r"), on=["s1", "r"], how="semi")
a = oof.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first").filter(pl.col("p") > 0.7).select("s1", "r")
gs, gr = gt["s1"].to_numpy(), gt["r"].to_numpy()
def F(pairs, mask=None):
    f = macro_f05(pairs["s1"].to_numpy(), pairs["r"].to_numpy(), gs, gr, n1, return_per_entity=True)
    out = {"ALL": f[keep].mean()}
    for c in ["US", "India"]:
        out[c] = f[keep & (ctry == c)].mean()
    return out
amb_true = (noaddr[cand_true["r"].to_numpy()]) & (nsame[cand_true["s1"].to_numpy()] >= 2)
amb_pred = (noaddr[a["r"].to_numpy()]) & (nsame[a["s1"].to_numpy()] >= 2)
A = cand_true
B = pl.concat([cand_true.filter(pl.Series(~amb_true)), a.filter(pl.Series(amb_pred))]).unique()
# B2: also give up only on the 'no candidate' misses — i.e. perfect matcher, nothing else changed (= A); and
# D: perfect recall AND perfect matcher (every true pair) = 1.0 by definition; E: all true pairs except the ambiguous address-less ones
E = gt.filter(pl.Series(~((noaddr[gs]) & (nsame[gs] >= 2))))
for tag, p in [("A  perfect matcher on today's candidates", A), ("B  ... but ambiguous address-less left as the model decides", B),
               ("C  current pipeline (dense validation)", a), ("E  every true pair except ambiguous address-less (perfect blocking too)", E)]:
    f = F(p)
    print(f"{tag:72s} ALL {f['ALL']:.5f} | US {f['US']:.5f} | India {f['India']:.5f}")
