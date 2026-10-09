"""Is the stale reverse-channel rank (rk_rev) in the dense training world a problem?
1) how often it differs from a correctly recomputed value (dense world, pruned train candidates)
2) how much the stage-1 / stage-2 models rely on it (gain share)"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, glob, json, argparse
import numpy as np, polars as pl
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
import exp03_dense as M3
C = _os.path.join(_WORK, 'data', 'cache')
R = _os.path.join(_WORK, 'runs')
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32), pl.lit(1).alias("y"))
parts = sorted(glob.glob(os.path.join(C, "feat_v3", "k80s0", "part*.parquet")))
d = pl.concat([pl.read_parquet(f, columns=["s1", "r", "rk_rev"]) for f in parts])
print("dense-world pruned train pairs:", d.height, " with a reverse rank (<999):", (d["rk_rev"] < 999).sum())
# recompute: among the record's reverse-retrieved S1s that are KEPT, re-rank by the stored rank
rv = d.filter(pl.col("rk_rev") < 999).with_columns(pl.col("rk_rev").rank("ordinal").over("r").cast(pl.Int16).alias("rk_new") - 1)
rv = rv.join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
diff = rv["rk_rev"] != rv["rk_new"]
print(f"reverse-ranked pairs: {rv.height:,}; stale (stored rank != recomputed): {int(diff.sum()):,} ({diff.mean():.3f})")
print(f"  among TRUE pairs: {rv.filter(pl.col('y')==1)['rk_rev'].ne(rv.filter(pl.col('y')==1)['rk_new']).mean():.4f};  "
      f"among non-matches: {rv.filter(pl.col('y')==0)['rk_rev'].ne(rv.filter(pl.col('y')==0)['rk_new']).mean():.4f}")
x = rv.filter(pl.col("y") == 0)
print("  non-matches: share with stored rank 0:", round((x["rk_rev"] == 0).mean(), 4), "| recomputed rank 0:", round((x["rk_new"] == 0).mean(), 4))
# importance in the saved XGBoost models
import xgboost as xgb
def gain_share(path, names=None):
    b = xgb.Booster(); b.load_model(path)
    g = b.get_score(importance_type="total_gain")
    tot = sum(g.values())
    fn = b.feature_names
    items = sorted(((fn[int(k[1:])] if (fn is None or k.startswith("f")) and fn else k, v / tot) for k, v in g.items()), key=lambda t: -t[1])
    return items, fn
for tag, pth in [("stage-2 (exp15, fold0)", os.path.join(R, "exp15", "stage2_fold0.json"))] + \
                [(f"stage-1 ({os.path.basename(p)})", p) for p in sorted(glob.glob(os.path.join(R, "exp13", "*.json")))[:1]]:
    if not os.path.exists(pth):
        continue
    b = xgb.Booster(); b.load_model(pth)
    g = b.get_score(importance_type="total_gain"); tot = sum(g.values())
    fn = b.feature_names
    name = lambda k: (fn[int(k[1:])] if fn and k.startswith("f") and k[1:].isdigit() else k)
    items = sorted(((name(k), v / tot) for k, v in g.items()), key=lambda t: -t[1])
    rank = [i for i, (n, _) in enumerate(items) if n == "rk_rev"]
    print(f"{tag}: {len(items)} features used; rk_rev gain share "
          f"{dict(items).get('rk_rev', 0):.5f} (rank {rank[0] + 1 if rank else 'unused'}); top 5 {[(n, round(v, 3)) for n, v in items[:5]]}")
