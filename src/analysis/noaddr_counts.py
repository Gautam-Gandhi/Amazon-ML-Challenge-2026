"""Address-less R whose top-2 candidate S1s share name_core: does the ratio of their accepted with-address record counts
predict the owner beyond the model's p? (dense OOF, labels)"""
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
from er_common import CACHE, RUNS
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32), pl.lit(1).alias("y"))
oof = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
s1 = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["name_core"])
r = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_r.parquet"), columns=["addr_clean"])
noaddr = (r["addr_clean"] == "").to_numpy()
# accepted with-address records per S1 (assigned, p > 0.9)
a = oof.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
acc = a.filter((pl.col("p") > 0.9) & pl.Series(~noaddr[a["r"].to_numpy()]))
cnt = np.bincount(acc["s1"].to_numpy(), minlength=len(keep))
t = oof.filter(pl.Series(noaddr[oof["r"].to_numpy()])).with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("rk"))
b1 = t.filter(pl.col("rk") == 1).select("r", pl.col("s1").alias("sA"), pl.col("p").alias("pA"))
b2 = t.filter(pl.col("rk") == 2).select("r", pl.col("s1").alias("sB"), pl.col("p").alias("pB"))
x = b1.join(b2, on="r")
nc = s1["name_core"].to_numpy()
x = x.filter(pl.Series(nc[x["sA"].to_numpy()] == nc[x["sB"].to_numpy()]))
x = x.with_columns(pl.Series("nA", cnt[x["sA"].to_numpy()]), pl.Series("nB", cnt[x["sB"].to_numpy()]))
x = x.join(gt.select(pl.col("s1").alias("sA"), "r", pl.col("y").alias("yA")), on=["sA", "r"], how="left") \
     .join(gt.select(pl.col("s1").alias("sB"), "r", pl.col("y").alias("yB")), on=["sB", "r"], how="left").fill_null(0)
print("address-less R, top-2 S1 share the name:", x.height, " A true", x["yA"].mean(), " B true", x["yB"].mean())
x = x.with_columns(pl.when(pl.col("nA") > 2 * pl.col("nB") + 1).then(pl.lit("A>>B")).when(pl.col("nB") > 2 * pl.col("nA") + 1).then(pl.lit("B>>A")).otherwise(pl.lit("similar")).alias("cmp"),
                   pl.when(pl.col("pA") <= 0.3).then(pl.lit("a<.3")).when(pl.col("pA") <= 0.5).then(pl.lit("b<.5")).when(pl.col("pA") <= 0.7).then(pl.lit("c<.7")).when(pl.col("pA") <= 0.9).then(pl.lit("d<.9")).otherwise(pl.lit("e>.9")).alias("bin"))
g = x.group_by(["cmp", "bin"]).agg(pl.len().alias("n"), pl.col("yA").mean().alias("A_true"), pl.col("yB").mean().alias("B_true")).sort(["cmp", "bin"])
pl.Config.set_tbl_rows(30)
print(g)
