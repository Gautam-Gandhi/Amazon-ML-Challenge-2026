"""Blocking recall on train (labels), per stage, per country, by address presence; anatomy of the misses;
and what an exact-name channel for address-less records would add (recall vs candidate cost)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, glob, argparse
import numpy as np, polars as pl
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
import exp03_dense as M3
C = _os.path.join(_WORK, 'data', 'cache')
R = _os.path.join(_WORK, 'runs')
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
s1 = pl.read_parquet(os.path.join(C, "prep_v1", "train_s1.parquet"), columns=["country", "name_core"]).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32))
r = pl.read_parquet(os.path.join(C, "prep_v1", "train_r.parquet"), columns=["country", "name_core", "addr_clean"]).with_row_index("r").with_columns(pl.col("r").cast(pl.Int32))
noaddr = (r["addr_clean"] == "").to_numpy()
gt_all = pl.read_parquet(os.path.join(C, "prep_v1", "train_gt.parquet")).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32))
gt_all = gt_all.with_columns(pl.Series("country", s1["country"].to_numpy()[gt_all["s1"].to_numpy()]), pl.Series("noaddr", noaddr[gt_all["r"].to_numpy()]))
gt_d = gt_all.filter(pl.Series(keep[gt_all["s1"].to_numpy()]))
gkeys = gt_d.select("s1", "r")
raw = pl.concat([pl.read_parquet(f, columns=["s1", "r"]).join(gkeys, on=["s1", "r"], how="semi") for f in sorted(glob.glob(os.path.join(C, "cand_v1", "train", "*.parquet")))]).unique().with_columns(pl.lit(True).alias("in_raw"))
prn = pl.concat([pl.read_parquet(f, columns=["s1", "r"]).join(gkeys, on=["s1", "r"], how="semi") for f in sorted(glob.glob(os.path.join(C, "feat_v3", "k80s0", "part*.parquet")))]).with_columns(pl.lit(True).alias("in_prn"))
fil = pl.read_parquet(os.path.join(R, "exp29g", "oof_combined.parquet"), columns=["s1", "r"]).with_columns(pl.lit(True).alias("in_fil"))
g = gt_d.join(raw, on=["s1", "r"], how="left").join(prn, on=["s1", "r"], how="left").join(fil, on=["s1", "r"], how="left").fill_null(False)
# name sharing among kept S1 (dense world) for the TRUE S1
cnt = s1.filter(pl.Series(keep)).group_by(["country", "name_core"]).len("nsame")
g = g.join(s1.select("s1", "name_core"), on="s1").join(cnt, on=["country", "name_core"], how="left")
print("TRAIN recall by stage (dense world = kept S1; the training/validation world of every model)")
for c in ["US", "India", None]:
    x = g if c is None else g.filter(pl.col("country") == c)
    print(f"  {c or 'ALL':6s} true pairs {x.height:>9,} | search union (~73/S1) {x['in_raw'].mean():.4f} | pruned (~19/S1) {x['in_prn'].mean():.4f} | "
          f"final candidates (~3.7/S1) {x['in_fil'].mean():.4f}")
for na in [False, True]:
    x = g.filter(pl.col("noaddr") == na)
    print(f"  record {'WITHOUT' if na else 'WITH   '} address: share of true pairs {x.height/g.height:.3f} | search {x['in_raw'].mean():.4f} | pruned {x['in_prn'].mean():.4f} | final {x['in_fil'].mean():.4f}")
m = g.filter(~pl.col("in_fil"))
cat = pl.when(pl.col("noaddr") & (pl.col("nsame") >= 2)).then(pl.lit("no address, name shared by >=2 S1")) \
        .when(pl.col("noaddr")).then(pl.lit("no address, unique S1 name")).otherwise(pl.lit("with address"))
mm = m.with_columns(cat.alias("cat")).group_by("cat").agg(pl.len().alias("n")).with_columns((pl.col("n") / g.height).alias("share_of_all_true")).sort("n", descending=True)
print(f"MISSED true pairs (not in final candidates): {m.height:,} = {m.height/g.height:.4f} of all")
print(mm)
# exact name channel for address-less records: all kept S1 with the same name_core (same country)
ra = r.filter(pl.Series(noaddr)).select("r", "country", "name_core")
s1k = s1.filter(pl.Series(keep)).select("s1", "country", "name_core")
ch = ra.join(s1k, on=["country", "name_core"])
new = ch.select("s1", "r").join(fil.select("s1", "r"), on=["s1", "r"], how="anti")
rec = new.join(gt_d.select("s1", "r"), on=["s1", "r"], how="semi").height
n_kept = int(keep.sum())
fin = g["in_fil"].sum()
print(f"EXACT-NAME channel for address-less records: +{new.height:,} pairs ({new.height/n_kept:.2f} per S1; now {fil.height/n_kept:.2f}) "
      f"-> recovers {rec:,} true pairs; recall {fin/g.height:.4f} -> {(fin + rec)/g.height:.4f}; precision of added pairs {rec/max(new.height,1):.4f}")
for cap in [5, 10, 20, 50]:
    sz = ch.group_by("r").len("k")
    ch2 = ch.join(sz, on="r").filter(pl.col("k") <= cap)
    new2 = ch2.select("s1", "r").join(fil.select("s1", "r"), on=["s1", "r"], how="anti")
    rec2 = new2.join(gt_d.select("s1", "r"), on=["s1", "r"], how="semi").height
    print(f"   only names shared by <= {cap:2d} S1: +{new2.height:,} pairs ({new2.height/n_kept:.2f}/S1), recovers {rec2:,} -> recall {(fin + rec2)/g.height:.4f}")
