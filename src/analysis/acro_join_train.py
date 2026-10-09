"""Join-based acronym channel on labeled data (dense world, kept S1): record name = initials of an S1 name, same first
house number, street words Jaccard >= 0.5, exactly one such S1 for the record. Precision + funnel (raw/pruned/filtered)."""
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
STOP = {"la", "le", "les", "de", "du", "des", "d", "l", "et", "and", "of", "the", "a", "en", "au", "aux"}
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32), pl.lit(1).alias("y"))
s1 = pl.read_parquet(os.path.join(C, "prep_v1", "train_s1.parquet"), columns=["country", "name_core", "addr_clean", "addr_nums"]).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32)).filter(pl.Series(keep))
r = pl.read_parquet(os.path.join(C, "prep_v1", "train_r.parquet"), columns=["country", "name_core", "addr_clean", "addr_nums"]).with_row_index("r").with_columns(pl.col("r").cast(pl.Int32))
def street(df):
    return df.with_columns(pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("h"),
                           pl.col("addr_clean").str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"\d") & (pl.element().str.len_chars() > 1))).alias("w"))
s1 = street(s1); r = street(r)
ini = lambda core, drop: "".join(t[0] for t in core.split() if t and (not drop or t not in STOP))
s1 = s1.with_columns(pl.col("name_core").map_elements(lambda c: ini(c, False), return_dtype=pl.Utf8).alias("i1"),
                     pl.col("name_core").map_elements(lambda c: ini(c, True), return_dtype=pl.Utf8).alias("i2"))
ra = r.filter(pl.col("name_core").str.contains(r"^[a-z]{2,5}$") & (pl.col("h") != ""))
pa = pl.concat([ra.join(s1.select("s1", "country", pl.col("i1").alias("name_core"), "h", pl.col("w").alias("w1")), on=["country", "name_core", "h"]),
                ra.join(s1.select("s1", "country", pl.col("i2").alias("name_core"), "h", pl.col("w").alias("w1")), on=["country", "name_core", "h"])]).unique(subset=["s1", "r"])
pa = pa.with_columns((pl.col("w").list.set_intersection("w1").list.len() / pl.col("w").list.set_union("w1").list.len().clip(1, None)).alias("j")).filter(pl.col("j") >= 0.5)
pa = pa.filter(pl.len().over("r") == 1).select("s1", "r", "country")
raw = pl.concat([pl.read_parquet(f, columns=["s1", "r"]) for f in sorted(glob.glob(os.path.join(C, "cand_v1", "train", "*.parquet")))]).join(pa.select("s1", "r"), on=["s1", "r"], how="semi").with_columns(pl.lit(True).alias("in_raw"))
pr = pl.concat([pl.read_parquet(f, columns=["s1", "r"]) for f in sorted(glob.glob(os.path.join(C, "feat_v3", "k80s0", "part*.parquet")))]).join(pa.select("s1", "r"), on=["s1", "r"], how="semi").with_columns(pl.lit(True).alias("in_pruned"))
fl = pl.read_parquet(os.path.join(R, "exp29g", "oof_combined.parquet")).join(pa.select("s1", "r"), on=["s1", "r"], how="semi").with_columns(pl.lit(True).alias("in_filt"))
d = pa.join(raw.unique(), on=["s1", "r"], how="left").join(pr, on=["s1", "r"], how="left").join(fl, on=["s1", "r"], how="left") \
      .join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0), pl.col("in_raw").fill_null(False), pl.col("in_pruned").fill_null(False), pl.col("in_filt").fill_null(False))
for c in ["US", "India"]:
    x = d.filter(pl.col("country") == c)
    lost = x.filter(~pl.col("in_filt"))
    print(f"{c}: acronym-join pairs {x.height:,}, TRUE rate {x['y'].mean():.4f} | in raw {x['in_raw'].mean():.3f}, pruned {x['in_pruned'].mean():.3f}, "
          f"filtered {x['in_filt'].mean():.3f} | not reaching the matcher: {lost.height:,} (true rate {lost['y'].mean() if lost.height else float('nan'):.4f}) | "
          f"accepted by the pipeline (p>0.7): {(x['p'].fill_null(0) > 0.7).mean():.3f}")
