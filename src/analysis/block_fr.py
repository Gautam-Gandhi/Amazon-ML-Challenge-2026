"""France blocking check with near-certain pairs found by construction (no labels needed):
 (a) acronym record: name = initials of an S1's name, same first house number, same street words
 (b) same name_core + same first house number + same street words (a clean copy with address noise)
Where along the funnel do these pairs survive? raw search (cand_v1) -> pruned (feat_v1) -> filtered (exp29gf)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, glob, re
import numpy as np, polars as pl
C = _os.path.join(_WORK, 'data', 'cache')
R = _os.path.join(_WORK, 'runs')
STOP = {"la", "le", "les", "de", "du", "des", "d", "l", "et", "and", "of", "the", "a", "en", "au", "aux"}
s1 = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["country", "name_core", "addr_clean", "addr_nums"]).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32))
r = pl.read_parquet(os.path.join(C, "prep_v2", "test_r.parquet"), columns=["country", "name_core", "addr_clean", "addr_nums"]).with_row_index("r").with_columns(pl.col("r").cast(pl.Int32))
def street(df):
    # first number + non-numeric address words (minus region codes / 1-letter tokens)
    return df.with_columns(pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("h"),
                           pl.col("addr_clean").str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"\d") & (pl.element().str.len_chars() > 1) & ~pl.element().str.starts_with("fr"))).alias("w"))
fs = street(s1.filter(pl.col("country") == "France")); fr = street(r.filter(pl.col("country") == "France"))
# (a) acronyms
ini = lambda core, drop: "".join(t[0] for t in core.split() if t and (not drop or t not in STOP))
fs = fs.with_columns(pl.col("name_core").map_elements(lambda c: ini(c, False), return_dtype=pl.Utf8).alias("i1"),
                     pl.col("name_core").map_elements(lambda c: ini(c, True), return_dtype=pl.Utf8).alias("i2"))
racro = fr.filter(pl.col("name_core").str.contains(r"^[a-z]{2,5}$") & (pl.col("h") != ""))
pa = pl.concat([racro.join(fs.select("s1", pl.col("i1").alias("name_core"), "h", pl.col("w").alias("w1")), on=["name_core", "h"]),
                racro.join(fs.select("s1", pl.col("i2").alias("name_core"), "h", pl.col("w").alias("w1")), on=["name_core", "h"])]).unique(subset=["s1", "r"])
# (b) same name_core + same number
pb = fr.filter(pl.col("h") != "").join(fs.select("s1", "name_core", "h", pl.col("w").alias("w1")), on=["name_core", "h"])
def same_street(d):
    return d.with_columns((pl.col("w").list.set_intersection("w1").list.len() / pl.col("w").list.set_union("w1").list.len().clip(1, None)).alias("j")).filter(pl.col("j") >= 0.5)
pa = same_street(pa); pb = same_street(pb)
# keep only pairs where that S1 is the ONLY such S1 for the record (unambiguous by construction)
pa = pa.filter(pl.len().over("r") == 1); pb = pb.filter(pl.len().over("r") == 1)
raw = pl.read_parquet(os.path.join(C, "cand_v1", "test", "France_fold0.parquet"), columns=["s1", "r", "rk_name", "rk_addr", "rk_joint", "rk_rev"]).with_columns(pl.lit(True).alias("in_raw"))
pr = pl.concat([pl.read_parquet(f, columns=["s1", "r"]) for f in sorted(glob.glob(os.path.join(C, "feat_v1", "test", "part*.parquet")))])
pr = pr.join(s1.select("s1", "country"), on="s1").filter(pl.col("country") == "France").select("s1", "r").with_columns(pl.lit(True).alias("in_pruned"))
fl = pl.read_parquet(os.path.join(R, "exp29gf", "test_pred.parquet"), columns=["s1", "r", "p"]).with_columns(pl.lit(True).alias("in_filtered"))
for tag, d in [("(a) acronym, same number+street", pa), ("(b) same name, same number+street", pb)]:
    d = d.select("s1", "r").join(raw, on=["s1", "r"], how="left").join(pr, on=["s1", "r"], how="left").join(fl, on=["s1", "r"], how="left").fill_null(False)
    n = d.height
    print(f"{tag}: {n:,} pairs | raw search {d['in_raw'].mean():.4f} | after pruning {d['in_pruned'].mean():.4f} | "
          f"after filter (candidate_pairs.tsv) {d['in_filtered'].mean():.4f} | accepted p>0.9 {(d['p'].fill_null(0) > 0.9).mean():.4f}")
    miss = d.filter(~pl.col("in_raw"))
    print(f"   lost at raw search: {miss.height:,}; lost at pruning: {d.filter(pl.col('in_raw') & ~pl.col('in_pruned')).height:,}; "
          f"lost at filter: {d.filter(pl.col('in_pruned') & ~pl.col('in_filtered')).height:,}")
    if tag.startswith("(a)"):
        dd = d.filter(pl.col("in_raw"))
        print("   channels that found them (share): name", round((dd["rk_name"] < 999).mean(), 3), "addr", round((dd["rk_addr"] < 999).mean(), 3),
              "joint", round((dd["rk_joint"] < 999).mean(), 3), "reverse", round((dd["rk_rev"] < 999).mean(), 3))
