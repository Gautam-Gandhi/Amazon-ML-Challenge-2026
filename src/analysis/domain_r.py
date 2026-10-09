"""All R records whose raw name is a domain/tag: per country, how many, accepted?, and for their best candidate:
number relation and p. Also: do non-accepted French domain records have a same-name S1 with a different number?"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys
import numpy as np, polars as pl
W = _WORK
s1 = pl.read_parquet(W + r"\data\cache\prep_v2\test_s1.parquet", columns=["country", "name_raw", "addr_raw", "name_core", "addr_nums", "addr_clean"])
r = pl.read_parquet(W + r"\data\cache\prep_v2\test_r.parquet", columns=["country", "name_raw", "addr_raw", "name_core", "addr_nums", "addr_clean"]).with_row_index("r")
pred = pl.read_parquet(W + r"\runs\exp29gf\test_pred.parquet")
a = pred.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
dom = r.filter(pl.col("name_raw").str.to_lowercase().str.contains(r"\.(com|net|org|in|fr|co)\b|www"))
n_s1 = dict(zip(*[s1["country"].value_counts()[c].to_list() for c in ["country", "count"]]))
dom = dom.join(a.select(pl.col("r").cast(pl.UInt32), "s1", "p"), on="r", how="left")
si = dom["s1"].fill_null(0).to_numpy()
dom = dom.with_columns(
    pl.Series("h1", s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null("")),
    pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("h2"))
dom = dom.with_columns(
    pl.when(pl.col("s1").is_null()).then(pl.lit("nocand"))
    .when(pl.col("addr_clean") == "").then(pl.lit("noaddr"))
    .when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("nonum"))
    .when(pl.col("h1") == pl.col("h2")).then(pl.lit("samenum")).otherwise(pl.lit("diffnum")).alias("num"))
for c in ["US", "India", "France"]:
    thr = 0.9 if c == "France" else 0.7
    d = dom.filter(pl.col("country") == c)
    print(f"{c}: domain R per 1K S1 = {1000*d.height/n_s1[c]:.1f}; accepted {1000*(d['p'].fill_null(0) > thr).sum()/n_s1[c]:.1f}")
    g = d.group_by("num").agg(pl.len().alias("n"), (pl.col("p").fill_null(0) > thr).mean().alias("acc"), pl.col("p").mean().alias("mp"))
    for row in g.sort("num").iter_rows(named=True):
        print(f"     {row['num']:8s} per1K {1000*row['n']/n_s1[c]:7.1f} acc {row['acc']:.3f} mp {row['mp'] if row['mp'] is not None else float('nan'):.3f}")
# samples: French domain R with diffnum or nocand
fr = dom.filter((pl.col("country") == "France") & pl.col("num").is_in(["diffnum", "nocand"]))
print("\nFrench domain R, diffnum/nocand samples:")
for row in fr.sample(min(25, fr.height), seed=2).iter_rows(named=True):
    best = f"best S1: {s1['name_raw'][row['s1']]} | {s1['addr_raw'][row['s1']]} (p={row['p']:.2f})" if row["s1"] is not None else "no candidate"
    print(f"  R: {row['name_raw']} | {row['addr_raw']}\n       {best}")
us = dom.filter((pl.col("country") == "US") & pl.col("num").is_in(["diffnum"]))
print("\nUS domain R, diffnum samples:")
for row in us.sample(min(10, us.height), seed=2).iter_rows(named=True):
    print(f"  R: {row['name_raw']} | {row['addr_raw']}\n       best S1: {s1['name_raw'][row['s1']]} | {s1['addr_raw'][row['s1']]} (p={row['p']:.2f})")
