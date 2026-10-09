"""Acronym records: R name_core is one short token equal to the initials of a candidate S1's name words.
Per country (test): count, p distribution, number relation, whether the acronym-matching S1 is the R's best S1.
Train (dense OOF): same with labels."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse, re
import numpy as np, polars as pl
sys.path.insert(0, _REPO)
os.environ["ER_WORK_DIR"] = _WORK
from er_common import RUNS, CACHE
import exp03_dense as M3
STOP = {"la", "le", "les", "de", "du", "des", "d", "l", "et", "and", "of", "the", "a", "en", "au", "aux"}


def initials(core, drop_stop):
    toks = [t for t in core.split() if t]
    if drop_stop:
        toks = [t for t in toks if t not in STOP]
    return "".join(t[0] for t in toks)


def acro_pairs(pred, split):
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    s1 = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=["country", "name_core", "addr_nums", "addr_clean"])
    r = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=["name_core", "addr_nums", "addr_clean"])
    rc = r["name_core"].to_numpy()
    short = np.array([(len(x) >= 2 and len(x) <= 5 and " " not in x and x.isalpha()) for x in rc])
    t = pred.filter(pl.Series(short[pred["r"].to_numpy()]))
    t = t.with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("rk"))
    si, ri = t["s1"].to_numpy(), t["r"].to_numpy()
    s1c = s1["name_core"].to_numpy()
    ini_a = np.array([initials(s1c[s], False) for s in si])
    ini_b = np.array([initials(s1c[s], True) for s in si])
    rn = rc[ri]
    m = (ini_a == rn) | (ini_b == rn)
    t = t.with_columns(pl.Series("acro", m)).filter(pl.col("acro"))
    si, ri = t["s1"].to_numpy(), t["r"].to_numpy()
    h1 = s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null("")
    h2 = r["addr_nums"].gather(ri).str.split(" ").list.first().fill_null("")
    noaddr = r["addr_clean"].gather(ri) == ""
    t = t.with_columns(pl.Series("country", s1["country"].to_numpy()[si]), h1.alias("h1"), h2.alias("h2"), noaddr.alias("noaddr"))
    t = t.with_columns(pl.when(pl.col("noaddr")).then(pl.lit("noaddr")).when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("nonum"))
                       .when(pl.col("h1") == pl.col("h2")).then(pl.lit("samenum")).otherwise(pl.lit("diffnum")).alias("num"))
    # how many acronym-matching S1 candidates does the R have?
    t = t.with_columns(pl.len().over("r").alias("n_acro_s1"))
    return t


keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32), pl.lit(1).alias("y"))
oof = pl.read_parquet(os.path.join(RUNS, "exp29g", "oof_combined.parquet"))
tr = acro_pairs(oof, "train").join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
print("TRAIN acronym pairs (candidate pairs whose S1 initials == R name):", tr.height, "per 1K kept S1", 1000 * tr.height / keep.sum())
g = tr.group_by(["num", "n_acro_s1"]).agg(pl.len().alias("n"), pl.col("y").mean().alias("true"), pl.col("p").mean().alias("mp"), (pl.col("p") > 0.7).mean().alias("acc")).sort(["num", "n_acro_s1"])
print(g)
bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
x = tr.filter(pl.col("num") == "samenum")
print("train samenum acronym true rate by p bin:")
for lo, hi in zip(bins[:-1], bins[1:]):
    s = x.filter((pl.col("p") > lo) & (pl.col("p") <= hi))
    print(f"  p ({lo},{hi}] n={s.height} true={s['y'].mean() if s.height else float('nan'):.3f}")
te = acro_pairs(pl.read_parquet(os.path.join(RUNS, "exp29gf", "test_pred.parquet")), "test")
s1te = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].value_counts()
nS1 = dict(zip(s1te["country"].to_list(), s1te["count"].to_list()))
for c in ["US", "India", "France"]:
    thr = 0.9 if c == "France" else 0.7
    d = te.filter(pl.col("country") == c)
    print(f"TEST {c}: acronym pairs {d.height} ({1000*d.height/nS1[c]:.1f}/1K)")
    g = d.group_by(["num", "n_acro_s1"]).agg(pl.len().alias("n"), pl.col("p").mean().alias("mp"), (pl.col("p") > thr).mean().alias("acc"), (pl.col("rk") == 1).mean().alias("isbest")).sort(["num", "n_acro_s1"])
    print(g)
    x = d.filter((pl.col("num") == "samenum") & (pl.col("n_acro_s1") == 1))
    h = np.histogram(x["p"].to_numpy(), bins=bins)[0]
    print("   samenum & unique acronym S1: p hist", dict(zip([f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])], h.tolist())))
