"""Noise-type census: per 1K S1, how many accepted pairs of each (name relation x house-number relation) type,
for train truth (dense world, kept S1), US / India / France test accepted (exp29gf decode)."""
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


def classify(pairs, split):
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    cols = ["country", "name_core", "name_raw", "addr_clean", "addr_nums", "name_sfx"]
    s1 = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=cols)
    r = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=cols[1:])
    si, ri = pairs["s1"].to_numpy(), pairs["r"].to_numpy()
    A = s1["name_core"].gather(si).str.split(" ")
    B = r["name_core"].gather(ri).str.split(" ")
    rn = r["name_raw"].gather(ri).str.to_lowercase()
    init = A.list.eval(pl.element().str.slice(0, 1)).list.join("")
    t = pl.DataFrame({"country": s1["country"].to_numpy()[si], "A": A, "B": B, "rn": rn, "init": init,
                      "h1": s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null(""),
                      "h2": r["addr_nums"].gather(ri).str.split(" ").list.first().fill_null(""),
                      "r_noaddr": (r["addr_clean"].gather(ri) == "")})
    t = t.with_columns(pl.col("A").list.unique().alias("As"), pl.col("B").list.unique().alias("Bs"))
    t = t.with_columns(pl.col("Bs").list.set_difference("As").list.len().alias("ne"),
                       pl.col("As").list.set_difference("Bs").list.len().alias("nm"),
                       pl.col("Bs").list.set_intersection("As").list.len().alias("nc"))
    name = (pl.when(pl.col("rn").str.contains(r"\b(dba|d\.b\.a|formerly|f/k/a|fka|aka|a/k/a|t/a)\b")).then(pl.lit("alias"))
            .when(pl.col("rn").str.contains(r"\.(com|net|org|in|fr|co)\b|^#|^@|www")).then(pl.lit("domain/tag"))
            .when(pl.col("A") == pl.col("B")).then(pl.lit("identical"))
            .when((pl.col("ne") == 0) & (pl.col("nm") == 0)).then(pl.lit("reorder"))
            .when((pl.col("B").list.len() == 1) & (pl.col("B").list.first() == pl.col("init")) & (pl.col("init").str.len_chars() >= 2)).then(pl.lit("acronym"))
            .when(pl.col("nc") == 0).then(pl.lit("brand/nooverlap"))
            .when((pl.col("ne") == 0) & (pl.col("nm") > 0)).then(pl.lit("drop"))
            .when((pl.col("ne") > 0) & (pl.col("nm") == 0)).then(pl.lit("add"))
            .when((pl.col("ne") == 1) & (pl.col("nm") == 1)).then(pl.lit("swap1"))
            .otherwise(pl.lit("multi")))
    num = (pl.when(pl.col("r_noaddr")).then(pl.lit("noaddr"))
           .when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("nonum"))
           .when(pl.col("h1") == pl.col("h2")).then(pl.lit("samenum")).otherwise(pl.lit("diffnum")))
    return t.with_columns(name.alias("name"), num.alias("num")).select("country", "name", "num")


keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"})
s1tr = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_s1.parquet"), columns=["country"])["country"].to_numpy()
ctr = classify(gt, "train")
n_tr = {c: int((keep & (s1tr == c)).sum()) for c in ["US", "India"]}
te = pl.read_parquet(os.path.join(RUNS, "exp29gf", "test_pred.parquet"))
a = te.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
s1te = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country"])["country"].to_numpy()
thr = np.where(s1te[a["s1"].to_numpy()] == "France", 0.9, 0.7)
acc = a.filter(pl.Series(a["p"].to_numpy() > thr))
cte = classify(acc, "test")
n_te = {c: int((s1te == c).sum()) for c in ["US", "India", "France"]}
rows = []
for tag, df, nn in [("trainGT_US", ctr.filter(pl.col("country") == "US"), n_tr["US"]),
                    ("trainGT_IN", ctr.filter(pl.col("country") == "India"), n_tr["India"]),
                    ("test_US", cte.filter(pl.col("country") == "US"), n_te["US"]),
                    ("test_IN", cte.filter(pl.col("country") == "India"), n_te["India"]),
                    ("test_FR", cte.filter(pl.col("country") == "France"), n_te["France"])]:
    g = df.group_by("name").len().with_columns((pl.col("len") * 1000 / nn).alias(tag)).select("name", tag)
    rows.append(g)
out = rows[0]
for g in rows[1:]:
    out = out.join(g, on="name", how="full", coalesce=True)
print("per 1K S1 by NAME relation")
print(out.sort("trainGT_US", descending=True))
rows = []
for tag, df, nn in [("trainGT_US", ctr.filter(pl.col("country") == "US"), n_tr["US"]),
                    ("trainGT_IN", ctr.filter(pl.col("country") == "India"), n_tr["India"]),
                    ("test_US", cte.filter(pl.col("country") == "US"), n_te["US"]),
                    ("test_IN", cte.filter(pl.col("country") == "India"), n_te["India"]),
                    ("test_FR", cte.filter(pl.col("country") == "France"), n_te["France"])]:
    g = df.group_by("num").len().with_columns((pl.col("len") * 1000 / nn).alias(tag)).select("num", tag)
    rows.append(g)
out = rows[0]
for g in rows[1:]:
    out = out.join(g, on="num", how="full", coalesce=True)
print("per 1K S1 by NUMBER relation")
print(out)
rows = []
for tag, df, nn in [("trainGT_US", ctr.filter(pl.col("country") == "US"), n_tr["US"]),
                    ("test_US", cte.filter(pl.col("country") == "US"), n_te["US"]),
                    ("test_FR", cte.filter(pl.col("country") == "France"), n_te["France"])]:
    g = df.group_by(["name", "num"]).len().with_columns((pl.col("len") * 1000 / nn).alias(tag)).select("name", "num", tag)
    rows.append(g)
out = rows[0]
for g in rows[1:]:
    out = out.join(g, on=["name", "num"], how="full", coalesce=True)
pl.Config.set_tbl_rows(60)
print(out.sort("trainGT_US", descending=True).head(45))
