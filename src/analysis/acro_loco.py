"""Acronym pairs in LOCO predictions (out-of-country model): true rate by p bin and by number relation."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys
import numpy as np, polars as pl
CACHE = _os.path.join(_WORK, 'data', 'cache')
RUNS = _os.path.join(_WORK, 'runs')
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
    short = np.array([(2 <= len(x) <= 5 and " " not in x and x.isalpha()) for x in rc])
    t = pred.filter(pl.Series(short[pred["r"].to_numpy()]))
    t = t.with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("rk"))
    si, ri = t["s1"].to_numpy(), t["r"].to_numpy()
    s1c = s1["name_core"].to_numpy()
    ini_a = np.array([initials(s1c[s], False) for s in si])
    ini_b = np.array([initials(s1c[s], True) for s in si])
    rn = rc[ri]
    t = t.with_columns(pl.Series("acro", (ini_a == rn) | (ini_b == rn))).filter(pl.col("acro"))
    si, ri = t["s1"].to_numpy(), t["r"].to_numpy()
    h1 = s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null("")
    h2 = r["addr_nums"].gather(ri).str.split(" ").list.first().fill_null("")
    noaddr = r["addr_clean"].gather(ri) == ""
    t = t.with_columns(pl.Series("country", s1["country"].to_numpy()[si]), h1.alias("h1"), h2.alias("h2"), noaddr.alias("noaddr"))
    t = t.with_columns(pl.when(pl.col("noaddr")).then(pl.lit("noaddr")).when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("nonum"))
                       .when(pl.col("h1") == pl.col("h2")).then(pl.lit("samenum")).otherwise(pl.lit("diffnum")).alias("num"))
    return t.with_columns(pl.len().over("r").alias("n_acro_s1"))


if __name__ == "__main__":
    bins = [0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01]
    for f in ["base_US_India", "base_India_US"]:
        pred = pl.read_parquet(os.path.join(RUNS, "loco_pred13", f + ".parquet"))
        t = acro_pairs(pred.select("s1", "r", "p", "y"), "train")
        print(f, "acronym pairs", t.height)
        print(t.group_by(["num", "n_acro_s1"]).agg(pl.len().alias("n"), pl.col("y").mean().alias("true"), pl.col("p").mean().alias("mp"),
                                                   (pl.col("p") > 0.9).mean().alias("acc90"), (pl.col("rk") == 1).mean().alias("isbest")).sort(["num", "n_acro_s1"]))
        x = t.filter((pl.col("num") == "samenum") & (pl.col("n_acro_s1") == 1) & (pl.col("rk") == 1))
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = x.filter((pl.col("p") > lo) & (pl.col("p") <= hi))
            if s.height:
                print(f"   samenum unique best: p ({lo},{hi}] n={s.height} true={s['y'].mean():.3f}")
