"""French transformation-type scan (label-free on France, labels out-of-country).
Each record's best pair (assign) is typed by name transformation x house-number relation.
LOCO (exp13 stage 1 trained on one country, scoring the other; dense world, 35% eval S1) gives the true rate per
type and p bin out of country. France (exp35 final p) gives the volume per type and p bin, and the current decision.
Flags: types whose out-of-country true rate is >= 0.85 in a band France rejects, or <= 0.6 in a band France accepts."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
import exp03_dense as M3
C = _os.path.join(_WORK, 'data', 'cache')
R = _os.path.join(_WORK, 'runs')
SP = os.path.dirname(os.path.abspath(__file__))
COLS = ["name_core", "name_sfx", "addr_nums", "addr_clean", "country"]


def abbrev(s, l):
    if len(s) < 1 or len(s) >= len(l) or s[0] != l[0]:
        return False
    it = iter(l)
    return all(ch in it for ch in s)


def ntype(a, b):
    """a = S1 name_core, b = record name_core."""
    if a == b:
        return "identical"
    ta, tb = a.split(), b.split()
    if not tb or not ta:
        return "empty"
    if sorted(ta) == sorted(tb):
        return "reorder"
    if a.replace(" ", "") == b.replace(" ", ""):
        return "spacing"
    if len(tb) == 1 and len(ta) >= 2 and 2 <= len(tb[0]) <= 6 and tb[0] == "".join(t[0] for t in ta if t):
        return "acronym"
    sa, sb = set(ta), set(tb)
    M, E = sa - sb, sb - sa
    if not M and E:
        return f"append{min(len(E), 2)}"
    if M and not E:
        return f"delete{min(len(M), 2)}"
    if len(M) == 1 and len(E) == 1:
        m, e = next(iter(M)), next(iter(E))
        if abbrev(e, m) or abbrev(m, e):
            return "abbrev"
        if Levenshtein.distance(m, e) <= 2:
            return "typo"
        return "swap1"
    if len(M) == len(E) and all(any(abbrev(e, m) or abbrev(m, e) for m in M) for e in E):
        return "abbrev_multi"
    if len(M) <= 2 and len(E) <= 2:
        return "swap2"
    return "other"


def typed(pairs, s1, r):
    si, ri = pairs["s1"].to_numpy(), pairs["r"].to_numpy()
    A, B = s1[si], r[ri]
    nt = [ntype(a, b) for a, b in zip(A["name_core"].to_list(), B["name_core"].to_list())]
    h1 = A["addr_nums"].str.split(" ").list.first().fill_null("").to_numpy()
    h2 = B["addr_nums"].str.split(" ").list.first().fill_null("").to_numpy()
    empty = (B["addr_clean"] == "").to_numpy()
    num = np.where(empty, "no_addr", np.where(h2 == "", "no_num", np.where(h1 == h2, "same_num", "diff_num")))
    sfa, sfb = A["name_sfx"].to_numpy(), B["name_sfx"].to_numpy()
    sfx = np.where(sfa == sfb, "sfx=", np.where((sfa == "") | (sfb == ""), "sfx1", "sfx!="))
    return pairs.with_columns(pl.Series("ntype", nt), pl.Series("num", num), pl.Series("sfx", sfx))


def best(d):
    return d.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")


BINS = [0.05, 0.3, 0.7, 0.9, 1.01]
LAB = ["0.05-0.3", "0.3-0.7", "0.7-0.9", "0.9-1"]
def binp(d):
    return d.filter(pl.col("p") > 0.05).with_columns(pl.col("p").cut(BINS[1:-1], labels=LAB).alias("bin"))

# ---- LOCO (labels)
s1tr = pl.read_parquet(os.path.join(C, "prep_v1", "train_s1.parquet"), columns=COLS)
rtr = pl.read_parquet(os.path.join(C, "prep_v1", "train_r.parquet"), columns=COLS)
lo = []
for f in ["base_US_India", "base_India_US"]:
    d = pl.read_parquet(os.path.join(R, "loco_pred13", f + ".parquet"))
    lo.append(typed(binp(best(d)), s1tr, rtr).with_columns(pl.lit(f).alias("dir")))
lo = pl.concat(lo)
print("LOCO best pairs typed:", lo.height, flush=True)
# ---- France (exp35 final p), rule-touched flag
s1te = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=COLS)
rte = pl.read_parquet(os.path.join(C, "prep_v2", "test_r.parquet"), columns=COLS)
fin = pl.read_parquet(os.path.join(R, "exp35", "test_pred.parquet"))
s2 = pl.read_parquet(os.path.join(R, "exp15", "test_pred.parquet")).rename({"p": "p2"})
ctry = s1te["country"].to_numpy()
fr = fin.filter(pl.Series(ctry[fin["s1"].to_numpy()] == "France"))
fr = best(fr).join(s2, on=["s1", "r"], how="left").with_columns((pl.col("p2").is_null() | ((pl.col("p") - pl.col("p2")).abs() > 1e-6)).alias("touched"))
fr = typed(binp(fr), s1te, rte)
print("France best pairs typed:", fr.height, flush=True)
lo.write_parquet(os.path.join(SP, "scan_loco.parquet")); fr.write_parquet(os.path.join(SP, "scan_fr.parquet"))

key = ["ntype", "num"]
lt = lo.group_by(key + ["bin"]).agg(pl.len().alias("loco_n"), pl.col("y").mean().alias("loco_true"),
                                    (pl.col("dir") == "base_US_India").sum().alias("n_ui"),
                                    pl.col("y").filter(pl.col("dir") == "base_US_India").mean().alias("true_ui"),
                                    pl.col("y").filter(pl.col("dir") == "base_India_US").mean().alias("true_iu"))
ft = fr.group_by(key + ["bin"]).agg(pl.len().alias("fr_n"), pl.col("touched").sum().alias("fr_touched"))
t = ft.join(lt, on=key + ["bin"], how="left").with_columns(pl.col("bin").cast(pl.Utf8))
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(220); pl.Config.set_float_precision(3)
# France accepts at p > 0.9 (final decode). Rejected bands: <= 0.9. Candidate flips: rejected bins with high OOC true rate.
up = t.filter(pl.col("bin").is_in(["0.3-0.7", "0.7-0.9"]) & (pl.col("true_ui") >= 0.85) & (pl.col("true_iu") >= 0.85) & (pl.col("loco_n") >= 100) & (pl.col("fr_n") >= 200))
down = t.filter((pl.col("bin") == "0.9-1") & ((pl.col("true_ui") <= 0.65) | (pl.col("true_iu") <= 0.65)) & (pl.col("loco_n") >= 100) & (pl.col("fr_n") >= 200))
print("\n=== FLIP-UP candidates: France rejects (p <= 0.9), out-of-country true rate >= 0.85 in BOTH directions ===")
print(up.sort("fr_n", descending=True))
print("\n=== DEMOTE candidates: France accepts (p > 0.9), out-of-country true rate <= 0.65 in a direction ===")
print(down.sort("fr_n", descending=True))
print("\n=== all types, France volume in 0.3-0.9 (rejected band), with OOC truth ===")
print(t.filter(pl.col("bin").is_in(["0.3-0.7", "0.7-0.9"])).sort("fr_n", descending=True).head(40))
