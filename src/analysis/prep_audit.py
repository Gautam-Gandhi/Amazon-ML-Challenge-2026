"""Audit of prep_v3 normalization: measure suspected issues on real data."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, re, sys
import numpy as np, polars as pl
SP = os.path.dirname(os.path.abspath(__file__))
C = _os.path.join(_WORK, 'data', 'cache')
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(220)

te_r = pl.read_parquet(os.path.join(C, "prep_v2", "test_r.parquet"), columns=["country", "name_raw", "name_core", "name_sfx", "addr_raw", "addr_clean", "addr_nums"])
te_s = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["country", "name_raw", "name_core", "name_sfx", "addr_raw", "addr_clean", "addr_nums"])
tr_r = pl.read_parquet(os.path.join(C, "prep_v1", "train_r.parquet"), columns=["country", "name_raw", "name_core", "addr_raw", "addr_nums"])
tr_s = pl.read_parquet(os.path.join(C, "prep_v1", "train_s1.parquet"), columns=["country", "name_raw", "name_core", "addr_raw", "addr_nums"])

print("=== 1. 'house number' = FIRST number in the address string")
def numcheck(pairs, S, R, label):
    si, ri = pairs["s1"].to_numpy(), pairs["r"].to_numpy()
    A = S["addr_nums"].gather(si).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    B = R["addr_nums"].gather(ri).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    d = pl.DataFrame({"a": A, "b": B}).with_columns(
        (pl.col("a").list.len() > 0).alias("ha"), (pl.col("b").list.len() > 0).alias("hb"),
        (pl.col("a").list.first() == pl.col("b").list.first()).fill_null(False).alias("first_eq"),
        (pl.col("a").list.set_intersection("b").list.len() > 0).alias("inter"))
    d = d.filter(pl.col("ha") & pl.col("hb"))
    n = d.height
    fe = d["first_eq"].mean(); hid = (~d["first_eq"] & d["inter"]).mean()
    print(f"  {label}: pairs with numbers on both sides {n:,}; first numbers equal {fe:.3f}; "
          f"first differs BUT a number is shared {hid:.3f}")
gt = pl.read_parquet(os.path.join(C, "prep_v1", "train_gt.parquet")).rename({"s1_idx": "s1", "r_idx": "r"})
numcheck(gt.sample(500000, seed=0), tr_s, tr_r, "train TRUE pairs (US+India)")
te = pl.read_parquet(os.path.join(SP, "test_typed.parquet")).select("s1", "r", "p", "country")
for c in ["US", "India", "France"]:
    numcheck(te.filter((pl.col("country") == c) & (pl.col("p") > 0.9)), te_s, te_r, f"test {c} accepted (p>0.9)")
    numcheck(te.filter((pl.col("country") == c) & (pl.col("p") > 0.05) & (pl.col("p") <= 0.9)), te_s, te_r, f"test {c} uncertain (0.05,0.9]")

print("\n=== 2. French 'Cie' (legal suffix) vs 'Compagnie' (plain word)")
fr_s = te_s.filter(pl.col("country") == "France"); fr_r = te_r.filter(pl.col("country") == "France")
for nm, d in [("S1", fr_s), ("records", fr_r)]:
    print(f"  {nm}: 'cie' as legal form {d['name_sfx'].str.contains('cie').sum():,}; 'compagnie' as a name word "
          f"{d['name_core'].str.contains(r'(^| )compagnie( |$)').sum():,}")

print("\n=== 3. two-letter names mapped to words: 'CB' -> club, 'FS' -> fils (acronym collision)")
for c in ["France", "US", "India"]:
    d = te_r.filter(pl.col("country") == c)
    raw2 = d["name_raw"].str.strip_chars().str.to_lowercase()
    print(f"  {c}: records whose whole name is 'CB' {int((raw2 == 'cb').sum())}, 'FS' {int((raw2 == 'fs').sum())}, "
          f"'SVC' {int((raw2 == 'svc').sum())}")

print("\n=== 4. French apostrophes: l'/d' glued to the next word (\"de l'Alma\" -> 'lalma')")
APOS = r"\b[ld]['’]\w"
SPLIT = r"\b(de|du) [ld] \w"
fa = fr_s["addr_raw"].str.to_lowercase()
n_ap_s1 = int(fa.str.contains(APOS).sum())
print(f"  French S1 addresses with l'/d' + word: {n_ap_s1:,} of {fr_s.height:,}")
fr2 = fr_r["addr_raw"].str.to_lowercase()
n_ap_r, n_sp_r = int(fr2.str.contains(APOS).sum()), int(fr2.str.contains(SPLIT).sum())
print(f"  French record addresses with l'/d' + word: {n_ap_r:,}; with 'de l word' (apostrophe replaced by a space): {n_sp_r:,}")

print("\n=== 5. Honorific strip vs name map: 'Shri/Sri X' -> 'x', but 'Shree/Sree X' -> 'shri x' / 'sri x'")
for nm, d in [("train S1", tr_s), ("train records", tr_r)]:
    low = d.filter(pl.col("country") == "India")["name_raw"].str.to_lowercase()
    a1 = int(low.str.contains(r"^(shri|sri)\b").sum()); a2 = int(low.str.contains(r"^(shree|sree)\b").sum())
    print(f"  {nm}: starts with shri/sri {a1:,}; starts with shree/sree {a2:,}")

print("\n=== 6. alias keyword written with dots ('D.B.A.', 'F.K.A.') is not split")
for c in ["US", "India", "France"]:
    d = te_r.filter(pl.col("country") == c)
    low = d["name_raw"].str.to_lowercase()
    dotted = low.str.contains(r"\b(d\.b\.a\.?|f\.k\.a\.?|a\.k\.a\.?)")
    print(f"  {c}: records with dotted alias keyword {int(dotted.sum()):,}; example cores: "
          f"{d.filter(dotted)['name_core'].head(3).to_list()}")
