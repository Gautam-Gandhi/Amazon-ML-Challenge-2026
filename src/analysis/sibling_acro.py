"""Are there sibling acronyms in France? At each S1 address (same first number + street), count acronym records whose
letters equal the S1's initials (A) vs equal them except the LAST letter (B: e.g. 'VA' next to 'Volley Comite'='VC').
B can only be siblings / coincidences; compare with US and India (which have no descriptor siblings)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os
import numpy as np, polars as pl
C = _os.path.join(_WORK, 'data', 'cache')
STOP = {"la", "le", "les", "de", "du", "des", "d", "l", "et", "and", "of", "the", "a", "en", "au", "aux"}
s1 = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["country", "name_core", "addr_clean", "addr_nums"]).with_row_index("s1")
r = pl.read_parquet(os.path.join(C, "prep_v2", "test_r.parquet"), columns=["country", "name_core", "addr_clean", "addr_nums"]).with_row_index("r")
def street(df):
    return df.with_columns(pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("h"),
                           pl.col("addr_clean").str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"\d") & (pl.element().str.len_chars() > 1) & ~pl.element().str.starts_with("fr"))).alias("w"))
s1 = street(s1); r = street(r)
ini = lambda c: "".join(t[0] for t in c.split() if t and t not in STOP)
s1 = s1.with_columns(pl.col("name_core").map_elements(ini, return_dtype=pl.Utf8).alias("ini")).filter(pl.col("ini").str.len_chars().is_between(2, 5) & (pl.col("h") != ""))
ra = r.filter(pl.col("name_core").str.contains(r"^[a-z]{2,5}$") & (pl.col("h") != ""))
for c in ["France", "US", "India"]:
    a = s1.filter(pl.col("country") == c).with_columns(pl.col("ini").str.slice(0, pl.col("ini").str.len_chars() - 1).alias("stem"))
    b = ra.filter(pl.col("country") == c).with_columns(pl.col("name_core").str.slice(0, pl.col("name_core").str.len_chars() - 1).alias("stem"))
    j = b.join(a.select("s1", "ini", "stem", "h", pl.col("w").alias("w1")), on=["stem", "h"])
    j = j.with_columns((pl.col("w").list.set_intersection("w1").list.len() / pl.col("w").list.set_union("w1").list.len().clip(1, None)).alias("jac")).filter(pl.col("jac") >= 0.5)
    A = j.filter(pl.col("name_core") == pl.col("ini"))
    B = j.filter(pl.col("name_core") != pl.col("ini"))
    n1 = a.height
    print(f"{c}: S1 with 2-5 initials {n1:,} | same-address acronym records: exact initials A={A['r'].n_unique():,} ({1000*A['r'].n_unique()/n1:.1f}/1K S1) | "
          f"last letter differs B={B['r'].n_unique():,} ({1000*B['r'].n_unique()/n1:.1f}/1K S1) | B/A = {B['r'].n_unique()/max(A['r'].n_unique(),1):.3f}")
    if c == "France":
        print("   B examples (S1 initials -> record):", B.select("ini", "name_core").head(12).rows())
