"""Print the 124 stage-1 features of one real test pair, layer by layer, with the normalized inputs."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, glob
import numpy as np, polars as pl
C = _os.path.join(_WORK, 'data', 'cache')
s1 = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["eid", "name_raw", "addr_raw", "name_core", "name_clean", "name_sfx", "addr_clean", "addr_nums"])
r = pl.read_parquet(os.path.join(C, "prep_v2", "test_r.parquet"), columns=["eid", "name_raw", "addr_raw", "name_core", "name_clean", "name_sfx", "addr_clean", "addr_nums"])
si = int(s1.with_row_index("i").filter(pl.col("eid") == "S1-594414511")["i"][0])
ri = int(r.with_row_index("i").filter(pl.col("eid") == "S3-892216570")["i"][0])
for tag, t, i in [("S1", s1, si), ("record", r, ri)]:
    row = t.row(i, named=True)
    print(f"{tag:6s} raw: {row['name_raw']} | {row['addr_raw']}")
    print(f"       clean name: {row['name_clean']!r}  core: {row['name_core']!r}  legal form: {row['name_sfx']!r}")
    print(f"       clean addr: {row['addr_clean']!r}  numbers: {row['addr_nums']!r}")
layers = {"base (exp01_match)": os.path.join(C, "feat_v1", "test"), "tokens (exp07)": os.path.join(C, "feat_v7", "test"),
          "consensus (exp09)": os.path.join(C, "feat_v9", "test"), "roles (exp13)": os.path.join(C, "feat_v13", "test")}
parts = sorted(glob.glob(os.path.join(layers["base (exp01_match)"], "part*.parquet")))
for pi, p in enumerate(parts):
    k = pl.read_parquet(p, columns=["s1", "r"]).with_row_index("row")
    hit = k.filter((pl.col("s1") == si) & (pl.col("r") == ri))
    if hit.height:
        row = int(hit["row"][0]); break
print(f"\nfound in part {pi}, row {row}")
for name, d in layers.items():
    f = sorted(glob.glob(os.path.join(d, "part*.parquet")))[pi]
    x = pl.read_parquet(f).slice(row, 1).drop([c for c in ["s1", "r"] if c in pl.read_parquet_schema(f)])
    vals = x.row(0, named=True)
    print(f"\n[{name}] {len(vals)} values")
    print("  " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in vals.items()))
