"""French S1s with an empty prediction: what are their candidates? (+ same stats for US/India)"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import sys
import numpy as np, polars as pl
W = _WORK
country = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3])
s1 = pl.read_parquet(W + r"\data\cache\prep_v1\test_s1.parquet", columns=["eid", "country", "name_raw", "addr_raw"])
r = pl.read_parquet(W + r"\data\cache\prep_v1\test_r.parquet", columns=["eid", "name_raw", "addr_raw"])
pred = pl.read_parquet(W + r"\runs\exp29gf\test_pred.parquet")
# the submitted matching file tells which S1 are empty
m = pl.read_csv(W + r"\runs\exp29gf\output\matching_results.tsv", separator="\t", infer_schema=False).fill_null("")
eid2i = dict(zip(s1["eid"].to_list(), range(s1.height)))
empty = np.zeros(s1.height, bool)
for sid, lst in zip(m["source1_entity_id"].to_list(), m["matched_entity_ids"].to_list()):
    if lst == "":
        empty[eid2i[sid]] = True
ctry = s1["country"].to_numpy()
pr = pred.with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("rk"))
best_of_r = pr.filter(pl.col("rk") == 1)
for c in ["US", "India", "France"]:
    e = empty & (ctry == c)
    idx = np.where(e)[0]
    sub = pr.filter(pl.col("s1").is_in(idx))
    mx = sub.group_by("s1").agg(pl.col("p").max().alias("pm"), (pl.col("rk") == 1).any().alias("isbest"))
    pm = np.zeros(len(idx)); d = dict(zip(mx["s1"].to_list(), mx["pm"].to_list())); pm = np.array([d.get(i, -1) for i in idx])
    h = np.histogram(pm, bins=[-2, -0.5, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 1.01])[0]
    print(f"{c}: empty {e.sum()} ({e.sum()/(ctry==c).sum():.4f}); max cand p hist [none, <.05, .05-.2, .2-.4, .4-.6, .6-.8, .8-.9, >.9]: {h.tolist()}")
idx = np.where(empty & (ctry == country))[0]
rng = np.random.default_rng(seed)
shown = 0
for s in rng.permutation(idx):
    sub = pr.filter(pl.col("s1") == int(s)).sort("p", descending=True)
    if sub.height == 0 or sub["p"].max() < 0.2:
        continue
    print("=" * 100)
    print(f"S1: {s1['name_raw'][int(s)]} | {s1['addr_raw'][int(s)]}")
    for row in sub.head(4).iter_rows(named=True):
        alt = pr.filter((pl.col("r") == row["r"]) & (pl.col("s1") != int(s))).sort("p", descending=True).head(1)
        altt = ""
        if alt.height:
            a0 = alt.row(0, named=True)
            altt = f"   [alt p={a0['p']:.2f}: {s1['name_raw'][a0['s1']]} | {s1['addr_raw'][a0['s1']]}]"
        print(f"   p={row['p']:.3f} rk={row['rk']} | {r['name_raw'][row['r']]} | {r['addr_raw'][row['r']]}{altt}")
    shown += 1
    if shown >= n:
        break
