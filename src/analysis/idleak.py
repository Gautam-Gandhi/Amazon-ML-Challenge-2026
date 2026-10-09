import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import numpy as np, polars as pl
D = _os.path.join(_DATA, 'train')
gt = pl.read_csv(D + r"\train_ground_truth.tsv", separator="\t", infer_schema=False).fill_null("")
gt = gt.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
gt = gt.head(2_000_000)
a = gt["source1_entity_id"].str.slice(3).cast(pl.Int64).to_numpy()
b = gt["matched_entity_ids"].str.slice(3).cast(pl.Int64).to_numpy()
print("n", len(a), "max ids", a.max(), b.max())
print("corr", np.corrcoef(a, b)[0, 1])
for m in [2, 3, 7, 10, 16, 100, 1000, 1024]:
    agree = (a % m == b % m).mean()
    print(f"mod {m}: agree {agree:.4f} (chance {1/m:.4f})")
# rank correlation
from scipy.stats import spearmanr
print("spearman", spearmanr(a[:200000], b[:200000]).correlation)
# digit patterns
sa = gt["source1_entity_id"].str.slice(3).to_numpy()
sb = gt["matched_entity_ids"].str.slice(3).to_numpy()
print("same len", np.mean([len(x) == len(y) for x, y in zip(sa[:100000], sb[:100000])]))
print("same first digit", np.mean([x[0] == y[0] for x, y in zip(sa[:100000], sb[:100000])]))
print("same last digit", np.mean([x[-1] == y[-1] for x, y in zip(sa[:100000], sb[:100000])]))
# record order in files: position of matched records
s2 = pl.read_csv(D + r"\train_source2.tsv", separator="\t", infer_schema=False, columns=["entity_id"])
s1 = pl.read_csv(D + r"\train_source1.tsv", separator="\t", infer_schema=False, columns=["entity_id"])
pos2 = dict(zip(s2["entity_id"].to_list(), range(len(s2))))
pos1 = dict(zip(s1["entity_id"].to_list(), range(len(s1))))
x = [(pos1[u], pos2[v]) for u, v in zip(gt["source1_entity_id"].to_list()[:300000], gt["matched_entity_ids"].to_list()[:300000]) if v in pos2]
x = np.array(x)
print("file-pos corr S1 vs S2", np.corrcoef(x[:, 0], x[:, 1])[0, 1], spearmanr(x[:, 0], x[:, 1]).correlation)
