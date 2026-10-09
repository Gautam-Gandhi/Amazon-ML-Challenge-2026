"""French records whose assigned (best) S1 has a different descriptor than the record, while another candidate S1 has
the record's descriptor. Counts, p of best vs alternative, samples."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import numpy as np, polars as pl
W = _WORK
DESC = set("amicale amis anciens atelier centre club collectif college comite conseil culture culturel culturelle danse ecole ehpad elementaire federation fetes foyer gestion groupement institut jeunes loisirs lycee maison maternelle musique parents patrimoine pharmacie primaire residence sante section service societe soins sport sportive theatre union".split())
C = W + r"\data\cache\prep_v2"
s1 = pl.read_parquet(C + r"\test_s1.parquet", columns=["country", "name_core", "name_raw", "addr_raw", "addr_nums"])
r = pl.read_parquet(C + r"\test_r.parquet", columns=["name_core", "name_raw", "addr_raw", "addr_nums"])
tp = pl.read_parquet(W + r"\runs\exp33\test_pred.parquet")
ctry = s1["country"].to_numpy()
tp = tp.filter(pl.Series(ctry[tp["s1"].to_numpy()] == "France"))
dset = pl.Series(list(DESC))
def descs(names):
    return names.str.split(" ").list.eval(pl.element().filter(pl.element().is_in(dset)))
tp = tp.with_columns(descs(s1["name_core"].gather(tp["s1"].to_numpy())).alias("d1"), descs(r["name_core"].gather(tp["r"].to_numpy())).alias("d2"))
tp = tp.with_columns((pl.col("d1").list.set_intersection("d2").list.len() > 0).alias("same_desc"),
                     ((pl.col("d1").list.len() > 0) & (pl.col("d2").list.len() > 0)).alias("both_desc"),
                     pl.col("p").rank("ordinal", descending=True).over("r").alias("rk"))
best = tp.filter(pl.col("rk") == 1)
alt = tp.filter((pl.col("rk") > 1) & pl.col("same_desc")).sort("p", descending=True).unique(subset=["r"], keep="first") \
        .select("r", pl.col("s1").alias("s1_alt"), pl.col("p").alias("p_alt"))
x = best.filter(pl.col("both_desc") & ~pl.col("same_desc")).join(alt, on="r", how="inner")
print("French R: best S1 has a different descriptor, another candidate S1 has the record's descriptor:", x.height)
print("  best accepted (p>0.9):", (x["p"] > 0.9).sum(), "; alt p hist:", np.histogram(x["p_alt"].to_numpy(), bins=[0, 0.05, 0.2, 0.5, 0.9, 1.01])[0].tolist())
print("  best p hist:", np.histogram(x["p"].to_numpy(), bins=[0, 0.05, 0.2, 0.5, 0.9, 1.01])[0].tolist())
for row in x.filter(pl.col("p") > 0.5).sample(min(12, x.filter(pl.col("p") > 0.5).height), seed=1).iter_rows(named=True):
    print(f"  R: {r['name_raw'][row['r']]} | {r['addr_raw'][row['r']]}")
    print(f"     best p={row['p']:.3f}: {s1['name_raw'][row['s1']]} | {s1['addr_raw'][row['s1']]}")
    print(f"     alt  p={row['p_alt']:.3f}: {s1['name_raw'][row['s1_alt']]} | {s1['addr_raw'][row['s1_alt']]}")
