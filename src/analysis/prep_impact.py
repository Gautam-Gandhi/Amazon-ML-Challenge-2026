"""Does each normalization gap actually cost accuracy? Dense OOF (labels) for US/India, test p for France."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, argparse
import numpy as np, polars as pl
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
import exp03_dense as M3
C = _os.path.join(_WORK, 'data', 'cache')
R = _os.path.join(_WORK, 'runs')
keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
gt = M3.labels_kept(keep).rename({"s1_idx": "s1", "r_idx": "r"}).with_columns(pl.col("s1").cast(pl.Int32), pl.col("r").cast(pl.Int32), pl.lit(1).alias("y"))
s1 = pl.read_parquet(os.path.join(C, "prep_v1", "train_s1.parquet"), columns=["country", "name_raw"])
r = pl.read_parquet(os.path.join(C, "prep_v1", "train_r.parquet"), columns=["name_raw"])
oof = pl.read_parquet(os.path.join(R, "exp29g", "oof_combined.parquet"))
a = oof.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
acc = a.filter(pl.col("p") > 0.7).select("s1", "r").with_columns(pl.lit(True).alias("acc"))
g = gt.join(acc, on=["s1", "r"], how="left").with_columns(pl.col("acc").fill_null(False))
sn = s1["name_raw"].str.to_lowercase(); rn = r["name_raw"].str.to_lowercase()
g = g.with_columns(pl.Series("s1n", sn.gather(g["s1"].to_numpy())), pl.Series("rn", rn.gather(g["r"].to_numpy())))
print("=== 5. Shree/Sree S1 vs Shri/Sri record (dense OOF, TRUE pairs): recall = share of true pairs accepted")
x1 = g.filter(pl.col("s1n").str.contains(r"^(shree|sree)\b"))
m_bad = x1["rn"].str.contains(r"^(shri|sri)\b"); m_ok = x1["rn"].str.contains(r"^(shree|sree)\b")
print(f"  S1 starts shree/sree, record starts shri/sri (prep drops the word on one side): n={int(m_bad.sum()):,} recall {x1.filter(m_bad)['acc'].mean():.4f}")
print(f"  S1 starts shree/sree, record starts shree/sree (consistent):                  n={int(m_ok.sum()):,} recall {x1.filter(m_ok)['acc'].mean():.4f}")
print(f"  all true pairs (reference): recall {g['acc'].mean():.4f}")
print("=== 6. dotted alias keyword (D.B.A.) vs split alias (dba / doing business as), TRUE pairs")
dotted = g["rn"].str.contains(r"\b(d\.b\.a\.?|f\.k\.a\.?|a\.k\.a\.?)"); plain = g["rn"].str.contains(r"\b(dba|fka|aka|doing business as|formerly|f/k/a|d/b/a)\b") & ~dotted
print(f"  dotted: n={int(dotted.sum()):,} recall {g.filter(dotted)['acc'].mean():.4f};  plain (split by prep): n={int(plain.sum()):,} recall {g.filter(plain)['acc'].mean():.4f}")
# false positives among accepted with dotted aliases
aa = acc.join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0)).with_columns(pl.Series("rn", rn.gather(acc["r"].to_numpy())))
d2 = aa["rn"].str.contains(r"\b(d\.b\.a\.?|f\.k\.a\.?|a\.k\.a\.?)")
print(f"  accepted dotted-alias pairs precision {aa.filter(d2)['y'].mean():.4f} (all accepted {aa['y'].mean():.4f})")
print("=== 2. France: Cie <-> Compagnie mismatch between S1 and its best record (test)")
ts = pl.read_parquet(os.path.join(C, "prep_v2", "test_s1.parquet"), columns=["country", "name_core", "name_sfx"])
tr = pl.read_parquet(os.path.join(C, "prep_v2", "test_r.parquet"), columns=["name_core", "name_sfx"])
tp = pl.read_parquet(os.path.join(R, "exp33", "test_pred.parquet"))
b = tp.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
b = b.filter(pl.Series(ts["country"].to_numpy()[b["s1"].to_numpy()] == "France"))
si, ri = b["s1"].to_numpy(), b["r"].to_numpy()
cw = r"(^| )compagnie( |$)"
s_comp = ts["name_core"].gather(si).str.contains(cw).to_numpy(); r_comp = tr["name_core"].gather(ri).str.contains(cw).to_numpy()
s_cie = ts["name_sfx"].gather(si).str.contains("cie").to_numpy(); r_cie = tr["name_sfx"].gather(ri).str.contains("cie").to_numpy()
mis = (s_comp & r_cie & ~r_comp) | (s_cie & r_comp & ~s_comp)
p = b["p"].to_numpy()
print(f"  French best-S1 pairs with Cie<->Compagnie mismatch: {int(mis.sum()):,}; accepted (p>0.9) {int((mis & (p > 0.9)).sum()):,}; "
      f"p in (0.1,0.9]: {int((mis & (p > 0.1) & (p <= 0.9)).sum()):,}; mean p {p[mis].mean():.3f}")
