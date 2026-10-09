"""Does the approximate (IVF) search lose true matches vs an exact search? Sample of US train S1 (fold 0), searched with
the fold1 model (as in production), forward channels name@20 / addr@20 / joint@30 only."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys
import numpy as np, polars as pl, torch
sys.path.insert(0, _SRC)
os.environ["ER_WORK_DIR"] = _WORK
import exp01_block as B
from er_common import s1_fold
dev = torch.device("cuda")
C = B.CACHE
s1c = pl.read_parquet(os.path.join(B.PREP, "train_s1.parquet"), columns=["country"])["country"].to_numpy()
rc = pl.read_parquet(os.path.join(B.PREP, "train_r.parquet"), columns=["country"])["country"].to_numpy()
gt = pl.read_parquet(os.path.join(B.PREP, "train_gt.parquet")).rename({"s1_idx": "s1", "r_idx": "r"})
fold = s1_fold(len(s1c))
rng = np.random.default_rng(3)
for country in ["US", "India"]:
    qs = np.where((s1c == country) & (fold == 0))[0]
    qs = np.sort(rng.choice(qs, 5000, replace=False))
    rr = np.where(rc == country)[0]
    enc = B.Encoder("train", dev)
    model = B.Towers(128).to(dev)
    model.load_state_dict(torch.load(os.path.join(B.EMB, "model_fold1.pt")))
    model.eval()
    ks = {"name": 20, "addr": 20, "joint": 30}
    res = B.search(enc, model, "s1", qs, "r", rr, ks, 128)
    ex = set()
    for ch, (sc, ix) in res.items():
        for qi in range(len(qs)):
            for j in ix[qi]:
                if j >= 0:
                    ex.add((int(qs[qi]), int(rr[j])))
    ivf = pl.read_parquet(os.path.join(C, "cand_v1", "train", f"{country}_fold1.parquet"), columns=["s1", "r", "rk_name", "rk_addr", "rk_joint"]) \
            .filter(pl.col("s1").is_in(pl.Series(qs.astype(np.int32)).implode())) \
            .filter((pl.col("rk_name") < 999) | (pl.col("rk_addr") < 999) | (pl.col("rk_joint") < 999))
    iv = set(zip(ivf["s1"].to_list(), ivf["r"].to_list()))
    g = gt.filter(pl.col("s1").is_in(pl.Series(qs.astype(np.int32)).implode()))
    tp = set(zip(g["s1"].to_list(), g["r"].to_list()))
    n = len(tp)
    print(f"{country}: {len(qs)} S1, {n} true pairs | exact finds {len(tp & ex)/n:.4f} | approximate (production) finds {len(tp & iv)/n:.4f} | "
          f"found by exact but not approximate: {len((tp & ex) - iv)} | by approximate but not exact: {len((tp & iv) - ex)}")
    del enc, model, res
    torch.cuda.empty_cache()
