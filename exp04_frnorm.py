"""exp04_frnorm: French region/department canonicalization on the test side (models unchanged).

Finding: French S1 addresses end with the *region* (95%: Hauts-de-France / Nouvelle-Aquitaine / Pays de la Loire)
while S2/S3 end with the region (~33%), the *department* (~31%: Nord, Gironde, Loire-Atlantique, Pas-de-Calais) or
nothing. US/India never have this conflict (states -> codes in prep_v1), so every French pair loses address
similarity (median ad_tset 94 vs 100, cos_a 0.82 vs 0.91) and the US/India-trained models under-score true matches.
Fix: map every French department and region name to one region token (generic domain normalization, like US
state codes). Train addresses contain none of these names (10 rows with 'nord'), so trained models stay valid:
only the test side is recomputed (prep_v2/test -> emb_v2 -> cand_v2 -> feat_v4) and scored with existing models.

Stages: prep | emb | search | feats | predict --base {exp01,exp03}
"""
import os
import re
import sys
import glob
import json
import shutil
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, write_outputs, validate_outputs  # noqa: E402
import exp01_block as B  # noqa: E402
import exp01_match as M1  # noqa: E402
import exp02_stack as M2  # noqa: E402

PREP1 = os.path.join(CACHE, "prep_v1")
PREP2 = os.path.join(CACHE, "prep_v2")
EMB1 = os.path.join(CACHE, "emb_v1")
EMB2 = os.path.join(CACHE, "emb_v2")
CAND2 = os.path.join(CACHE, "cand_v2")
FEAT4 = os.path.join(CACHE, "feat_v4")

# French metropolitan regions (2016) -> short code; departments -> their region.  Names as they appear after
# prep_v1 normalization (lowercase, accents stripped, hyphens -> spaces).
FR_REGIONS = {
    "hauts de france": "frhdf", "nouvelle aquitaine": "frnaq", "pays de la loire": "frpdl", "ile de france": "fridf",
    "grand est": "frges", "normandie": "frnor", "bretagne": "frbre", "centre val de loire": "frcvl",
    "bourgogne franche comte": "frbfc", "auvergne rhone alpes": "frara", "occitanie": "frocc",
    "provence alpes cote d azur": "frpac", "provence alpes cote dazur": "frpac", "corse": "frcor",
}
FR_DEPTS = {
    "frhdf": ["nord", "pas de calais", "somme", "oise", "aisne"],
    "frnaq": ["gironde", "landes", "pyrenees atlantiques", "lot et garonne", "dordogne", "charente", "charente maritime",
              "deux sevres", "vienne", "haute vienne", "creuse", "correze"],
    "frpdl": ["loire atlantique", "maine et loire", "vendee", "sarthe", "mayenne"],
    "fridf": ["paris", "seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis", "seine st denis",
              "val de marne", "val d oise", "val doise"],
    "frbre": ["finistere", "morbihan", "ille et vilaine", "cotes d armor", "cotes darmor"],
    "frnor": ["calvados", "manche", "orne", "eure", "seine maritime"],
}
_PHR = dict(FR_REGIONS)
for code, ds in FR_DEPTS.items():
    for d in ds:
        _PHR[d] = code
# prep_v1 already turned 'saint' into 'st'
_PHR = {k.replace("saint", "st"): v for k, v in _PHR.items()}
FR_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _PHR), key=len, reverse=True)) + r")\b")


def canon_fr(addr):
    if not addr:
        return addr
    s = FR_RE.sub(lambda m: _PHR[m.group(1)], addr)
    # a region token repeated (department + region both present) -> keep one
    toks = s.split()
    out = [t for i, t in enumerate(toks) if not (t.startswith("fr") and t in toks[:i] and t in _PHR.values())]
    return " ".join(out)


def patch_dirs():
    B.PREP, B.EMB, B.CAND = PREP2, EMB2, CAND2
    M1.PREP, M1.CAND, M1.FEAT = PREP2, CAND2, FEAT4
    M2.PREP = PREP2


def stage_prep(args, log):
    os.makedirs(PREP2, exist_ok=True)
    for tag in ["s1", "r"]:
        d = pl.read_parquet(os.path.join(PREP1, f"test_{tag}.parquet"))
        new = [canon_fr(a) if c == "France" else a for a, c in zip(d["addr_clean"].to_list(), d["country"].to_list())]
        changed = sum(a != b for a, b in zip(new, d["addr_clean"].to_list()))
        d = d.with_columns(pl.Series("addr_clean", new))
        d.write_parquet(os.path.join(PREP2, f"test_{tag}.parquet"))
        log(f"test_{tag}: {changed} addresses changed")
    # train is untouched: hardlink the prep_v1 files so code reading PREP2 for train (GT, labels) works
    for f in ["train_s1.parquet", "train_r.parquet", "train_gt.parquet"]:
        dst = os.path.join(PREP2, f)
        if not os.path.exists(dst):
            os.link(os.path.join(PREP1, f), dst)
    for code in sorted(set(_PHR.values())):
        pass
    ex = pl.read_parquet(os.path.join(PREP2, "test_r.parquet"), columns=["addr_raw", "addr_clean", "country"]) \
           .filter(pl.col("country") == "France").sample(5, seed=1)
    for a, b in zip(ex["addr_raw"], ex["addr_clean"]):
        log(f"   {a}  ->  {b}")


def stage_emb(args, log):
    """Hashed features: re-hash test addresses, reuse everything else via hardlinks (models, name features)."""
    from multiprocessing import Pool
    os.makedirs(EMB2, exist_ok=True)
    for f in ["model_fold0.pt", "model_fold1.pt", "test_s1_name.npz", "test_r_name.npz"] + \
             [f"train_{t}_{x}.npz" for t in ["s1", "r"] for x in ["name", "addr"]]:
        dst = os.path.join(EMB2, f)
        if not os.path.exists(dst):
            os.link(os.path.join(EMB1, f), dst)
    with Pool(6) as pool:
        for tag in ["s1", "r"]:
            texts = pl.read_parquet(os.path.join(PREP2, f"test_{tag}.parquet"), columns=["addr_clean"])["addr_clean"].to_list()
            lens, idx = [], []
            for a, b in pool.imap(B._hash_chunk, ((texts[i:i + 50000], "addr") for i in range(0, len(texts), 50000))):
                lens.append(a); idx.append(b)
            lens = np.concatenate(lens)
            indptr = np.zeros(len(lens) + 1, np.int64)
            np.cumsum(lens, out=indptr[1:])
            np.savez(os.path.join(EMB2, f"test_{tag}_addr.npz"), indptr=indptr, indices=np.concatenate(idx))
            log(f"hashed test_{tag}_addr")


def stage_search(args, log):
    patch_dirs()
    # US/India addresses are unchanged -> reuse their exp01 test candidate files; only France is searched again
    os.makedirs(os.path.join(CAND2, "test"), exist_ok=True)
    for c in ["US", "India"]:
        src = os.path.join(CACHE, "cand_v1", "test", f"{c}_fold0.parquet")
        dst = os.path.join(CAND2, "test", f"{c}_fold0.parquet")
        if not os.path.exists(dst):
            os.link(src, dst)
    a = argparse.Namespace(split="test", test_tag="fold0", dim=128, k_name=20, k_addr=20, k_joint=30, k_rev=3, nprobe=32)
    B.stage_search(a, log)


def stage_feats(args, log):
    patch_dirs()
    a = argparse.Namespace(split="test", max_cand=15, keep_rrank=2)
    M1.stage_feats(a, log)


def predict_stage1(run_dir, prefix, parts):
    import xgboost as xgb
    bs = []
    for k in (0, 1):
        b = xgb.Booster(); b.load_model(os.path.join(run_dir, f"{prefix}_fold{k}.json")); bs.append(b)
    outs = []
    for p in parts:
        part = pl.read_parquet(p)
        X = part.select(bs[0].feature_names).to_numpy().astype(np.float32)
        outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"],
                                  "p": np.mean([b.inplace_predict(X) for b in bs], axis=0).astype(np.float32)}))
    return pl.concat(outs)


def stage_predict(args, log):
    import xgboost as xgb
    patch_dirs()
    parts = sorted(glob.glob(os.path.join(FEAT4, "test", "part*.parquet")))
    run_dir = os.path.join(RUNS, args.run)
    os.makedirs(run_dir, exist_ok=True)
    base_dir = os.path.join(RUNS, args.base)
    if args.base == "exp01":
        res = predict_stage1(base_dir, "xgb", parts)
        thr = args.thr or 0.75
    else:  # exp03: stage 1 -> stage-2 features -> stage 2
        s1p = predict_stage1(base_dir, "stage1", parts)
        t = M2.group_feats(s1p.with_row_index("row"))
        r_tab = pl.read_parquet(os.path.join(PREP2, "test_r.parquet"), columns=["name_core", "addr_clean", "addr_nums", "src"])
        cons = M2.consistency_feats(t, r_tab, 3, log)
        f2 = pl.concat([t.drop("row"), cons.drop("row")], how="horizontal").rename({"p": "p1"}).drop("s1", "r")
        bs = []
        for k in (0, 1):
            b = xgb.Booster(); b.load_model(os.path.join(base_dir, f"stage2_fold{k}.json")); bs.append(b)
        outs, st = [], 0
        for p in parts:
            part = pl.read_parquet(p)
            part = pl.concat([part, f2.slice(st, part.height)], how="horizontal")
            st += part.height
            X = part.select(bs[0].feature_names).to_numpy().astype(np.float32)
            outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"],
                                      "p": np.mean([b.inplace_predict(X) for b in bs], axis=0).astype(np.float32)}))
        res = pl.concat(outs)
        with open(os.path.join(base_dir, "metrics.json")) as f:
            thr = args.thr or json.load(f)["best_thr"]
    res.write_parquet(os.path.join(run_dir, "test_pred.parquet"))
    ms, mr = M1.decode(res, thr, "assign")
    s1_ids = pl.read_parquet(os.path.join(PREP2, "test_s1.parquet"), columns=["eid"])["eid"].to_numpy()
    r_ids = pl.read_parquet(os.path.join(PREP2, "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(run_dir, "output")
    write_outputs(out_dir, s1_ids, ms, mr, res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    log(f"base={args.base} thr={thr}")
    M2.PREP = PREP2
    M2.summarize_test(ms, s1_ids, log)
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "emb", "search", "feats", "predict"])
    ap.add_argument("--base", default="exp03")
    ap.add_argument("--run", default="exp04")
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    from er_common import set_determinism
    set_determinism(0)  # reproducible reruns
    {"prep": stage_prep, "emb": stage_emb, "search": stage_search, "feats": stage_feats,
     "predict": stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
