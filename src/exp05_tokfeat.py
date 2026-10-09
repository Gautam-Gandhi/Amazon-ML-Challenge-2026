"""exp05_tokfeat: token-alignment + house-number-difference features (language-agnostic) on top of exp03/exp04.

Why (loss breakdown + borderline-pair inspection): ~60% of the remaining loss is matcher FN/FP, dominated by
"family" distractors - same street / same address with ONE name word swapped ("Recherche Club" vs "Recherche
Pharmacie", "Thompson Academy" vs "Thompson Realty") or a nearby house number (1704 vs 1708), while noise produces
typos ("Lighmtwave") and digit deletions (1616 -> 161). Whole-string fuzzy scores blur the two. New features:
  name / address token alignment (Jaro-Winkler best match per token, both directions):
     weakest-token similarity, #unmatched tokens (<0.85), max IDF of an unmatched token, IDF-weighted coverage
  house numbers: first-number absolute / relative difference, prefix/suffix (digit deletion) flag, digit edit
     distance, #S1 numbers without an equal number in R
These do not depend on language or country, so they should also transfer to France.

Stages (dense world k80s0 for train as in exp03; test = exp04 features feat_v4/test):
  feats   : new features for train (feat_v3/k80s0 pairs) and test (feat_v4/test pairs) -> feat_v5/{train_k80s0,test}
  train1  : stage-1 cross-fit on feat_v3 + feat_v5
  feats2  : stage-2 (exp02-style) features from stage-1 p
  train2  : stage-2 cross-fit + decoding
  predict : test -> decode with in-country thr and --unseen_thr for countries absent from train (France)
"""
import os
import sys
import glob
import json
import argparse
from multiprocessing import Pool

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, save_json, write_outputs, validate_outputs  # noqa: E402
import exp01_match as M1  # noqa: E402
import exp02_stack as M2  # noqa: E402
import exp03_dense as M3  # noqa: E402

FEAT3 = os.path.join(CACHE, "feat_v3", "k80s0")
FEAT4T = os.path.join(CACHE, "feat_v4", "test")
FEAT5 = os.path.join(CACHE, "feat_v5")
PREP = {"train": os.path.join(CACHE, "prep_v1"), "test": os.path.join(CACHE, "prep_v2")}
S2DIR = os.path.join(CACHE, "feat_v5s2")
TRAIN_COUNTRIES = {"US", "India"}

_IDF = {}


def _init(idf):
    global _IDF
    _IDF = idf


def _align(ta, tb, jw):
    """Best JW similarity of each token in ta against tb -> (min_sim, n_unmatched, max_idf_unmatched, idf_coverage)."""
    if not ta:
        return 1.0, 0, 0.0, 1.0
    if not tb:
        return 0.0, len(ta), max(_IDF.get(t, 8.0) for t in ta), 0.0
    mins, nun, mx, cov, tot = 1.0, 0, 0.0, 0.0, 0.0
    for t in ta:
        best = max(jw(t, u) for u in tb) if t not in tb else 1.0
        w = _IDF.get(t, 8.0)
        tot += w
        cov += w * best
        if best < mins:
            mins = best
        if best < 0.85:
            nun += 1
            if w > mx:
                mx = w
    return mins, nun, mx, cov / tot if tot else 1.0


def _num_feats(na, nb):
    a = na.split()
    b = nb.split()
    if not a or not b:
        return -1.0, -1.0, -1, -1, (len(a) if not b else 0)
    x, y = a[0], b[0]
    xi, yi = int(x[:12]), int(y[:12])
    diff = abs(xi - yi)
    rel = diff / max(xi, yi, 1)
    pre = int(x != y and (x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x)))
    from rapidfuzz.distance import Levenshtein
    ed = Levenshtein.distance(x, y)
    sb = set(b)
    miss = sum(1 for t in a if t not in sb)
    return float(np.log1p(diff)), float(rel), pre, ed, miss


def _chunk(args):
    from rapidfuzz.distance import JaroWinkler
    jw = JaroWinkler.normalized_similarity
    an, bn, aa, ba, anum, bnum = args
    out = np.zeros((len(an), 21), np.float32)
    for i in range(len(an)):
        ta, tb = an[i].split(), bn[i].split()
        f1 = _align(ta, tb, jw)
        f2 = _align(tb, ta, jw)
        sa, sb = [t for t in aa[i].split() if not t.isdigit()], [t for t in ba[i].split() if not t.isdigit()]
        f3 = _align(sa, sb, jw) if sb else (-1.0, -1, -1.0, -1.0)
        f4 = _align(sb, sa, jw) if sb else (-1.0, -1, -1.0, -1.0)
        f5 = _num_feats(anum[i], bnum[i])
        out[i] = (*f1, *f2, *f3, *f4, *f5)
    return out


NEW_COLS = ["tk_a_min", "tk_a_nun", "tk_a_idfun", "tk_a_cov", "tk_b_min", "tk_b_nun", "tk_b_idfun", "tk_b_cov",
            "atk_a_min", "atk_a_nun", "atk_a_idfun", "atk_a_cov", "atk_b_min", "atk_b_nun", "atk_b_idfun", "atk_b_cov",
            "hn_logdiff", "hn_rel", "hn_prefix", "hn_edit", "hn_nmiss"]


def idf_table(split):
    """Token IDF over the split's S1+R name and address tokens (documents = records)."""
    cols = ["name_core", "addr_clean"]
    s1 = pl.read_parquet(os.path.join(PREP[split], f"{split}_s1.parquet"), columns=cols)
    r = pl.read_parquet(os.path.join(PREP[split], f"{split}_r.parquet"), columns=cols)
    n = s1.height + r.height
    idf = {}
    for c in cols:
        toks = pl.concat([s1[c], r[c]]).str.split(" ").list.unique().explode()
        vc = toks.value_counts()
        for t, k in zip(vc[vc.columns[0]].to_list(), vc["count"].to_list()):
            if t:
                idf[t] = min(idf.get(t, 99.0), float(np.log((n + 1) / (k + 1)) + 1))
    return idf


def stage_feats(args, log):
    for split, src in [("train", FEAT3), ("test", FEAT4T)]:
        out_dir = os.path.join(FEAT5, split)
        os.makedirs(out_dir, exist_ok=True)
        idf = idf_table(split)
        log(f"{split}: idf vocab {len(idf)}")
        s1 = pl.read_parquet(os.path.join(PREP[split], f"{split}_s1.parquet"), columns=["name_core", "addr_clean", "addr_nums"])
        r = pl.read_parquet(os.path.join(PREP[split], f"{split}_r.parquet"), columns=["name_core", "addr_clean", "addr_nums"])
        with Pool(8, initializer=_init, initargs=(idf,)) as pool:
            for p in sorted(glob.glob(os.path.join(src, "part*.parquet"))):
                dst = os.path.join(out_dir, os.path.basename(p))
                if os.path.exists(dst):
                    continue
                k = pl.read_parquet(p, columns=["s1", "r"])
                A = s1[k["s1"].to_numpy()]
                B = r[k["r"].to_numpy()]
                cols = [A["name_core"].to_list(), B["name_core"].to_list(), A["addr_clean"].to_list(),
                        B["addr_clean"].to_list(), A["addr_nums"].to_list(), B["addr_nums"].to_list()]
                step = 20000
                jobs = [tuple(c[i:i + step] for c in cols) for i in range(0, k.height, step)]
                X = np.concatenate(pool.map(_chunk, jobs, chunksize=4))
                pl.DataFrame({c: X[:, j] for j, c in enumerate(NEW_COLS)}).write_parquet(dst)
                log(f"{split} {os.path.basename(p)}: {k.height} rows")


# ---------------------------------------------------------------------------- training / prediction
def w_args(args):
    return argparse.Namespace(keep_frac=0.8, world_seed=0, **{k: v for k, v in vars(args).items()})


def iter_s1(split):
    src = FEAT3 if split == "train" else FEAT4T
    p1 = sorted(glob.glob(os.path.join(src, "part*.parquet")))
    p5 = sorted(glob.glob(os.path.join(FEAT5, split, "part*.parquet")))
    assert len(p1) == len(p5), (len(p1), len(p5))
    return lambda: (pl.concat([pl.read_parquet(a), pl.read_parquet(b)], how="horizontal") for a, b in zip(p1, p5))


def iter_s2(split):
    base = iter_s1(split)
    p2 = sorted(glob.glob(os.path.join(S2DIR, split, "part*.parquet")))
    return lambda: (pl.concat([a, pl.read_parquet(b)], how="horizontal") for a, b in zip(base(), p2))


def stage_train1(args, log):
    run_dir = os.path.join(RUNS, args.run)
    os.makedirs(run_dir, exist_ok=True)
    keep = M3.keep_mask(w_args(args))
    res = M3.cross_fit(args, log, iter_s1("train"), run_dir, "stage1", keep)
    res.write_parquet(os.path.join(run_dir, "s1_oof.parquet"))
    m = M3.decode_eval(res, keep, log, "exp05 stage1")
    save_json(m, os.path.join(run_dir, "metrics_stage1.json"))


def _predict(run_dir, prefix, it):
    import xgboost as xgb
    bs = []
    for k in (0, 1):
        b = xgb.Booster(); b.load_model(os.path.join(run_dir, f"{prefix}_fold{k}.json")); bs.append(b)
    outs = []
    for part in it():
        X = part.select(bs[0].feature_names).to_numpy().astype(np.float32)
        outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"],
                                  "p": np.mean([b.inplace_predict(X) for b in bs], axis=0).astype(np.float32)}))
    return pl.concat(outs)


def stage_feats2(args, log):
    run_dir = os.path.join(RUNS, args.run)
    for split in ["train", "test"]:
        if split == "train":
            t = pl.read_parquet(os.path.join(run_dir, "s1_oof.parquet"), columns=["s1", "r", "p"])
        else:
            t = _predict(run_dir, "stage1", iter_s1("test"))
        t = M2.group_feats(t.with_row_index("row"))
        r_tab = pl.read_parquet(os.path.join(PREP[split], f"{split}_r.parquet"),
                                columns=["name_core", "addr_clean", "addr_nums", "src"])
        cons = M2.consistency_feats(t, r_tab, 3, log)
        full = pl.concat([t.drop("row"), cons.drop("row")], how="horizontal").rename({"p": "p1"}).drop("s1", "r")
        out = os.path.join(S2DIR, split)
        os.makedirs(out, exist_ok=True)
        st = 0
        for i, p in enumerate(sorted(glob.glob(os.path.join(FEAT5, split, "part*.parquet")))):
            n = pl.scan_parquet(p).select(pl.len()).collect().item()
            full.slice(st, n).write_parquet(os.path.join(out, os.path.basename(p)))
            st += n
        assert st == full.height
        log(f"stage-2 feats {split}: {full.height}")


def stage_train2(args, log):
    run_dir = os.path.join(RUNS, args.run)
    keep = M3.keep_mask(w_args(args))
    res = M3.cross_fit(args, log, iter_s2("train"), run_dir, "stage2", keep)
    res.write_parquet(os.path.join(run_dir, "oof.parquet"))
    m = M3.decode_eval(res, keep, log, "exp05 stage2")
    save_json(m, os.path.join(run_dir, "metrics.json"))


def stage_predict(args, log):
    run_dir = os.path.join(RUNS, args.run)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        thr = args.thr if args.thr is not None else json.load(f)["best_thr"]
    res = _predict(run_dir, "stage2", iter_s2("test"))
    res.write_parquet(os.path.join(run_dir, "test_pred.parquet"))
    s1 = pl.read_parquet(os.path.join(PREP["test"], "test_s1.parquet"), columns=["eid", "country"])
    country = s1["country"].to_numpy()
    t = np.where(np.isin(country, list(TRAIN_COUNTRIES)), thr, args.unseen_thr).astype(np.float32)
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.with_columns(pl.Series("t", t[a["s1"].to_numpy()])).filter(pl.col("p") > pl.col("t"))
    r_ids = pl.read_parquet(os.path.join(PREP["test"], "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(run_dir, "output")
    write_outputs(out_dir, s1["eid"].to_numpy(), a["s1"].to_numpy(), a["r"].to_numpy(),
                  res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    n = np.bincount(a["s1"].to_numpy(), minlength=len(country))
    for c in np.unique(country):
        m = country == c
        log(f"test {c}: thr {thr if c in TRAIN_COUNTRIES else args.unseen_thr} mean matches {n[m].mean():.3f} "
            f"empty {np.mean(n[m] == 0):.4f}")
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "train1", "feats2", "train2", "predict"])
    ap.add_argument("--run", default="exp05")
    ap.add_argument("--neg_frac", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    args = ap.parse_args()
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    from er_common import set_determinism
    set_determinism(0)  # reproducible reruns
    {"feats": stage_feats, "train1": stage_train1, "feats2": stage_feats2, "train2": stage_train2,
     "predict": stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
