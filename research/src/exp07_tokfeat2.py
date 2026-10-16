"""exp07_tokfeat2: token features v2 = exp05 features with abbreviation-aware alignment + NOISE-WORD features.

Why (token-substitution mining, tools/subst_mining.py): the noise generator *inserts or swaps in* generic words
("center", "services", "partners", "enterprises"...: 12K extra "center" in 400K US GT pairs), while a swapped
*content* word ("Solidair Sportive" vs "Solidair Loisirs", "Thompson Academy" vs "Thompson Realty") signals a
different business. exp05 only knows that a token is unmatched, not what kind of word it is.
Noise score of a word t (per split, from the split's own records - like IDF, no labels):
    ns(t) = log((#R names containing t + 1) / (#S1 names containing t + 1)) - log(N_R / N_S1)
Generic noise words are strongly over-represented in S2/S3 names (ns >> 0); real name words are not (ns ~ 0).
Because it is computed from each split's data, French words get their own scores from the French test records.
Also: token alignment treats abbreviations as matches (frs ~ freres, svc ~ service: same first letter, subsequence).

Stages (the model part reuses exp05 code with this feature layer instead of feat_v5):
  feats   : feat_v7/{train,test} (row-aligned with feat_v3/k80s0 train parts and --test_feat test parts)
  train1 / feats2 / train2 / predict : as exp05 (dense world k80s0, unseen-country threshold)
"""
import os
import sys
import glob
import argparse
from multiprocessing import Pool

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger  # noqa: E402
import exp05_tokfeat as E5  # noqa: E402

FEAT7 = os.path.join(CACHE, "feat_v7")
S2DIR7 = os.path.join(CACHE, "feat_v7s2")
_IDF, _NS, _ANS = {}, {}, {}


def _init(idf, ns, ans):
    global _IDF, _NS, _ANS
    _IDF, _NS, _ANS = idf, ns, ans


def _is_abbrev(short, long_):
    """short is an abbreviation of long_: same first letter, len >= 2, letters appear in order (frs ~ freres)."""
    if len(short) < 2 or len(short) >= len(long_) or short[0] != long_[0]:
        return False
    it = iter(long_)
    return all(ch in it for ch in short)


def _align(ta, tb, jw, ns):
    """Tokens of ta vs tb. Returns (min_sim, n_unmatched, max_idf_unmatched, idf_coverage,
    max_ns_unmatched, min_ns_unmatched, n_unmatched_noise(ns>1), n_unmatched_content(ns<=0.5))."""
    if not ta:
        return 1.0, 0, 0.0, 1.0, -9.0, -9.0, 0, 0
    mins, nun, mx, cov, tot = 1.0, 0, 0.0, 0.0, 0.0
    nsmax, nsmin, nnoise, ncont = -9.0, 9.0, 0, 0
    for t in ta:
        if t in tb:
            best = 1.0
        elif tb:
            best = max(jw(t, u) for u in tb)
            if best < 0.85 and any(_is_abbrev(t, u) or _is_abbrev(u, t) for u in tb):
                best = 0.9
        else:
            best = 0.0
        w = _IDF.get(t, 8.0)
        tot += w
        cov += w * best
        mins = min(mins, best)
        if best < 0.85:
            nun += 1
            mx = max(mx, w)
            s = ns.get(t, 0.0)
            nsmax, nsmin = max(nsmax, s), min(nsmin, s)
            nnoise += s > 1.0
            ncont += s <= 0.5
    if nsmin == 9.0:
        nsmin = -9.0
    return mins, nun, mx, (cov / tot if tot else 1.0), nsmax, nsmin, nnoise, ncont


NEW_COLS = [f"{p}_{k}" for p in ["tk_a", "tk_b", "atk_a", "atk_b"]
            for k in ["min", "nun", "idfun", "cov", "nsmax", "nsmin", "nnoise", "ncont"]] + \
           ["hn_logdiff", "hn_rel", "hn_prefix", "hn_edit", "hn_nmiss"]


def _chunk(args):
    from rapidfuzz.distance import JaroWinkler
    jw = JaroWinkler.normalized_similarity
    an, bn, aa, ba, anum, bnum = args
    out = np.zeros((len(an), len(NEW_COLS)), np.float32)
    empty = (-1.0, -1, -1.0, -1.0, -9.0, -9.0, -1, -1)
    for i in range(len(an)):
        ta, tb = an[i].split(), bn[i].split()
        sa = [t for t in aa[i].split() if not t.isdigit()]
        sb = [t for t in ba[i].split() if not t.isdigit()]
        f1 = _align(ta, tb, jw, _NS)
        f2 = _align(tb, ta, jw, _NS)
        f3 = _align(sa, sb, jw, _ANS) if sb else empty
        f4 = _align(sb, sa, jw, _ANS) if sb else empty
        f5 = E5._num_feats(anum[i], bnum[i])
        out[i] = (*f1, *f2, *f3, *f4, *f5)
    return out


def noise_table(split, col):
    """ns(t) = log((#R docs with t + 1)/(#S1 docs with t + 1)) - log(N_R/N_S1), per split."""
    s1 = pl.read_parquet(os.path.join(E5.PREP[split], f"{split}_s1.parquet"), columns=[col])[col]
    r = pl.read_parquet(os.path.join(E5.PREP[split], f"{split}_r.parquet"), columns=[col])[col]
    def df(x):
        vc = x.str.split(" ").list.unique().explode().value_counts()
        return dict(zip(vc[vc.columns[0]].to_list(), vc["count"].to_list()))
    c1, c2 = df(s1), df(r)
    base = np.log(len(r) / len(s1))
    return {t: float(np.log((c2.get(t, 0) + 1) / (c1.get(t, 0) + 1)) - base) for t in set(c1) | set(c2) if t}


def stage_feats(args, log):
    for split, src in [("train", E5.FEAT3), ("test", E5.FEAT4T)]:
        out_dir = os.path.join(FEAT7, split)
        os.makedirs(out_dir, exist_ok=True)
        idf = E5.idf_table(split)
        ns = noise_table(split, "name_core")
        ans = noise_table(split, "addr_clean")
        top = sorted(ns.items(), key=lambda kv: -kv[1])[:15]
        log(f"{split}: idf {len(idf)}, name noise vocab {len(ns)}; noisiest name words: "
            + ", ".join(f"{t}:{v:.1f}" for t, v in top))
        s1 = pl.read_parquet(os.path.join(E5.PREP[split], f"{split}_s1.parquet"), columns=["name_core", "addr_clean", "addr_nums"])
        r = pl.read_parquet(os.path.join(E5.PREP[split], f"{split}_r.parquet"), columns=["name_core", "addr_clean", "addr_nums"])
        with Pool(8, initializer=_init, initargs=(idf, ns, ans)) as pool:
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


def use_layer(args):
    """Point exp05's model stages at this feature layer (and an optional test stage-1 feature dir)."""
    E5.FEAT5 = FEAT7
    E5.S2DIR = S2DIR7
    if args.test_feat:
        E5.FEAT4T = os.path.join(CACHE, args.test_feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "train1", "feats2", "train2", "predict"])
    ap.add_argument("--run", default="exp07")
    ap.add_argument("--test_feat", default="", help="test stage-1 feature dir under CACHE (default exp05's feat_v4/test)")
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
    set_determinism(0)
    use_layer(args)
    if args.stage == "feats":
        stage_feats(args, log)
    else:
        {"train1": E5.stage_train1, "feats2": E5.stage_feats2, "train2": E5.stage_train2,
         "predict": E5.stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
