"""Token features (37 columns): word alignment in both directions for names and addresses, and house-number arithmetic.

Each word looks for its best partner on the other side (exact = 1, otherwise Jaro-Winkler; an abbreviation such as
"frs" ~ "freres" counts 0.9). Per direction: weakest word score, number of unmatched words (< 0.85), highest IDF of an
unmatched word, IDF-weighted coverage, and the noise score of the unmatched words. The noise score of a word is its
over-representation in S2/S3 names compared with S1 names, computed from the split's own records (no labels):
generator-injected words ("services", "groupe") score high, real name words near 0.
"""
import os
import glob
from multiprocessing import Pool

import numpy as np
import polars as pl

from ..utils import LOG, progress
from .pair_features import layer_dir

_IDF, _NS, _ANS = {}, {}, {}

NEW_COLS = [f"{p}_{k}" for p in ["tk_a", "tk_b", "atk_a", "atk_b"]
            for k in ["min", "nun", "idfun", "cov", "nsmax", "nsmin", "nnoise", "ncont"]] + \
           ["hn_logdiff", "hn_rel", "hn_prefix", "hn_edit", "hn_nmiss"]


def _init_worker(idf, ns, ans):
    global _IDF, _NS, _ANS
    _IDF, _NS, _ANS = idf, ns, ans


def _is_abbrev(short, long_):
    """short abbreviates long_: same first letter, len >= 2, letters appear in order (frs ~ freres)."""
    if len(short) < 2 or len(short) >= len(long_) or short[0] != long_[0]:
        return False
    it = iter(long_)
    return all(ch in it for ch in short)


def _align(ta, tb, jw, ns):
    """Tokens of ta vs tb -> (min_sim, n_unmatched, max_idf_unmatched, idf_coverage, max_ns_unmatched,
    min_ns_unmatched, n_unmatched_noise (ns > 1), n_unmatched_content (ns <= 0.5))."""
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


def _house_number_feats(na, nb):
    """First house number: log |difference|, relative difference, prefix/suffix (dropped digit), edit distance,
    and the number of S1 numbers missing from the record."""
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
        f5 = _house_number_feats(anum[i], bnum[i])
        out[i] = (*f1, *f2, *f3, *f4, *f5)
    return out


def idf_table(ws, split):
    """Word IDF over the split's S1 + record name and address words (documents = records)."""
    cols = ["name_core", "addr_clean"]
    s1 = pl.read_parquet(ws.table(split, "s1"), columns=cols)
    r = pl.read_parquet(ws.table(split, "r"), columns=cols)
    n = s1.height + r.height
    idf = {}
    for c in cols:
        toks = pl.concat([s1[c], r[c]]).str.split(" ").list.unique().explode()
        vc = toks.value_counts()
        for t, k in zip(vc[vc.columns[0]].to_list(), vc["count"].to_list()):
            if t:
                idf[t] = min(idf.get(t, 99.0), float(np.log((n + 1) / (k + 1)) + 1))
    return idf


def noise_table(ws, split, col):
    """ns(t) = log((#records with t + 1) / (#S1 with t + 1)) - log(N_records / N_S1), per split."""
    s1 = pl.read_parquet(ws.table(split, "s1"), columns=[col])[col]
    r = pl.read_parquet(ws.table(split, "r"), columns=[col])[col]

    def df(x):
        vc = x.str.split(" ").list.unique().explode().value_counts()
        return dict(zip(vc[vc.columns[0]].to_list(), vc["count"].to_list()))
    c1, c2 = df(s1), df(r)
    base = np.log(len(r) / len(s1))
    return {t: float(np.log((c2.get(t, 0) + 1) / (c1.get(t, 0) + 1)) - base) for t in set(c1) | set(c2) if t}


def build(ws):
    """Stage: token features for every base-feature part (row-aligned)."""
    for split in ["train", "test"]:
        src = layer_dir(ws, "base", split)
        out_dir = layer_dir(ws, "tokens", split)
        idf = idf_table(ws, split)
        ns = noise_table(ws, split, "name_core")
        ans = noise_table(ws, split, "addr_clean")
        s1 = pl.read_parquet(ws.table(split, "s1"), columns=["name_core", "addr_clean", "addr_nums"])
        r = pl.read_parquet(ws.table(split, "r"), columns=["name_core", "addr_clean", "addr_nums"])
        with Pool(8, initializer=_init_worker, initargs=(idf, ns, ans)) as pool:
            for p in sorted(glob.glob(os.path.join(src, "part*.parquet"))):
                k = pl.read_parquet(p, columns=["s1", "r"])
                A = s1[k["s1"].to_numpy()]
                B = r[k["r"].to_numpy()]
                cols = [A["name_core"].to_list(), B["name_core"].to_list(), A["addr_clean"].to_list(),
                        B["addr_clean"].to_list(), A["addr_nums"].to_list(), B["addr_nums"].to_list()]
                step = 20000
                jobs = [tuple(c[i:i + step] for c in cols) for i in range(0, k.height, step)]
                X = np.concatenate(list(progress(pool.imap(_chunk, jobs, chunksize=4), total=len(jobs), unit="chunk",
                                                 desc=f"token features {split} {os.path.basename(p)}")))
                pl.DataFrame({c: X[:, j] for j, c in enumerate(NEW_COLS)}).write_parquet(
                    os.path.join(out_dir, os.path.basename(p)))
                LOG.info("token features %s %s: %s pairs, %d columns", split, os.path.basename(p), f"{k.height:,}",
                         len(NEW_COLS))
