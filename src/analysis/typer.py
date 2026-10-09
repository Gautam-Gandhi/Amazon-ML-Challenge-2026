"""Fine pair typer (name-derivation noise types) for assigned pairs (R's best S1).
Used to find unseen-country fixes that are safe OUT OF COUNTRY (LOCO labels) and to size them on French test."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, re
import numpy as np, polars as pl
from rapidfuzz import fuzz

CACHE = _os.path.join(_WORK, 'data', 'cache')
RUNS = _os.path.join(_WORK, 'runs')
STOP = {"la", "le", "les", "de", "du", "des", "d", "l", "et", "and", "of", "the", "a", "en", "au", "aux", "&"}
DOM_RE = re.compile(r"^(?:www\.)?([a-z0-9\-]+)\.(com|net|org|in|fr|co|biz|info)\b")
ALIAS_RE = re.compile(r"\b(?:doing business as|d\.?b\.?a\.?|formerly known as|formerly|f/k/a|fka|a/k/a|aka|t/a|trading as)\b")


def _concats(core_toks, clean_toks):
    ns = [t for t in core_toks if t not in STOP]
    out = {"".join(core_toks), "".join(clean_toks), "".join(ns)}
    if len(core_toks) >= 2:
        out.add("".join(t[0] for t in core_toks[:-1]) + core_toks[-1])
    return {x for x in out if x}


def ptype(n1_core, n1_clean, r_raw, r_core):
    """name relation type of (S1, R)"""
    A = n1_core.split(); B = r_core.split()
    rl = r_raw.lower().strip()
    rl_ascii = re.sub(r"[^a-z0-9#@\.\s/\-&]", "", rl)
    if ALIAS_RE.search(rl):
        return "alias"
    m = DOM_RE.search(rl_ascii.replace(" ", ""))
    if m:
        stem = re.sub(r"[^a-z0-9]", "", m.group(1))
        cands = _concats(A, n1_clean.split())
        best = max((fuzz.ratio(stem, c) for c in cands), default=0)
        return "domain_m" if best >= 85 else "domain_x"
    if rl.startswith("#") or rl.startswith("@"):
        stem = re.sub(r"[^a-z0-9]", "", rl_ascii)
        pref = {"".join(A[:k]) for k in range(1, len(A) + 1)}
        best = max((fuzz.ratio(stem, c) for c in pref), default=0)
        return "tag_m" if best >= 85 else "tag_x"
    if len(B) == 1 and 2 <= len(B[0]) <= 5 and B[0].isalpha():
        ini = {"".join(t[0] for t in A), "".join(t[0] for t in A if t not in STOP)}
        if B[0] in ini:
            return "acro"
    if A == B:
        return "identical"
    sa, sb = set(A), set(B)
    if sa == sb:
        return "reorder"
    ne, nm, nc = len(sb - sa), len(sa - sb), len(sa & sb)
    if nc == 0:
        return "brand"
    if ne == 0:
        return "drop"
    if nm == 0:
        return "add"
    if ne == 1 and nm == 1:
        a, = sa - sb; b, = sb - sa
        if fuzz.ratio(a, b) >= 75:
            return "typo1"
        return "swap1"
    return "multi"


def typed(pred, split, only_best=True):
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    s1 = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=["country", "name_core", "name_clean", "addr_nums", "addr_clean", "name_sfx"])
    r = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=["name_raw", "name_core", "addr_nums", "addr_clean", "name_sfx"])
    t = pred.with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("rk"))
    if only_best:
        t = t.filter(pl.col("rk") == 1)
    si, ri = t["s1"].to_numpy(), t["r"].to_numpy()
    a_core = s1["name_core"].to_numpy()[si]; a_clean = s1["name_clean"].to_numpy()[si]
    b_raw = r["name_raw"].to_numpy()[ri]; b_core = r["name_core"].to_numpy()[ri]
    types = [ptype(a, c, br, bc) for a, c, br, bc in zip(a_core, a_clean, b_raw, b_core)]
    h1 = s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null("")
    h2 = r["addr_nums"].gather(ri).str.split(" ").list.first().fill_null("")
    noaddr = r["addr_clean"].gather(ri) == ""
    f1 = s1["name_sfx"].gather(si).fill_null(""); f2 = r["name_sfx"].gather(ri).fill_null("")
    t = t.with_columns(pl.Series("ntype", types), pl.Series("country", s1["country"].to_numpy()[si]),
                       h1.alias("h1"), h2.alias("h2"), noaddr.alias("noaddr"), f1.alias("f1"), f2.alias("f2"))
    t = t.with_columns(pl.when(pl.col("noaddr")).then(pl.lit("noaddr")).when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("nonum"))
                       .when(pl.col("h1") == pl.col("h2")).then(pl.lit("samenum")).otherwise(pl.lit("diffnum")).alias("num"),
                       pl.when(pl.col("f1") == pl.col("f2")).then(pl.lit("sfx=")).when((pl.col("f1") == "") | (pl.col("f2") == ""))
                       .then(pl.lit("sfx1")).otherwise(pl.lit("sfx!")).alias("sfx"))
    return t
