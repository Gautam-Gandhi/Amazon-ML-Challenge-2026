"""prep_v3: normalization v3 (supersedes prep_v1 normalization; same output schema and file layout).

Gaps found by token-substitution mining (tools/subst_mining.py) on train GT pairs (US/India) and confident French
test pairs - each one lowers similarity for every pair that contains it:
  names    : alias phrases not split ("X doing business as Y", "formerly known as", "also known as", "nee"/"nee");
             French "et" vs "&"; legal forms SCI/EI/EIRL/SCP/SELARL/GIE/SCOP; digit-for-letter typos (5tar, 8lue,
             5arl, temp1e); abbreviations (frs=freres, fs=fils, cb=club, svc=service, mgmt, intl); stop words
             dropped by the noise ("and", "of", "de", "du", "la"...); spelling variants (laxmi/lakshmi, jay/jai)
  addresses: unit designators (unit/apt/#/pmb/po box/door/h.no/plot/flat/shop/no); city-type suffixes (city, cdp,
             twp, county, town of, borough, village); ordinals (7th/seventh, 2nd/2th); French street types (quai,
             cours, residence, passage, lotissement, faubourg) and bis/ter; French regions/departments (exp04);
             India state codes (TS/TG, Kerala/Keralam) and renamed cities (Calcutta, Bombay, Bengaluru...)
All maps are applied identically to S1 and S2/S3, train and test (generic normalization, no country branching).

Output: <CACHE>/prep_v1/ (directory name kept so all downstream stages run unchanged; VERSION.txt says v3).
Run it in a fresh ER_WORK_DIR (see scripts/run_world_v3.sh).
"""
import os
import re
import sys
import json
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
from anyascii import anyascii

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import read_tsv, DATA, CACHE, Logger  # noqa: E402
import prep_v1 as P1  # noqa: E402  (tables, native-script mining)

# French metropolitan regions -> short code, departments -> their region (same tables as exp04_frnorm, inlined so
# that worker processes do not import torch). Both "saint" and "st" spellings are accepted.
_FR_REGIONS = {
    "hauts de france": "frhdf", "nouvelle aquitaine": "frnaq", "pays de la loire": "frpdl", "ile de france": "fridf",
    "grand est": "frges", "normandie": "frnor", "bretagne": "frbre", "centre val de loire": "frcvl",
    "bourgogne franche comte": "frbfc", "auvergne rhone alpes": "frara", "occitanie": "frocc",
    "provence alpes cote d azur": "frpac", "provence alpes cote dazur": "frpac", "corse": "frcor",
}
_FR_DEPTS = {
    "frhdf": ["nord", "pas de calais", "somme", "oise", "aisne"],
    "frnaq": ["gironde", "landes", "pyrenees atlantiques", "lot et garonne", "dordogne", "charente", "charente maritime",
              "deux sevres", "vienne", "haute vienne", "creuse", "correze"],
    "frpdl": ["loire atlantique", "maine et loire", "vendee", "sarthe", "mayenne"],
    "fridf": ["paris", "seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis",
              "val de marne", "val d oise", "val doise"],
    "frbre": ["finistere", "morbihan", "ille et vilaine", "cotes d armor", "cotes darmor"],
    "frnor": ["calvados", "manche", "orne", "eure", "seine maritime"],
}
FR_PHR = dict(_FR_REGIONS)
for _code, _ds in _FR_DEPTS.items():
    for _d in _ds:
        FR_PHR[_d] = _code
FR_PHR.update({k.replace("saint", "st"): v for k, v in list(FR_PHR.items()) if "saint" in k})
FR_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, FR_PHR), key=len, reverse=True)) + r")\b")

OUT = os.path.join(CACHE, "prep_v1")
VERSION = "prep_v3"

# ------------------------------------------------------------------------------------------------ names
ALIAS_RE3 = re.compile(
    r"^(.*?)\s+(?:formerly\s+known\s+as|also\s+known\s+as|known\s+as|doing\s+business\s+as|trading\s+as|"
    r"f\s*/\s*k\s*/\s*a|fka|formerly|d\s*/\s*b\s*/\s*a|dba|a\s*/\s*k\s*/\s*a|aka|t\s*/\s*a|n[eé]e)"
    r"\s*:?\s+(.+)$", re.I)
SUFFIX3 = dict(P1.SUFFIX_CANON)
SUFFIX3.update({"sci": "sci", "ei": "ei", "eirl": "eirl", "scp": "scp", "selarl": "selarl", "gie": "gie",
                "scop": "scop", "lnc": "inc", "llp": "llp"})
NAME_MAP = {"et": "and", "frs": "freres", "fs": "fils", "cb": "club", "svc": "service", "svcs": "services",
            "mgmt": "management", "intl": "international", "saint": "st", "sainte": "ste",
            "laxmi": "lakshmi", "jay": "jai", "shree": "shri", "sree": "sri"}
NAME_STOP = {"and", "of", "the", "de", "du", "des", "la", "le", "les", "d", "l"}
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def fix_leet(t):
    """Digit-for-letter typo inside a word (5tar, 8lue, temp1e, 5arl): exactly one digit and >= 3 letters."""
    nd = sum(ch.isdigit() for ch in t)
    if nd == 1 and sum(ch.isalpha() for ch in t) >= 3:
        return t.translate(LEET)
    return t


def norm_name3(raw):
    flags = 0
    s = raw
    if P1.SCRIPT_RE.search(s):
        flags |= 1
    s2 = P1.ID_JUNK_RE.sub(" ", s)
    if s2 != s:
        flags |= 16
    s = s2
    alias = ""
    m = ALIAS_RE3.match(s)
    if m:
        alias, s = m.group(1), m.group(2)
        flags |= 8
    s = P1.translit_name(s)
    alias = P1.translit_name(alias) if alias else ""
    st = s.strip()
    dm = P1.DOMAIN_RE.match(st)
    if dm:
        s = dm.group(1)
        flags |= 2
    elif st.startswith("#") and " " not in st:
        s = st[1:]
        flags |= 4
    s = P1.HONORIFIC_RE.sub(" ", s)
    toks = [NAME_MAP.get(t, t) for t in (fix_leet(t) for t in P1._basic_tokens(s))]
    clean, core, sfx = [], [], set()
    for t in toks:
        c = SUFFIX3.get(t)
        if c is not None:
            clean.append(c)
            sfx.add(c)
        else:
            clean.append(t)
            if t not in NAME_STOP and (not core or core[-1] != t):
                core.append(t)
    if not core:
        core = [t for t in clean if t not in NAME_STOP] or clean[:]
    atoks = P1._basic_tokens(alias) if alias else []
    return (" ".join(clean), " ".join(core), " ".join(atoks), " ".join(sorted(sfx)), flags)


# ------------------------------------------------------------------------------------------------ addresses
ADDR3 = dict(P1.ADDR_CANON)
ADDR3.update({"quai": "q", "cours": "crs", "residence": "res", "passage": "psg", "lotissement": "lot",
              "faubourg": "fbg", "turnpike": "tpke", "keralam": "kl", "tg": "ts",
              "calcutta": "kolkata", "bombay": "mumbai", "bengaluru": "bangalore", "madras": "chennai",
              "gurugram": "gurgaon", "poona": "pune", "mysuru": "mysore", "belagavi": "belgaum",
              "cochin": "kochi", "baroda": "vadodara", "trivandrum": "thiruvananthapuram"})
for i, w in enumerate(["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                       "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                       "eighteenth", "nineteenth", "twentieth"], start=1):
    ADDR3[w] = f"{i}th"
ADDR_DROP = {"no", "number", "num", "unit", "apt", "apts", "apartment", "appartement", "app", "appt", "ste", "suite",
             "pmb", "po", "box", "door", "h", "hn", "hno", "flt", "flat", "shop", "fl", "floor", "flr", "etage",
             "city", "cdp", "twp", "township", "town", "village", "borough", "county", "of", "region", "urban",
             "suburban", "divreportingcircle", "district", "dist", "bis", "ter", "null"}
ORD_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
BISTER_RE = re.compile(r"^(\d+)(bis|ter)$")


def norm_addr3(raw):
    if not raw:
        return ("", "", 0)
    raw = P1.N_DEG_RE.sub("no ", raw)
    comps = [c.strip() for c in raw.split(",")]
    ncomp = len([c for c in comps if c])
    out = []
    for c in comps:
        if not c:
            continue
        if P1.SCRIPT_RE.search(c):
            c = P1.NATIVE_COMP.get(c) or anyascii(c)
        elif not c.isascii():
            c = anyascii(c)
        out.append(c)
    s = " , ".join(out).lower()
    s = s.replace("'", "").replace(".", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " " + s + " "
    s = P1.STATE_RE.sub(lambda m: " " + P1.STATE_PHRASES[m.group(1)] + " ", s)
    s = FR_RE.sub(lambda m: FR_PHR[m.group(1)], s)
    toks = s.split()
    res, nums = [], []
    for t in toks:
        mb = BISTER_RE.match(t)
        if mb:
            t = mb.group(1)
        if t.isdigit():
            t = t.lstrip("0") or "0"
            nums.append(t)
            res.append(t)
            continue
        t = ADDR3.get(t, t)
        mo = ORD_RE.match(t)
        if mo:
            t = mo.group(1).lstrip("0") + "th"
        if t in ADDR_DROP:
            continue
        res.append(t)
    for t in res:
        if not t.isdigit() and any(ch.isdigit() for ch in t):
            for d in re.findall(r"\d+", t):
                nums.append(d.lstrip("0") or "0")
    # a region token repeated (department + region) -> keep one
    seen_fr = set()
    res2 = []
    for t in res:
        if t in FR_PHR.values():
            if t in seen_fr:
                continue
            seen_fr.add(t)
        res2.append(t)
    return (" ".join(res2), " ".join(nums), ncomp)


def _work(chunk):
    names, addrs = chunk
    return [norm_name3(x) for x in names], [norm_addr3(x) for x in addrs]


def _init(tok_map, comp_map):
    P1._init(tok_map, comp_map)


def normalize_df(df, pool, log, chunk=50000):
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    chunks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    cols = {k: [] for k in ["name_clean", "name_core", "name_alias", "name_sfx", "name_flags",
                            "addr_clean", "addr_nums", "addr_ncomp"]}
    for nn, aa in pool.imap(_work, chunks):
        for a, b, c, d, e in nn:
            cols["name_clean"].append(a); cols["name_core"].append(b); cols["name_alias"].append(c)
            cols["name_sfx"].append(d); cols["name_flags"].append(e)
        for a, b, c in aa:
            cols["addr_clean"].append(a); cols["addr_nums"].append(b); cols["addr_ncomp"].append(c)
    out = df.select(pl.col("entity_id").alias("eid"), pl.col("country"),
                    pl.col("business_name").alias("name_raw"), pl.col("business_address").alias("addr_raw"))
    return out.with_columns(
        [pl.Series(k, v, dtype=pl.Int8 if k == "name_flags" else (pl.Int16 if k == "addr_ncomp" else pl.Utf8))
         for k, v in cols.items()])


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "VERSION.txt"), "w") as f:
        f.write(VERSION + "\n")
    log = Logger(os.path.join(OUT, "prep_log.txt"))
    pool = None
    for split in ["train", "test"]:
        s1 = read_tsv(os.path.join(DATA, split, f"{split}_source1.tsv"))
        r = pl.concat([read_tsv(os.path.join(DATA, split, f"{split}_source2.tsv")),
                       read_tsv(os.path.join(DATA, split, f"{split}_source3.tsv"))])
        log(f"{split}: S1={s1.height} R={r.height}")
        if split == "train":
            gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
            s1_pos = pl.DataFrame({"source1_entity_id": s1["entity_id"], "s1_idx": np.arange(s1.height, dtype=np.int32)})
            r_pos = pl.DataFrame({"rid": r["entity_id"], "r_idx": np.arange(r.height, dtype=np.int32)})
            gtp = (gt.filter(pl.col("matched_entity_ids") != "")
                     .with_columns(pl.col("matched_entity_ids").str.split(",").alias("rid"))
                     .explode("rid")
                     .join(s1_pos, on="source1_entity_id", how="inner")
                     .join(r_pos, on="rid", how="inner")
                     .select("s1_idx", "r_idx"))
            gtp.write_parquet(os.path.join(OUT, "train_gt.parquet"))
            log(f"GT pairs mapped: {gtp.height}")
            tok_map, comp_map = P1.mine_native_maps(s1, r, gtp, log)
            with open(os.path.join(OUT, "native_tok_map.json"), "w", encoding="utf-8") as f:
                json.dump(tok_map, f, ensure_ascii=False, sort_keys=True)
            with open(os.path.join(OUT, "native_comp_map.json"), "w", encoding="utf-8") as f:
                json.dump(comp_map, f, ensure_ascii=False, sort_keys=True)
            del gt, s1_pos, r_pos, gtp
            pool = Pool(max(1, os.cpu_count() - 1), initializer=_init, initargs=(tok_map, comp_map))
        for tag, df in [("s1", s1), ("r", r)]:
            t = time.time()
            out = normalize_df(df, pool, log)
            src = pl.lit(1, dtype=pl.Int8) if tag == "s1" else \
                pl.when(pl.col("eid").str.starts_with("S2")).then(2).otherwise(3).cast(pl.Int8)
            out = out.with_columns(src.alias("src"))
            out.write_parquet(os.path.join(OUT, f"{split}_{tag}.parquet"))
            log(f"{split}_{tag}: {out.height} rows normalized in {time.time() - t:.0f}s")
            del out
        del s1, r
    pool.close()
    # exp04/exp05/exp06 read the (France-canonicalized) test side from prep_v2: in the v3 world prep already does it
    p2 = os.path.join(CACHE, "prep_v2")
    os.makedirs(p2, exist_ok=True)
    for f in ["test_s1.parquet", "test_r.parquet", "train_s1.parquet", "train_r.parquet", "train_gt.parquet"]:
        dst = os.path.join(p2, f)
        if os.path.exists(dst):
            os.remove(dst)
        os.link(os.path.join(OUT, f), dst)
    log("done")


if __name__ == "__main__":
    main()
