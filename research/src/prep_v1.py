"""prep_v1: one-time preprocessing of train + test -> data/cache/prep_v1/*.parquet

Outputs per split (train/test):
  {split}_s1.parquet : S1 records, row index == s1_idx
  {split}_r.parquet  : S2 then S3 records, row index == r_idx
  train_gt.parquet   : (s1_idx, r_idx) ground-truth pairs
  native_tok_map.json / native_comp_map.json : mined native-script -> latin mappings (from train GT only)

Normalized columns:
  name_clean : full normalized name (lowercase ascii, legal suffixes canonicalized)
  name_core  : name without legal suffixes / honorifics / leading 'the', consecutive dups removed
  name_alias : the random alias part of "X f/k/a Y" style names (real name Y goes to core)
  name_sfx   : sorted canonical legal-suffix codes
  name_flags : bit flags 1=native script, 2=domain, 4=hashtag, 8=alias, 16=id/number junk removed
  addr_clean : normalized address tokens (abbreviations -> canonical short forms, states -> codes)
  addr_nums  : numeric tokens in order (leading zeros stripped)
  addr_ncomp : number of comma components in raw address
"""
import os
import re
import sys
import json
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import polars as pl
from anyascii import anyascii

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import read_tsv, DATA, CACHE, Logger  # noqa: E402

OUT = os.path.join(CACHE, "prep_v1")

# ----------------------------------------------------------------------------------------------
# Dictionaries (generic normalization knowledge; applied universally, never branched on country)
# ----------------------------------------------------------------------------------------------
SUFFIX_CANON = {
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "llc": "llc", "pllc": "pllc", "llp": "llp", "lp": "lp",
    "pc": "pc", "plc": "plc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa",
    "eurl": "eurl", "snc": "snc", "cie": "cie", "gmbh": "gmbh", "opc": "opc",
}
HONORIFIC_RE = re.compile(r"^\s*(?:m\s*/\s*s|mr|mrs|ms|smt|shri|sri|dr)\b\.?\s+", re.I)
ALIAS_RE = re.compile(
    r"^(.*?)\s+(?:f\s*/\s*k\s*/\s*a|fka|formerly|d\s*/\s*b\s*/\s*a|dba|a\s*/\s*k\s*/\s*a|aka|t\s*/\s*a|trading\s+as)"
    r"\s*:?\s+(.+)$", re.I)
ID_JUNK_RE = re.compile(r"\(\s*id\s*:\s*\d+\s*\)|#\d+\b|\|\s*www\.\S*", re.I)
DOMAIN_RE = re.compile(r"^\s*(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|biz|info|co|us)\s*$", re.I)
SCRIPT_RE = re.compile(r"[Ͱ-￿]")  # anything beyond Latin/Latin-ext = native script

ADDR_CANON = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave", "avenu": "ave",
    "drive": "dr", "drv": "dr", "court": "ct", "lane": "ln", "trail": "trl", "boulevard": "blvd", "bd": "blvd",
    "boul": "blvd", "bld": "blvd", "place": "pl", "parkway": "pkwy", "highway": "hwy", "circle": "cir",
    "terrace": "ter", "square": "sq", "suite": "ste", "apartment": "apt", "apartments": "apts", "floor": "fl",
    "flr": "fl", "near": "nr", "opposite": "opp", "building": "bldg", "bldng": "bldg", "rue": "r",
    "impasse": "imp", "chemin": "ch", "route": "rte", "allee": "all", "saint": "st", "sainte": "ste",
    "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt", "fort": "ft", "point": "pt",
    "number": "no", "num": "no", "flat": "flt", "sector": "sec",
    "township": "twp", "crossing": "xing", "expressway": "expy",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "goa", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "jammu & kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "ladakh": "la", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "lakshadweep": "ld", "andaman and nicobar islands": "an",
}
N_DEG_RE = re.compile(r"(?i)\bn\s*[°º]")  # "N°" / "Nº" (French numero) -> "no"
STATE_PHRASES = {**US_STATES, **IN_STATES}
STATE_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in STATE_PHRASES), key=len, reverse=True)) + r")\b")

# globals populated in workers
NATIVE_TOK = {}
NATIVE_COMP = {}


def _init(tok_map, comp_map):
    global NATIVE_TOK, NATIVE_COMP
    NATIVE_TOK = tok_map
    NATIVE_COMP = comp_map


def _strip_tok(t):
    return t.strip(".,;:()[]{}\"'|-_/")


def translit_name(s):
    """Token-wise native->latin using mined dictionary, anyascii fallback (also strips diacritics)."""
    if not SCRIPT_RE.search(s):
        return anyascii(s) if not s.isascii() else s
    out = []
    for t in s.split():
        if SCRIPT_RE.search(t):
            k = _strip_tok(t)
            out.append(NATIVE_TOK.get(k) or anyascii(t))
        else:
            out.append(anyascii(t) if not t.isascii() else t)
    return " ".join(out)


def _basic_tokens(s):
    s = s.lower().replace("&", " and ").replace("'", "").replace(".", "")
    return re.sub(r"[^a-z0-9]+", " ", s).split()


def norm_name(raw):
    flags = 0
    s = raw
    if SCRIPT_RE.search(s):
        flags |= 1
    s2 = ID_JUNK_RE.sub(" ", s)
    if s2 != s:
        flags |= 16
    s = s2
    alias = ""
    m = ALIAS_RE.match(s)
    if m:
        alias, s = m.group(1), m.group(2)
        flags |= 8
    s = translit_name(s)
    alias = translit_name(alias) if alias else ""
    st = s.strip()
    dm = DOMAIN_RE.match(st)
    if dm:
        s = dm.group(1)
        flags |= 2
    elif st.startswith("#") and " " not in st:
        s = st[1:]
        flags |= 4
    s = HONORIFIC_RE.sub(" ", s)
    toks = _basic_tokens(s)
    clean, core, sfx = [], [], set()
    for t in toks:
        c = SUFFIX_CANON.get(t)
        if c is not None:
            clean.append(c)
            sfx.add(c)
        else:
            clean.append(t)
            if not core or core[-1] != t:
                core.append(t)
    if core and core[0] == "the" and len(core) > 1:
        core = core[1:]
    if not core:
        core = clean[:] if clean else []
    atoks = _basic_tokens(alias) if alias else []
    return (" ".join(clean), " ".join(core), " ".join(atoks), " ".join(sorted(sfx)), flags)


def norm_addr(raw):
    if not raw:
        return ("", "", 0)
    raw = N_DEG_RE.sub("no ", raw)
    comps = [c.strip() for c in raw.split(",")]
    ncomp = len([c for c in comps if c])
    out = []
    for c in comps:
        if not c:
            continue
        if SCRIPT_RE.search(c):
            c = NATIVE_COMP.get(c) or anyascii(c)
        elif not c.isascii():
            c = anyascii(c)
        out.append(c)
    s = " , ".join(out).lower()
    s = s.replace("'", "").replace(".", " ")
    s = STATE_RE.sub(lambda m: " " + STATE_PHRASES[m.group(1)] + " ", s)
    toks = re.sub(r"[^a-z0-9]+", " ", s).split()
    res, nums = [], []
    for t in toks:
        if t == "null":
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
            nums.append(t)
            res.append(t)
            continue
        t = ADDR_CANON.get(t, t)
        res.append(t)
    # numbers glued to letters (e.g. 12b, b404) -> numeric part also as a number token
    for t in toks:
        if not t.isdigit() and any(ch.isdigit() for ch in t):
            for d in re.findall(r"\d+", t):
                nums.append(d.lstrip("0") or "0")
    return (" ".join(res), " ".join(nums), ncomp)


def _work(chunk):
    names, addrs = chunk
    nn = [norm_name(x) for x in names]
    aa = [norm_addr(x) for x in addrs]
    return nn, aa


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
    out = df.select(
        pl.col("entity_id").alias("eid"), pl.col("country"),
        pl.col("business_name").alias("name_raw"), pl.col("business_address").alias("addr_raw"))
    out = out.with_columns(
        [pl.Series(k, v, dtype=pl.Int8 if k in ("name_flags",) else (pl.Int16 if k == "addr_ncomp" else pl.Utf8))
         for k, v in cols.items()])
    return out


# ----------------------------------------------------------------------------------------------
# Mining native-script -> latin mappings from train ground truth
# ----------------------------------------------------------------------------------------------
def mine_native_maps(s1, r, gt_pairs, log):
    """gt_pairs: DataFrame(s1_idx, r_idx). Returns (token_map, component_map)."""
    s1n = s1["business_name"].to_numpy()
    s1a = s1["business_address"].to_numpy()
    rn = r["business_name"].to_numpy()
    ra = r["business_address"].to_numpy()
    si = gt_pairs["s1_idx"].to_numpy()
    ri = gt_pairs["r_idx"].to_numpy()

    tokc = defaultdict(Counter)
    compc = defaultdict(Counter)
    n_aligned = n_native = 0
    for a, b in zip(si, ri):
        nm = rn[b]
        if SCRIPT_RE.search(nm):
            n_native += 1
            nt = [_strip_tok(t) for t in nm.split()]
            nt = [t for t in nt if t]
            lt = _basic_tokens(anyascii(ID_JUNK_RE.sub(" ", s1n[a])))
            if len(nt) == len(lt):
                n_aligned += 1
                for x, y in zip(nt, lt):
                    if SCRIPT_RE.search(x):
                        tokc[x][y] += 1
        ad = ra[b]
        if ad and SCRIPT_RE.search(ad):
            s1c = [c.strip().lower() for c in s1a[a].split(",") if c.strip()]
            for c in ad.split(","):
                c = c.strip()
                if c and SCRIPT_RE.search(c):
                    for y in s1c:
                        compc[c][y] += 1
    tok_map = {}
    for x, c in tokc.items():
        y, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= 2 and n / tot >= 0.5:
            tok_map[x] = y
    comp_map = {}
    for x, c in compc.items():
        y, n = c.most_common(1)[0]
        tot = sum(c.values())
        # the component co-occurs with every S1 component; state is the dominant one
        if n >= 3:
            comp_map[x] = y
    log(f"native names in GT pairs: {n_native}, aligned (same #tokens): {n_aligned}, "
        f"token map size: {len(tok_map)}, component map size: {len(comp_map)}")
    return tok_map, comp_map


def main():
    os.makedirs(OUT, exist_ok=True)
    log = Logger(os.path.join(OUT, "prep_log.txt"))
    tok_map = comp_map = None
    pool = None
    splits = sys.argv[1].split(",") if len(sys.argv) > 1 else ["train", "test"]
    for split in splits:
        s1 = read_tsv(os.path.join(DATA, split, f"{split}_source1.tsv"))
        s2 = read_tsv(os.path.join(DATA, split, f"{split}_source2.tsv"))
        s3 = read_tsv(os.path.join(DATA, split, f"{split}_source3.tsv"))
        r = pl.concat([s2, s3])
        del s2, s3
        log(f"{split}: S1={s1.height} R={r.height}")
        if pool is None and split != "train":
            with open(os.path.join(OUT, "native_tok_map.json"), encoding="utf-8") as f:
                tok_map = json.load(f)
            with open(os.path.join(OUT, "native_comp_map.json"), encoding="utf-8") as f:
                comp_map = json.load(f)
            pool = Pool(max(1, os.cpu_count() - 1), initializer=_init, initargs=(tok_map, comp_map))
        if split == "train":
            # ground truth -> index pairs
            gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
            s1_pos = pl.DataFrame({"source1_entity_id": s1["entity_id"],
                                   "s1_idx": np.arange(s1.height, dtype=np.int32)})
            r_pos = pl.DataFrame({"rid": r["entity_id"], "r_idx": np.arange(r.height, dtype=np.int32)})
            gtp = (gt.filter(pl.col("matched_entity_ids") != "")
                     .with_columns(pl.col("matched_entity_ids").str.split(",").alias("rid"))
                     .explode("rid")
                     .join(s1_pos, on="source1_entity_id", how="inner")
                     .join(r_pos, on="rid", how="inner")
                     .select("s1_idx", "r_idx"))
            log(f"GT pairs mapped: {gtp.height}")
            gtp.write_parquet(os.path.join(OUT, "train_gt.parquet"))
            del gt, s1_pos, r_pos
            tok_map, comp_map = mine_native_maps(s1, r, gtp, log)
            with open(os.path.join(OUT, "native_tok_map.json"), "w", encoding="utf-8") as f:
                json.dump(tok_map, f, ensure_ascii=False)
            with open(os.path.join(OUT, "native_comp_map.json"), "w", encoding="utf-8") as f:
                json.dump(comp_map, f, ensure_ascii=False)
            pool = Pool(max(1, os.cpu_count() - 1), initializer=_init, initargs=(tok_map, comp_map))
        for tag, df in [("s1", s1), ("r", r)]:
            t = time.time()
            out = normalize_df(df, pool, log)
            if tag == "r":
                out = out.with_columns(
                    pl.when(pl.col("eid").str.starts_with("S2")).then(2).otherwise(3).cast(pl.Int8).alias("src"))
            else:
                out = out.with_columns(pl.lit(1, dtype=pl.Int8).alias("src"))
            out.write_parquet(os.path.join(OUT, f"{split}_{tag}.parquet"))
            log(f"{split}_{tag}: {out.height} rows normalized in {time.time() - t:.0f}s")
            del out
        del s1, r
    pool.close()
    log("done")


if __name__ == "__main__":
    main()
