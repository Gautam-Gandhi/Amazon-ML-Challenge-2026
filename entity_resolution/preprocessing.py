"""Normalization of names and addresses (applied identically to every source, split and country).

Names: transliteration (anyascii + a native-script word dictionary mined from the training ground truth), junk and
alias removal, domain / hashtag names, honorifics, digit-for-letter typos, abbreviations, legal forms moved to a
separate field (name_sfx) and a core name without legal forms and stop words (name_core).
Addresses: canonical short forms, US / Indian state codes, French regions and departments mapped to one region code,
unit designators and city-type suffixes dropped, ordinals, house numbers extracted (addr_nums).

Output tables (row index = entity index used everywhere downstream):
  {split}_s1 : S1 entities          {split}_r : S2 then S3 records          train_gt : (s1_idx, r_idx) true pairs
"""
import os
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import polars as pl
from anyascii import anyascii

from .data_io import read_tsv
from .utils import LOG, Timer, progress

# ------------------------------------------------------------------------------------------------ dictionaries
SUFFIX_CANON = {
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "llc": "llc", "pllc": "pllc", "llp": "llp", "lp": "lp",
    "pc": "pc", "plc": "plc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa",
    "eurl": "eurl", "snc": "snc", "cie": "cie", "gmbh": "gmbh", "opc": "opc",
}
HONORIFIC_RE = re.compile(r"^\s*(?:m\s*/\s*s|mr|mrs|ms|smt|shri|sri|dr)\b\.?\s+", re.I)
ID_JUNK_RE = re.compile(r"\(\s*id\s*:\s*\d+\s*\)|#\d+\b|\|\s*www\.\S*", re.I)
DOMAIN_RE = re.compile(r"^\s*(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|biz|info|co|us)\s*$", re.I)
SCRIPT_RE = re.compile(r"[Ͱ-￿]")  # anything beyond Latin / Latin-extended = native script

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

# French metropolitan regions -> short code, departments -> their region ("saint" and "st" spellings accepted)
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

ALIAS_RE = re.compile(
    r"^(.*?)\s+(?:formerly\s+known\s+as|also\s+known\s+as|known\s+as|doing\s+business\s+as|trading\s+as|"
    r"f\s*/\s*k\s*/\s*a|fka|formerly|d\s*/\s*b\s*/\s*a|dba|a\s*/\s*k\s*/\s*a|aka|t\s*/\s*a|n[eé]e)"
    r"\s*:?\s+(.+)$", re.I)
SUFFIX = dict(SUFFIX_CANON)
SUFFIX.update({"sci": "sci", "ei": "ei", "eirl": "eirl", "scp": "scp", "selarl": "selarl", "gie": "gie",
               "scop": "scop", "lnc": "inc", "llp": "llp"})
NAME_MAP = {"et": "and", "frs": "freres", "fs": "fils", "cb": "club", "svc": "service", "svcs": "services",
            "mgmt": "management", "intl": "international", "saint": "st", "sainte": "ste",
            "laxmi": "lakshmi", "jay": "jai", "shree": "shri", "sree": "sri"}
NAME_STOP = {"and", "of", "the", "de", "du", "des", "la", "le", "les", "d", "l"}
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})

ADDR = dict(ADDR_CANON)
ADDR.update({"quai": "q", "cours": "crs", "residence": "res", "passage": "psg", "lotissement": "lot",
             "faubourg": "fbg", "turnpike": "tpke", "keralam": "kl", "tg": "ts",
             "calcutta": "kolkata", "bombay": "mumbai", "bengaluru": "bangalore", "madras": "chennai",
             "gurugram": "gurgaon", "poona": "pune", "mysuru": "mysore", "belagavi": "belgaum",
             "cochin": "kochi", "baroda": "vadodara", "trivandrum": "thiruvananthapuram"})
for _i, _w in enumerate(["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                         "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                         "eighteenth", "nineteenth", "twentieth"], start=1):
    ADDR[_w] = f"{_i}th"
ADDR_DROP = {"no", "number", "num", "unit", "apt", "apts", "apartment", "appartement", "app", "appt", "ste", "suite",
             "pmb", "po", "box", "door", "h", "hn", "hno", "flt", "flat", "shop", "fl", "floor", "flr", "etage",
             "city", "cdp", "twp", "township", "town", "village", "borough", "county", "of", "region", "urban",
             "suburban", "divreportingcircle", "district", "dist", "bis", "ter", "null"}
ORD_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
BISTER_RE = re.compile(r"^(\d+)(bis|ter)$")

# native-script dictionaries, set in each worker process
NATIVE_TOK = {}
NATIVE_COMP = {}


def _init_worker(tok_map, comp_map):
    global NATIVE_TOK, NATIVE_COMP
    NATIVE_TOK = tok_map
    NATIVE_COMP = comp_map


def _strip_tok(t):
    return t.strip(".,;:()[]{}\"'|-_/")


def _basic_tokens(s):
    s = s.lower().replace("&", " and ").replace("'", "").replace(".", "")
    return re.sub(r"[^a-z0-9]+", " ", s).split()


def translit_name(s):
    """Word-wise native script -> Latin with the mined dictionary, anyascii otherwise (also strips diacritics)."""
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


def fix_leet(t):
    """Digit-for-letter typo inside a word (5tar, 8lue, temp1e): exactly one digit and >= 3 letters."""
    nd = sum(ch.isdigit() for ch in t)
    if nd == 1 and sum(ch.isalpha() for ch in t) >= 3:
        return t.translate(LEET)
    return t


def normalize_name(raw):
    """-> (name_clean, name_core, name_alias, name_sfx, flags). Flags: 1 native script, 2 domain, 4 hashtag,
    8 alias, 16 id junk removed."""
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
    toks = [NAME_MAP.get(t, t) for t in (fix_leet(t) for t in _basic_tokens(s))]
    clean, core, sfx = [], [], set()
    for t in toks:
        c = SUFFIX.get(t)
        if c is not None:
            clean.append(c)
            sfx.add(c)
        else:
            clean.append(t)
            if t not in NAME_STOP and (not core or core[-1] != t):
                core.append(t)
    if not core:
        core = [t for t in clean if t not in NAME_STOP] or clean[:]
    atoks = _basic_tokens(alias) if alias else []
    return (" ".join(clean), " ".join(core), " ".join(atoks), " ".join(sorted(sfx)), flags)


def normalize_address(raw):
    """-> (addr_clean, addr_nums, addr_ncomp)."""
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
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " " + s + " "
    s = STATE_RE.sub(lambda m: " " + STATE_PHRASES[m.group(1)] + " ", s)
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
        t = ADDR.get(t, t)
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
    seen_fr = set()                       # a region token repeated (department + region) -> keep one
    res2 = []
    for t in res:
        if t in FR_PHR.values():
            if t in seen_fr:
                continue
            seen_fr.add(t)
        res2.append(t)
    return (" ".join(res2), " ".join(nums), ncomp)


def _normalize_chunk(chunk):
    names, addrs = chunk
    return [normalize_name(x) for x in names], [normalize_address(x) for x in addrs]


def normalize_table(df, pool, chunk=50000, desc="normalize"):
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    chunks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    cols = {k: [] for k in ["name_clean", "name_core", "name_alias", "name_sfx", "name_flags",
                            "addr_clean", "addr_nums", "addr_ncomp"]}
    for nn, aa in progress(pool.imap(_normalize_chunk, chunks), desc=desc, total=len(chunks), unit="chunk"):
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


def mine_native_maps(s1, r, gt_pairs):
    """Native-script word -> Latin word dictionary (and address components), from the training ground truth."""
    s1n = s1["business_name"].to_numpy()
    s1a = s1["business_address"].to_numpy()
    rn = r["business_name"].to_numpy()
    ra = r["business_address"].to_numpy()
    si = gt_pairs["s1_idx"].to_numpy()
    ri = gt_pairs["r_idx"].to_numpy()
    tokc = defaultdict(Counter)
    compc = defaultdict(Counter)
    for a, b in zip(si, ri):
        nm = rn[b]
        if SCRIPT_RE.search(nm):
            nt = [_strip_tok(t) for t in nm.split()]
            nt = [t for t in nt if t]
            lt = _basic_tokens(anyascii(ID_JUNK_RE.sub(" ", s1n[a])))
            if len(nt) == len(lt):
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
        if n >= 2 and n / sum(c.values()) >= 0.5:
            tok_map[x] = y
    comp_map = {}
    for x, c in compc.items():
        y, n = c.most_common(1)[0]
        if n >= 3:
            comp_map[x] = y
    LOG.info("native-script dictionary: %d words, %d address components", len(tok_map), len(comp_map))
    return tok_map, comp_map


def run(ws):
    """Stage: raw TSVs -> normalized tables in the workspace."""
    pool = None
    for split in ["train", "test"]:
        d = os.path.join(ws.data_dir, split)
        s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
        r = pl.concat([read_tsv(os.path.join(d, f"{split}_source2.tsv")), read_tsv(os.path.join(d, f"{split}_source3.tsv"))])
        LOG.info("%s: %s S1 entities, %s S2/S3 records", split, f"{s1.height:,}", f"{r.height:,}")
        if split == "train":
            gt = read_tsv(os.path.join(d, "train_ground_truth.tsv"))
            s1_pos = pl.DataFrame({"source1_entity_id": s1["entity_id"], "s1_idx": np.arange(s1.height, dtype=np.int32)})
            r_pos = pl.DataFrame({"rid": r["entity_id"], "r_idx": np.arange(r.height, dtype=np.int32)})
            gtp = (gt.filter(pl.col("matched_entity_ids") != "")
                     .with_columns(pl.col("matched_entity_ids").str.split(",").alias("rid"))
                     .explode("rid")
                     .join(s1_pos, on="source1_entity_id", how="inner")
                     .join(r_pos, on="rid", how="inner")
                     .select("s1_idx", "r_idx"))
            gtp.write_parquet(ws.table("train", "gt"))
            tok_map, comp_map = mine_native_maps(s1, r, gtp)
            del gt, s1_pos, r_pos, gtp
            pool = Pool(max(1, os.cpu_count() - 1), initializer=_init_worker, initargs=(tok_map, comp_map))
        for tag, df in [("s1", s1), ("r", r)]:
            t0 = Timer()
            out = normalize_table(df, pool, desc=f"normalize {split} {'S1' if tag == 's1' else 'S2/S3'}")
            src = pl.lit(1, dtype=pl.Int8) if tag == "s1" else \
                pl.when(pl.col("eid").str.starts_with("S2")).then(2).otherwise(3).cast(pl.Int8)
            out.with_columns(src.alias("src")).write_parquet(ws.table(split, tag))
            LOG.info("normalized %s %s: %s rows in %s", split, "S1" if tag == "s1" else "S2/S3", f"{out.height:,}", t0)
            del out
        del s1, r
    pool.close()
