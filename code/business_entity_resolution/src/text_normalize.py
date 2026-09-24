"""Text normalization for business names and addresses.

Grounded in the noise patterns measured in docs/dataset_description.md:
  - ~7% of Latin-script names carry synthetic diacritic noise (e.g. "Nétwork").
  - ~18-28% of India-labeled S2/S3 names are in genuine non-Latin script
    (Devanagari/Tamil/Gujarati) with no Latin counterpart in the same field.
    These two cases need different handling, so diacritic stripping is gated
    on a script check rather than applied blindly (blind NFKD + ascii-encode
    would delete non-Latin text outright instead of leaving it untouched).
  - Legal-suffix tokens (limited/private/llc/inc/...) are frequent enough to
    blow up naive token blocks, so they're canonicalized to a short form here
    and the blocking layer can choose to exclude them from block keys.
  - Addresses have no fixed component order/schema and mix abbreviated and
    full street-type words (Rd/Road, St/Street) that should collapse to one
    canonical token.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "is_mostly_latin",
    "strip_latin_diacritics",
    "canonicalize_suffixes",
    "normalize_name",
    "normalize_address",
    "tokenize",
    "extract_postal_code",
]

_WHITESPACE_RE = re.compile(r"\s+")

# Unicode block ranges for the non-Latin scripts observed in the dataset
# (Devanagari, Gujarati, Tamil -- see docs/dataset_description.md section
# 4.1), plus other major Indic scripts as a defensive superset. Presence of
# any of these codepoints, not a character-ratio heuristic, is what decides
# "genuine non-Latin script" -- a ratio breaks down on short strings (a
# 3-letter word with one accented letter has a 33% "non-ASCII ratio" despite
# being unambiguously Latin).
_INDIC_SCRIPT_RANGES = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Anchored to the end of the string (trailing punctuation/whitespace
# allowed): a US ZIP or India PIN is the last token of an address when
# present at all, and a bare unanchored \d{5} match instead tends to catch a
# street number that happens to be five digits (e.g. "17560 Ellis Road,
# Tahlequah, OK" has no ZIP at all, but an unanchored regex would wrongly
# extract "17560").
_US_ZIP_RE = re.compile(r"(?<!\d)(\d{5}(?:-\d{4})?)[.,\s]*$")
_IN_PIN_RE = re.compile(r"(?<!\d)(\d{6})[.,\s]*$")

# Variant -> canonical short token. Order-independent: applied as whole-token
# replacement after tokenization, so "Corp." / "Corp" / "Corporation" all
# collapse to "corp" regardless of surrounding punctuation.
_NAME_SUFFIX_MAP = {
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "limited": "ltd",
    "ltd": "ltd",
    "llc": "llc",
    "l.l.c": "llc",
    "llp": "llp",
    "l.l.p": "llp",
    "private": "pvt",
    "pvt": "pvt",
    "company": "co",
    "co": "co",
    "plc": "plc",
    "pc": "pc",
    "corporations": "corp",
}

# Whole-token address abbreviation -> canonical full word. Expanding (rather
# than abbreviating) avoids collisions with common short words.
_ADDRESS_ABBR_MAP = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "dr": "drive",
    "ln": "lane",
    "blvd": "boulevard",
    "ct": "court",
    "cir": "circle",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "ter": "terrace",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "no": "number",
    "nos": "number",
    "opp": "opposite",
    "nr": "near",
}

# Tokens that carry little discriminating power for blocking (legal suffixes
# plus a few filler words seen across both languages of address noise).
STOPWORDS = frozenset(_NAME_SUFFIX_MAP.values()) | {
    "and",
    "the",
    "of",
    "near",
    "opposite",
    "number",
}


def is_mostly_latin(text: str) -> bool:
    """False iff `text` contains at least one character from a non-Latin
    Indic script block (see `_INDIC_SCRIPT_RANGES`).

    Used to gate diacritic stripping: genuine non-Latin script (Devanagari,
    Tamil, Gujarati, ...) must pass through untouched, while Latin text with
    a handful of injected accents (Nétwork, Ínc) should be cleaned. A
    presence check is used rather than a non-ASCII character ratio because
    the ratio is unreliable on short strings (e.g. "ÁND" is 33% non-ASCII by
    character count despite being unambiguously Latin).
    """
    if not text:
        return True
    return not any(
        any(lo <= ord(ch) <= hi for lo, hi in _INDIC_SCRIPT_RANGES) for ch in text
    )


def strip_latin_diacritics(text: str) -> str:
    """Drop combining diacritical marks from mostly-Latin text.

    No-op on genuine non-Latin script because of the `is_mostly_latin` guard
    (a plain NFKD + ascii-encode would instead delete that text entirely).
    """
    if not is_mostly_latin(text):
        return text
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list:
    """Lowercase alphanumeric tokens. Safe on non-Latin text (regex just
    won't match those characters, so it yields whatever Latin/numeric
    fragments are present, e.g. the plot number in an otherwise-Devanagari
    address line)."""
    return _TOKEN_RE.findall(text.lower())


def canonicalize_suffixes(tokens: list, mapping: dict = _NAME_SUFFIX_MAP) -> list:
    return [mapping.get(tok, tok) for tok in tokens]


def normalize_name(raw: str) -> str:
    """Return a cleaned, space-joined name: diacritics stripped (if Latin),
    lowercased, tokenized, legal suffixes canonicalized."""
    if not raw:
        return ""
    cleaned = strip_latin_diacritics(raw).replace("&", " and ")
    tokens = tokenize(cleaned)
    tokens = canonicalize_suffixes(tokens, _NAME_SUFFIX_MAP)
    return " ".join(tokens)


def normalize_address(raw: str) -> str:
    """Return a cleaned, space-joined address: diacritics stripped (if
    Latin), lowercased, tokenized, street-type abbreviations expanded."""
    if not raw:
        return ""
    cleaned = strip_latin_diacritics(raw)
    tokens = tokenize(cleaned)
    tokens = canonicalize_suffixes(tokens, _ADDRESS_ABBR_MAP)
    return " ".join(tokens)


def extract_postal_code(raw_address: str, country: str) -> str:
    """Best-effort postal code extraction. Coverage is low by design of the
    source data (~10% for US ZIP, ~0% for a clean isolated India PIN per
    docs/dataset_description.md) -- treat any hit as a bonus exact-match
    signal, never as a required blocking key. Returns "" when not found.
    """
    if not raw_address:
        return ""
    if country == "US":
        match = _US_ZIP_RE.search(raw_address)
        return match.group(1) if match else ""
    if country == "India":
        match = _IN_PIN_RE.search(raw_address)
        return match.group(1) if match else ""
    return ""
