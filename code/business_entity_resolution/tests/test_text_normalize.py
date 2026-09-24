"""Sanity checks for text_normalize, anchored on real noise patterns
observed in the dataset (see docs/dataset_description.md section 4)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from text_normalize import (
    extract_postal_code,
    is_mostly_latin,
    normalize_address,
    normalize_name,
    strip_latin_diacritics,
    tokenize,
)


def test_strip_diacritics_on_synthetic_noise():
    assert strip_latin_diacritics("Nétwork") == "Network"
    assert strip_latin_diacritics("Ínfrastructure") == "Infrastructure"
    assert strip_latin_diacritics("ÁND") == "AND"


def test_diacritic_strip_is_noop_on_non_latin_script():
    devanagari = "राम मार्केटिंग प्राइवेट लिमिटेड"
    assert strip_latin_diacritics(devanagari) == devanagari
    tamil = "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்"
    assert strip_latin_diacritics(tamil) == tamil


def test_is_mostly_latin():
    assert is_mostly_latin("Orelee's Barbershop")
    assert not is_mostly_latin("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert is_mostly_latin("")


def test_normalize_name_collapses_legal_suffix_variants():
    assert normalize_name("Acme Corp") == normalize_name("Acme Corporation")
    assert normalize_name("Sharma & Sons Pvt Ltd") == normalize_name("Sharma and Sons Private Limited")


def test_normalize_name_on_pure_non_latin_is_empty():
    # No Latin/numeric tokens survive -- expected, since blocking must fall
    # back to the address channel for these rows (see dataset_description.md).
    assert normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड") == ""


def test_normalize_address_expands_street_abbreviations():
    assert normalize_address("105 ELM ST, MORGANTON, NC") == normalize_address(
        "105 ELM STREET, MORGANTON, NC"
    )
    assert normalize_address("17560 Ellis Rd, Tahlequah, OK") == normalize_address(
        "17560 Ellis Road, Tahlequah, OK"
    )


def test_normalize_address_preserves_latin_fragments_in_mixed_script():
    addr = "PLOT NO. 74, SECOND FLOOR ZONE-2, M.P. NAGAR, BHOPAL, मध्य प्रदेश"
    norm = normalize_address(addr)
    assert "plot" in norm and "bhopal" in norm
    # Devanagari state name contributes no tokens, which is fine -- the
    # Latin portion of the address still carries the blocking signal.


def test_tokenize_basic():
    assert tokenize("B+ Retail Inc.") == ["b", "retail", "inc"]


def test_extract_postal_code_us():
    assert extract_postal_code("1795 Westchester Drive, High Point, NC 27262", "US") == "27262"
    assert extract_postal_code("No zip here", "US") == ""


def test_extract_postal_code_does_not_match_leading_street_number():
    # Regression: an unanchored 5-digit regex wrongly matched the street
    # number here (17560) as if it were a ZIP -- this address has no ZIP at
    # all, and the real signal only lives at the end of the string.
    assert extract_postal_code("17560 Ellis Road, Tahlequah, OK", "US") == ""


def test_extract_postal_code_india_is_sparse_by_design():
    # India PINs are essentially absent from this dataset (~0% detected per
    # docs/dataset_description.md) -- this asserts the extractor still works
    # mechanically on a well-formed 6-digit PIN when one is present.
    assert extract_postal_code("MG Road, Bengaluru, Karnataka 560001", "India") == "560001"
    assert extract_postal_code("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India") == ""


def test_extract_postal_code_unknown_country_returns_empty():
    assert extract_postal_code("1 Rue de Paris, 75001 Paris", "France") == ""


if __name__ == "__main__":
    import inspect

    module = sys.modules[__name__]
    tests = [obj for name, obj in vars(module).items() if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {test.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
