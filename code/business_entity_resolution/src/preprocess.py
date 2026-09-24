"""Batch preprocessing: apply text_normalize to all six source files.

Reads dataset/{train,test}/{train,test}_source{1,2,3}.tsv and writes one
processed TSV per input file, adding normalized name/address fields plus a
couple of derived flags used downstream by blocking:

  entity_id, country, business_name, business_address   (kept as-is)
  name_norm, addr_norm                                   (see text_normalize)
  postal_code                                             (best-effort, sparse)
  name_is_latin                                           (routing flag)

`name_is_latin` matters because ~18-28% of India-labeled S2/S3 names are in
genuine non-Latin script (Devanagari/Tamil/Gujarati) with no Latin
counterpart, while S1 names are always Latin -- see
docs/dataset_description.md section 4.1. Name-based blocking channels have
zero signal for name_is_latin=False rows and should defer to address-based
channels for them.

Usage:
    python src/preprocess.py
    python src/preprocess.py --dataset-dir /path/to/dataset --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from text_normalize import extract_postal_code, is_mostly_latin, normalize_address, normalize_name

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"

SOURCE_SPECS = [
    ("train", "source1"),
    ("train", "source2"),
    ("train", "source3"),
    ("test", "source1"),
    ("test", "source2"),
    ("test", "source3"),
]

CHUNKSIZE = 500_000


def process_file(input_path: pathlib.Path, output_path: pathlib.Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"rows": 0, "non_latin_name": 0, "postal_code_found": 0, "empty_address": 0}
    t0 = time.time()
    first_chunk = True

    reader = pd.read_csv(
        input_path,
        sep="\t",
        chunksize=CHUNKSIZE,
        keep_default_na=False,
        na_filter=False,
        dtype=str,
    )
    for chunk in reader:
        chunk["name_norm"] = chunk["business_name"].map(normalize_name)
        chunk["addr_norm"] = chunk["business_address"].map(normalize_address)
        chunk["postal_code"] = [
            extract_postal_code(addr, ctry)
            for addr, ctry in zip(chunk["business_address"], chunk["country"])
        ]
        chunk["name_is_latin"] = chunk["business_name"].map(is_mostly_latin)

        stats["rows"] += len(chunk)
        stats["non_latin_name"] += int((~chunk["name_is_latin"]).sum())
        stats["postal_code_found"] += int((chunk["postal_code"] != "").sum())
        stats["empty_address"] += int((chunk["business_address"] == "").sum())

        out_cols = [
            "entity_id",
            "country",
            "business_name",
            "business_address",
            "name_norm",
            "addr_norm",
            "postal_code",
            "name_is_latin",
        ]
        chunk[out_cols].to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False
        print(f"  ... {stats['rows']:,} rows written ({time.time() - t0:.1f}s elapsed)")

    stats["seconds"] = time.time() - t0
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = []
    for split, source in SOURCE_SPECS:
        input_path = args.dataset_dir / split / f"{split}_{source}.tsv"
        output_path = args.output_dir / f"{split}_{source}_processed.tsv"
        print(f"Processing {input_path} -> {output_path}")
        stats = process_file(input_path, output_path)
        summary.append((split, source, stats))

    print("\n=== Summary ===")
    header = f"{'file':<20}{'rows':>10}{'non_latin_name':>16}{'postal_found':>14}{'empty_addr':>12}{'seconds':>10}"
    print(header)
    for split, source, s in summary:
        print(
            f"{split + '_' + source:<20}{s['rows']:>10,}{s['non_latin_name']:>16,}"
            f"{s['postal_code_found']:>14,}{s['empty_address']:>12,}{s['seconds']:>10.1f}"
        )


if __name__ == "__main__":
    main()
