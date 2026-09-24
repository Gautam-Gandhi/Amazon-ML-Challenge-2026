"""Exact-match blocking channels: group S1/S2/S3 by (country, normalized
name) and by (country, normalized address), and emit every same-key pair as
a candidate. This is the cheapest channel in the union in
docs/dataset_description.md's blocking plan and the first one implemented,
so it also doubles as the first real input to the recall-measurement
harness (src/measure_recall.py).

Country partitioning is applied unconditionally per the verified finding in
docs/dataset_description.md section 3 (zero cross-country matches in
7.6M ground-truth training pairs).

Empty normalized keys are never indexed: ~9% of S2/India and ~5% of
S3/India names normalize to "" (non-Latin script with no Latin/numeric
tokens -- see docs/dataset_description.md section 4.1), and indexing "" as
an ordinary key would collide every such record into one giant, useless
block instead of producing zero candidates for that channel (which is the
correct outcome -- these rows need the address channel or a future
script-aware channel, not this one).

Usage:
    python src/channels/exact_match.py
    python src/channels/exact_match.py --split train   # default; train S1 only
    python src/channels/exact_match.py --split test
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "candidates"


def _load(split: str, source: str, processed_dir: pathlib.Path) -> pd.DataFrame:
    path = processed_dir / f"{split}_{source}_processed.tsv"
    return pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id", "country", "name_norm", "addr_norm"],
        keep_default_na=False,
        na_filter=False,
        dtype=str,
    )


def _exact_match_pairs(s1: pd.DataFrame, other: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Inner-join S1 against an S2 or S3 frame on (country, key_col),
    dropping rows where the key is empty on either side."""
    s1_keyed = s1[s1[key_col] != ""][["entity_id", "country", key_col]]
    other_keyed = other[other[key_col] != ""][["entity_id", "country", key_col]]
    merged = s1_keyed.merge(
        other_keyed, on=["country", key_col], suffixes=("_s1", "_other")
    )
    return merged[["entity_id_s1", "entity_id_other"]].rename(
        columns={"entity_id_s1": "source1_entity_id", "entity_id_other": "candidate_entity_id"}
    )


def build_channel(split: str, processed_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    s1 = _load(split, "source1", processed_dir)
    s2 = _load(split, "source2", processed_dir)
    s3 = _load(split, "source3", processed_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for key_col, channel_name in [("name_norm", "exact_name"), ("addr_norm", "exact_address")]:
        pairs = pd.concat(
            [_exact_match_pairs(s1, s2, key_col), _exact_match_pairs(s1, s3, key_col)],
            ignore_index=True,
        ).drop_duplicates()
        out_path = output_dir / f"{split}_{channel_name}.tsv"
        pairs.to_csv(out_path, sep="\t", index=False)
        print(f"{channel_name}: {len(pairs):,} pairs -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--processed-dir", type=pathlib.Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build_channel(args.split, args.processed_dir, args.output_dir)


if __name__ == "__main__":
    main()
