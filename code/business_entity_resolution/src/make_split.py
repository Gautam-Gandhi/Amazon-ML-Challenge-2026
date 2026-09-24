"""Hold out a validation split of Source-1 (reference) entities from the
training set, so blocking (and later, matching) recall can be measured on
data not used to design the pipeline -- per the README's guidance to
"hold out a validation split from the training data and score it yourself".

Only S1 entity_ids are split. S2 and S3 stay intact as the full corpus to
search against in both train and val -- entity resolution recall is about
"did we find this S1 entity's matches in the corpus", not about holding out
S2/S3 rows, and the corpus is shared regardless of which S1 entity is being
queried.

The split is stratified by country so both splits keep the train-time
country mix (US/India), with a fixed seed for reproducibility.

Usage:
    python src/make_split.py
    python src/make_split.py --val-fraction 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import pathlib
import random

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_S1_PATH = REPO_ROOT / "dataset" / "train" / "train_source1.tsv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "splits"


def make_split(s1_path: pathlib.Path, val_fraction: float, seed: int) -> tuple:
    s1 = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"], dtype=str)

    rng = random.Random(seed)
    val_ids, train_ids = [], []
    for country, group in s1.groupby("country"):
        ids = group["entity_id"].tolist()
        rng.shuffle(ids)
        n_val = round(len(ids) * val_fraction)
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    return sorted(train_ids), sorted(val_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-path", type=pathlib.Path, default=DEFAULT_S1_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ids, val_ids = make_split(args.s1_path, args.val_fraction, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_s1_ids.txt").write_text("\n".join(train_ids) + "\n")
    (args.output_dir / "val_s1_ids.txt").write_text("\n".join(val_ids) + "\n")

    print(f"train: {len(train_ids):,} entities")
    print(f"val:   {len(val_ids):,} entities ({len(val_ids) / (len(train_ids) + len(val_ids)):.1%})")
    print(f"written to {args.output_dir}")


if __name__ == "__main__":
    main()
