"""Recall-measurement harness for blocking channels.

Reads one candidate-pairs file per channel (long format: source1_entity_id,
candidate_entity_id -- as written by scripts under src/channels/) plus
train_ground_truth.tsv, and reports, evaluated only on the held-out
validation split (data/splits/val_s1_ids.txt, made by make_split.py):

  - per-channel recall (micro: fraction of ground-truth pairs found) and
    macro (average per-entity recall across non-singleton entities)
  - per-channel UNIQUE contribution: how much overall recall would drop if
    this channel were removed from the union -- this is what tells you
    whether a channel is pulling its weight or is redundant with the others
    (the "redundancy-positive blocking evaluation" from Papadakis et al.'s
    survey, see docs/dataset_description.md and the blocking research notes)
  - candidate volume and an approximate reduction ratio against the naive
    same-country brute-force pair count
  - the overall (unioned) blocking recall -- the number that sets the
    ceiling for the final matching model

This measures RECALL only. It says nothing about precision -- that's the
final matching model's job, and over-generating candidates here is fine as
long as recall keeps climbing (see docs/dataset_description.md section 6).

Usage:
    python src/measure_recall.py
    python src/measure_recall.py --candidates-dir data/candidates --channels exact_name,exact_address
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_GROUND_TRUTH = REPO_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
DEFAULT_CANDIDATES_DIR = REPO_ROOT / "data" / "candidates"
DEFAULT_VAL_IDS = REPO_ROOT / "data" / "splits" / "val_s1_ids.txt"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "reports" / "blocking_recall_train.txt"


def load_val_ids(path: pathlib.Path) -> set:
    return set(path.read_text().split())


def load_ground_truth(path: pathlib.Path, val_ids: set) -> dict:
    """s1_entity_id -> frozenset of true match ids, restricted to val_ids."""
    gt = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_id, matches = row
            if s1_id not in val_ids:
                continue
            gt[s1_id] = frozenset(matches.split(",")) if matches else frozenset()
    return gt


def discover_channels(candidates_dir: pathlib.Path, split: str = "train") -> list:
    prefix = f"{split}_"
    return sorted(
        p.stem[len(prefix):] for p in candidates_dir.glob(f"{prefix}*.tsv")
    )


def load_channel(path: pathlib.Path, val_ids: set) -> dict:
    """s1_entity_id -> set of candidate ids, restricted to val_ids. Also
    returns the unfiltered row count (candidate volume over the full run,
    not just the val subset) via the second return value."""
    candidates = defaultdict(set)
    total_rows = 0
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for s1_id, cand_id in reader:
            total_rows += 1
            if s1_id in val_ids:
                candidates[s1_id].add(cand_id)
    return candidates, total_rows


def recall_stats(gt: dict, candidates: dict) -> dict:
    """Micro recall, macro recall, and per-entity found/total, over
    non-singleton entities only (singletons have nothing to recall)."""
    total_gt_pairs = 0
    found_pairs = 0
    per_entity_recalls = []
    singleton_ids = [s1 for s1, matches in gt.items() if not matches]
    triggered_singletons = 0

    for s1_id, true_matches in gt.items():
        cand = candidates.get(s1_id, frozenset())
        if not true_matches:
            if cand:
                triggered_singletons += 1
            continue
        hit = len(true_matches & cand)
        total_gt_pairs += len(true_matches)
        found_pairs += hit
        per_entity_recalls.append(hit / len(true_matches))

    return {
        "micro_recall": found_pairs / total_gt_pairs if total_gt_pairs else float("nan"),
        "macro_recall": sum(per_entity_recalls) / len(per_entity_recalls)
        if per_entity_recalls
        else float("nan"),
        "found_pairs": found_pairs,
        "total_gt_pairs": total_gt_pairs,
        "fully_covered_entities": sum(1 for r in per_entity_recalls if r == 1.0),
        "non_singleton_entities": len(per_entity_recalls),
        "singleton_entities": len(singleton_ids),
        "triggered_singletons": triggered_singletons,
    }


def union_candidates(channel_dicts: dict, exclude: str = None) -> dict:
    union = defaultdict(set)
    for name, candidates in channel_dicts.items():
        if name == exclude:
            continue
        for s1_id, cand_ids in candidates.items():
            union[s1_id] |= cand_ids
    return union


def country_corpus_sizes(processed_dir: pathlib.Path) -> dict:
    """{country: n_s2 + n_s3} from the processed train files, for the
    reduction-ratio estimate."""
    sizes = defaultdict(int)
    for source in ("source2", "source3"):
        df = pd.read_csv(
            processed_dir / f"train_{source}_processed.tsv",
            sep="\t",
            usecols=["country"],
            dtype=str,
        )
        for country, count in df["country"].value_counts().items():
            sizes[country] += int(count)
    return dict(sizes)


def val_country_counts(processed_dir: pathlib.Path, val_ids: set) -> dict:
    df = pd.read_csv(
        processed_dir / "train_source1_processed.tsv",
        sep="\t",
        usecols=["entity_id", "country"],
        dtype=str,
    )
    df = df[df["entity_id"].isin(val_ids)]
    return df["country"].value_counts().to_dict()


def format_report(
    channel_dicts: dict,
    channel_volumes: dict,
    gt: dict,
    reduction_ratio_info: dict,
) -> str:
    lines = []
    lines.append("=== Per-channel recall (measured on validation split) ===")
    header = f"{'channel':<16}{'micro_recall':>13}{'macro_recall':>13}{'unique_contrib':>15}{'pairs(all-train)':>18}"
    lines.append(header)

    full_union = union_candidates(channel_dicts)
    full_stats = recall_stats(gt, full_union)

    for name in sorted(channel_dicts):
        stats = recall_stats(gt, channel_dicts[name])
        without = union_candidates(channel_dicts, exclude=name)
        without_stats = recall_stats(gt, without)
        unique_contrib = full_stats["found_pairs"] - without_stats["found_pairs"]
        lines.append(
            f"{name:<16}{stats['micro_recall']:>13.4f}{stats['macro_recall']:>13.4f}"
            f"{unique_contrib:>15,}{channel_volumes.get(name, 0):>18,}"
        )

    lines.append("")
    lines.append("=== Union (overall blocking recall) ===")
    lines.append(f"micro recall (pair completeness): {full_stats['micro_recall']:.4f}")
    lines.append(f"macro recall (avg per non-singleton entity): {full_stats['macro_recall']:.4f}")
    lines.append(
        f"non-singleton entities fully covered: {full_stats['fully_covered_entities']:,} "
        f"/ {full_stats['non_singleton_entities']:,} "
        f"({full_stats['fully_covered_entities'] / full_stats['non_singleton_entities']:.1%})"
    )
    lines.append(
        f"singleton entities with >=1 spurious candidate: {full_stats['triggered_singletons']:,} "
        f"/ {full_stats['singleton_entities']:,} "
        f"({full_stats['triggered_singletons'] / full_stats['singleton_entities']:.1%})"
    )

    if reduction_ratio_info:
        lines.append("")
        lines.append("=== Volume vs. naive same-country brute force ===")
        lines.append(f"naive same-country pairs: {reduction_ratio_info['naive_pairs']:,}")
        lines.append(f"union candidate pairs (val subset): {reduction_ratio_info['union_pairs']:,}")
        lines.append(f"reduction ratio: {reduction_ratio_info['reduction_ratio']:.4%}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=pathlib.Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--candidates-dir", type=pathlib.Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--val-ids", type=pathlib.Path, default=DEFAULT_VAL_IDS)
    parser.add_argument("--processed-dir", type=pathlib.Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--channels", type=str, default=None, help="comma-separated; default auto-discovers all")
    parser.add_argument("--report-path", type=pathlib.Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--skip-reduction-ratio", action="store_true")
    args = parser.parse_args()

    val_ids = load_val_ids(args.val_ids)
    print(f"loaded {len(val_ids):,} validation entity ids")

    gt = load_ground_truth(args.ground_truth, val_ids)
    print(f"loaded ground truth for {len(gt):,} validation entities")

    channel_names = (
        args.channels.split(",") if args.channels else discover_channels(args.candidates_dir)
    )
    if not channel_names:
        raise SystemExit(f"no channel files found in {args.candidates_dir} (expected train_*.tsv)")

    channel_dicts = {}
    channel_volumes = {}
    for name in channel_names:
        path = args.candidates_dir / f"train_{name}.tsv"
        candidates, total_rows = load_channel(path, val_ids)
        channel_dicts[name] = candidates
        channel_volumes[name] = total_rows
        print(f"loaded channel '{name}': {total_rows:,} pairs total, {len(candidates):,} val entities touched")

    reduction_ratio_info = {}
    if not args.skip_reduction_ratio:
        corpus_sizes = country_corpus_sizes(args.processed_dir)
        val_counts = val_country_counts(args.processed_dir, val_ids)
        naive_pairs = sum(val_counts.get(c, 0) * corpus_sizes.get(c, 0) for c in val_counts)
        full_union = union_candidates(channel_dicts)
        union_pairs = sum(len(v) for v in full_union.values())
        reduction_ratio_info = {
            "naive_pairs": naive_pairs,
            "union_pairs": union_pairs,
            "reduction_ratio": union_pairs / naive_pairs if naive_pairs else float("nan"),
        }

    report = format_report(channel_dicts, channel_volumes, gt, reduction_ratio_info)
    print("\n" + report)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report + "\n")
    print(f"\nreport written to {args.report_path}")


if __name__ == "__main__":
    main()
