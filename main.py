"""Business entity resolution: from the raw challenge TSVs to matching_results.tsv and candidate_pairs.tsv.

    python main.py --data dataset --output output

Intermediate data (normalized tables, models, candidates, feature layers) lives in a temporary folder that is deleted
at the end; only the output folder remains. See README.md for the requirements and the pipeline description.
"""
import os
import sys
import shutil
import argparse
import tempfile

from entity_resolution.utils import setup_logging


def main():
    ap = argparse.ArgumentParser(description="Business entity resolution pipeline (Amazon ML Challenge 2026)")
    ap.add_argument("--data", default="dataset", help="folder with train/ and test/ (default: dataset)")
    ap.add_argument("--output", default="output", help="folder for the two output files (default: output)")
    ap.add_argument("--temp-dir", default=None, help="where the temporary workspace is created (needs ~40 GB free; "
                                                     "default: the system temporary folder)")
    ap.add_argument("--keep-temp", action="store_true", help="keep the temporary workspace (for debugging)")
    args = ap.parse_args()
    log = setup_logging()
    data_dir = os.path.abspath(args.data)
    output_dir = os.path.abspath(args.output)
    for f in ["train/train_source1.tsv", "train/train_source2.tsv", "train/train_source3.tsv",
              "train/train_ground_truth.tsv", "test/test_source1.tsv", "test/test_source2.tsv", "test/test_source3.tsv"]:
        if not os.path.exists(os.path.join(data_dir, f)):
            sys.exit(f"missing input file: {os.path.join(data_dir, f)}")
    from entity_resolution.pipeline import run       # imported here: worker processes re-import this file
    root = tempfile.mkdtemp(prefix="entity_resolution_", dir=args.temp_dir)
    try:
        run(root, data_dir, output_dir)
    finally:
        if args.keep_temp:
            log.info("temporary workspace kept: %s", root)
        else:
            shutil.rmtree(root, ignore_errors=True)
            log.info("temporary workspace deleted")


if __name__ == "__main__":
    main()
