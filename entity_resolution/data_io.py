"""Reading the challenge TSVs, writing the two output files and running the official validator."""
import os
import sys
import subprocess

import numpy as np
import polars as pl

from .utils import LOG

VALIDATOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "validate_submission.py")


def read_tsv(path, n_rows=None):
    """Read a challenge TSV as all-string columns (no quoting; tabs only)."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, n_rows=n_rows)
    return df.with_columns([pl.col(c).fill_null("") for c in df.columns])


def write_outputs(out_dir, s1_ids, match_s1, match_r, cand_s1, cand_r, r_ids):
    """Write matching_results.tsv and candidate_pairs.tsv (one row per S1 entity, pairs sorted, deterministic bytes).
    s1_ids / r_ids map row indices to entity ids."""
    os.makedirs(out_dir, exist_ok=True)
    s1_ids = np.asarray(s1_ids, dtype=object)
    r_ids = np.asarray(r_ids, dtype=object)

    def _write(path, col, a_s1, a_r):
        df = pl.DataFrame({"s1": a_s1.astype(np.int64), "r": a_r.astype(np.int64)}).unique().sort(["s1", "r"])
        g = df.group_by("s1", maintain_order=True).agg(pl.col("r"))
        lists = {}
        for s1, rs in zip(g["s1"].to_list(), g["r"].to_list()):
            lists[s1] = ",".join(r_ids[np.asarray(rs)])
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{col}\n")
            for i, sid in enumerate(s1_ids):
                f.write(f"{sid}\t{lists.get(i, '')}\n")

    _write(os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids", match_s1, match_r)
    _write(os.path.join(out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", cand_s1, cand_r)


def validate_outputs(out_dir, data_dir):
    """Run the official validator on the output folder. Returns True when the files pass."""
    cmd = [sys.executable, VALIDATOR,
           "--matching", os.path.join(out_dir, "matching_results.tsv"),
           "--candidate", os.path.join(out_dir, "candidate_pairs.tsv"),
           "--test-dir", os.path.join(data_dir, "test")]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lines = [ln.rstrip() for ln in res.stdout.splitlines() if ln.strip()]
    for ln in lines[2:] if lines and "validator" in lines[0] else lines:      # skip the title and the test-dir lines
        LOG.info("validator | %s", ln.strip())
    if res.returncode != 0 and res.stderr.strip():
        LOG.info("validator | %s", res.stderr.strip()[-1500:])
    return res.returncode == 0
