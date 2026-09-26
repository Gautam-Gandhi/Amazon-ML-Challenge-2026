"""Shared, stable helpers for all experiments: paths, IO, metric, submission writing, logging.

Keep this file backward compatible: experiments import it, so only ADD functions here.
"""
import os
import sys
import time
import json
import subprocess

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
# Paths can be overridden with environment variables (used by the packaged submission code):
#   ER_DATA_DIR  : folder containing train/ and test/ TSVs      (default <ROOT>/dataset)
#   ER_WORK_DIR  : folder for caches (data/cache) and runs/      (default <ROOT>)
#   ER_VALIDATOR : path to utils/validate_submission.py         (default <ROOT>/utils/validate_submission.py)
DATA = os.environ.get("ER_DATA_DIR", os.path.join(ROOT, "dataset"))
_WORK = os.environ.get("ER_WORK_DIR", ROOT)
CACHE = os.path.join(_WORK, "data", "cache")
RUNS = os.path.join(_WORK, "runs")
VALIDATOR = os.environ.get("ER_VALIDATOR", os.path.join(ROOT, "utils", "validate_submission.py"))


def read_tsv(path, n_rows=None):
    """Read a challenge TSV as all-string columns (no quoting; tabs only)."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, n_rows=n_rows)
    return df.with_columns([pl.col(c).fill_null("") for c in df.columns])


class Logger:
    """Tee stdout to a log file and time every step."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        line = f"[{time.time() - self.t0:8.1f}s | {rss_gb():5.1f}GB] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        return float("nan")


def macro_f05(pred_s1, pred_r, gt_s1, gt_r, n_s1, beta=0.5, return_per_entity=False):
    """Macro F-beta over S1 entities 0..n_s1-1.

    pred_* / gt_* are parallel int arrays of (s1_idx, r_idx) pairs (duplicates must not exist).
    Singletons: empty prediction -> 1.0, any prediction -> 0.0.
    """
    b2 = beta * beta
    n_pred = np.bincount(pred_s1, minlength=n_s1).astype(np.float64)
    n_gt = np.bincount(gt_s1, minlength=n_s1).astype(np.float64)
    # true positives via hashing pairs into int64 keys
    kp = pred_s1.astype(np.int64) * (1 << 32) + pred_r.astype(np.int64)
    kg = gt_s1.astype(np.int64) * (1 << 32) + gt_r.astype(np.int64)
    hit = np.isin(kp, kg)
    tp = np.bincount(pred_s1[hit], minlength=n_s1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tp / n_pred, 0.0)
        r = np.where(n_gt > 0, tp / n_gt, 0.0)
        f = np.where(tp > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    f = np.where((n_gt == 0) & (n_pred == 0), 1.0, f)
    if return_per_entity:
        return f
    return float(f.mean())


def write_outputs(out_dir, s1_ids, match_s1, match_r, cand_s1, cand_r, r_ids):
    """Write matching_results.tsv and candidate_pairs.tsv.

    s1_ids: array of S1 entity id strings (index = s1_idx); r_ids: array of S2/S3 id strings (index = r_idx).
    """
    os.makedirs(out_dir, exist_ok=True)
    s1_ids = np.asarray(s1_ids, dtype=object)
    r_ids = np.asarray(r_ids, dtype=object)

    def _write(path, col, a_s1, a_r):
        # deterministic output: unique pairs sorted by (s1, r) -> identical bytes on every regeneration
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


def validate_outputs(out_dir, log=print):
    """Run the official validator on out_dir (skipped with a warning if the validator script is not found)."""
    if not os.path.exists(VALIDATOR):
        log(f"validator not found at {VALIDATOR}; skipping validation")
        return True
    cmd = [sys.executable, VALIDATOR,
           "--matching", os.path.join(out_dir, "matching_results.tsv"),
           "--candidate", os.path.join(out_dir, "candidate_pairs.tsv"),
           "--test-dir", os.path.join(DATA, "test")]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log("validator:", res.stdout.strip()[-2000:], res.stderr.strip()[-1000:])
    return res.returncode == 0


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def s1_fold(n, k=2):
    """Deterministic fold id per S1 row index (shared by all stages for honest cross-fitting)."""
    h = (np.arange(n, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(12345)) % np.uint64(2**32)
    return ((h >> np.uint64(16)) % np.uint64(k)).astype(np.int8)


def set_determinism(seed=0):
    """Make GPU/CPU training reproducible run-to-run on the same machine. Call before any CUDA work.
    Seeds python/numpy/torch, forces cuBLAS deterministic workspace and torch deterministic algorithms
    (warn_only: ops without a deterministic kernel emit a warning instead of failing)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
