"""Shared helpers: logging and progress bars, the temporary workspace, determinism, cross-fitting folds and the metric."""
import os
import sys
import json
import time
import logging

import numpy as np
from tqdm import tqdm

LOG = logging.getLogger("entity_resolution")


class _ProgressAwareHandler(logging.Handler):
    """Writes log lines above any active progress bar instead of breaking it."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging():
    if not LOG.handlers:
        h = _ProgressAwareHandler()
        h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        LOG.addHandler(h)
        LOG.setLevel(logging.INFO)
        LOG.propagate = False
    return LOG


def progress(iterable=None, desc="", total=None, unit="it", **kwargs):
    """Progress bar for the steps inside a stage. It disappears when done (the stage logs a summary line instead) and is
    disabled when the output is not a terminal, so log files stay clean."""
    return tqdm(iterable, desc=f"          {desc}", total=total, unit=unit, leave=False, dynamic_ncols=True,
                mininterval=0.5, file=sys.stderr, disable=None, **kwargs)


def format_duration(seconds):
    if seconds < 10:
        return f"{seconds:.1f}s"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class Workspace:
    """Folder for the intermediate data of one run (tables, hashed features, models, candidates, feature layers,
    predictions). It is created inside a temporary directory and deleted when the pipeline finishes."""

    def __init__(self, root, data_dir):
        self.root = root
        self.data_dir = data_dir

    def path(self, *parts):
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def dir(self, *parts):
        p = os.path.join(self.root, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    def table(self, split, kind):
        """Normalized tables: kind in {s1, r, gt}."""
        return self.path("tables", f"{split}_{kind}.parquet")

    def save_json(self, obj, *parts):
        with open(self.path(*parts), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=float)

    def load_json(self, *parts):
        with open(os.path.join(self.root, *parts), encoding="utf-8") as f:
            return json.load(f)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    @property
    def seconds(self):
        return time.time() - self.t0

    def __str__(self):
        return format_duration(self.seconds)


def set_determinism(seed=0):
    """Reproducible GPU/CPU training on the same machine: seeds python/numpy/torch, deterministic cuBLAS workspace,
    deterministic cuDNN and torch algorithms (warn-only for ops without a deterministic kernel)."""
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


def s1_fold(n, k=2):
    """Deterministic fold id per S1 row index, shared by every model for honest cross-fitting."""
    h = (np.arange(n, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(12345)) % np.uint64(2**32)
    return ((h >> np.uint64(16)) % np.uint64(k)).astype(np.int8)


def macro_f05(pred_s1, pred_r, gt_s1, gt_r, n_s1, beta=0.5, return_per_entity=False):
    """Macro F-beta over S1 entities 0..n_s1-1 (singletons: empty prediction -> 1, any prediction -> 0)."""
    b2 = beta * beta
    n_pred = np.bincount(pred_s1, minlength=n_s1).astype(np.float64)
    n_gt = np.bincount(gt_s1, minlength=n_s1).astype(np.float64)
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
