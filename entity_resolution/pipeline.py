"""The full pipeline as an ordered list of stages, grouped in phases.

Each stage runs in its own child process: it starts from a fresh random state (seeded), and its memory is released
when it ends, which keeps the peak RAM within a 16 GB machine. Stages exchange data through a temporary workspace.
"""
import os
import time
import warnings
import importlib
import multiprocessing as mp

from .utils import Timer, Workspace, format_duration, set_determinism, setup_logging

NB, PF = "Normalization and blocking", "Pair features"
TM, MA = "Token matcher and cross-encoder 1", "Matchers"
CE, SD = "Cross-encoders", "Stacking, rules and decoding"

# (phase, stage, "module:function", keyword arguments)
STAGES = [
    (NB, "Normalize records", "preprocessing:run", {}),
    (NB, "Hash name and address features", "blocking.embedding_model:compute_hashed_features", {}),
    (NB, "Train blocker (fold 0)", "blocking.embedding_model:train_blocker", {"fold_id": 0}),
    (NB, "Train blocker (fold 1)", "blocking.embedding_model:train_blocker", {"fold_id": 1}),
    (NB, "Search candidates (train)", "blocking.candidate_generation:search_candidates", {"split": "train"}),
    (NB, "Search candidates (test)", "blocking.candidate_generation:search_candidates", {"split": "test"}),
    (PF, "Prune candidates + base features (test)", "features.pair_features:build_test_features", {}),
    (PF, "Prune candidates + base features (train, dense world)", "features.pair_features:build_train_features", {}),
    (PF, "Token features", "features.token_features:build", {}),
    (PF, "Consensus features", "features.consensus_features:build", {}),
    (PF, "Word-role features", "features.role_features:build", {}),
    (TM, "Token matcher, stage 1", "models.gradient_boosting:train_token_matcher_stage1", {}),
    (TM, "Token matcher, stage-2 features", "models.gradient_boosting:build_token_matcher_stage2_features", {}),
    (TM, "Token matcher, stage 2", "models.gradient_boosting:train_token_matcher_stage2", {}),
    (TM, "Cross-encoder 1: uncertain band", "models.cross_encoder:build_band", {"name": "ce1", "base_model": "token_matcher"}),
    (TM, "Cross-encoder 1: training and scoring", "models.cross_encoder:train", {"name": "ce1"}),
    (MA, "Stage-1 matcher", "models.gradient_boosting:train_stage1", {}),
    (MA, "Stage-1 probabilities", "models.gradient_boosting:predict_stage1", {}),
    (MA, "Candidate filter + stage-2 features", "models.gradient_boosting:build_stage2_features", {}),
    (MA, "Stage-2 matcher", "models.gradient_boosting:train_stage2", {}),
    (CE, "Cross-encoder 1: fill the stage-2 band", "models.cross_encoder:fill_ce1", {}),
    (CE, "Cross-encoder 2: uncertain band", "models.cross_encoder:build_band", {"name": "ce2", "base_model": "stage2"}),
    (CE, "Cross-encoder 2: training and scoring", "models.cross_encoder:train", {"name": "ce2"}),
    (CE, "Cross-encoder 3: uncertain band", "models.cross_encoder:build_band", {"name": "ce3", "base_model": "stage2"}),
    (CE, "Cross-encoder 3: training and scoring", "models.cross_encoder:train", {"name": "ce3"}),
    (SD, "Logistic stack", "models.stacking:logistic_stack", {}),
    (SD, "GBDT stack", "models.stacking:gbdt_stack", {}),
    (SD, "Rules, decoding and output files", "postprocessing.decoding:finalize", {"output_dir": None}),
]

WIDTH = 92


def _run_stage(index, root, data_dir, output_dir):
    setup_logging()
    # library notices that do not affect the results (polars API deprecations, XGBoost CPU-input prediction fallback)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
    set_determinism(0)
    ws = Workspace(root, data_dir)
    _, _, target, kwargs = STAGES[index]
    module, function = target.split(":")
    fn = getattr(importlib.import_module(f"{__package__}.{module}"), function)
    kwargs = {k: (output_dir if k == "output_dir" else v) for k, v in kwargs.items()}
    fn(ws, **kwargs)


def _line(ch="-"):
    return ch * WIDTH


def _output_stats(path):
    """(S1 entities, pairs) of an output file."""
    rows = pairs = 0
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            rows += 1
            ids = line.rstrip("\n").split("\t")[1]
            pairs += ids.count(",") + 1 if ids else 0
    return rows, pairs


def _summary(log, times, total, failed=None):
    log.info(_line("="))
    if failed is None:
        log.info("  Pipeline finished in %s", format_duration(total))
    else:
        log.info("  Pipeline FAILED at stage %d (%s) after %s", failed + 1, STAGES[failed][1], format_duration(total))
    log.info(_line())
    log.info("  %3s  %-58s %12s %7s", "#", "Stage", "Time", "Share")
    phase = None
    for i, (ph, name, _, _) in enumerate(STAGES):
        if i >= len(times):
            break
        if ph != phase:
            phase = ph
            log.info("       %s", ph.upper())
        log.info("  %3d  %-58s %12s %6.1f%%", i + 1, name, format_duration(times[i]), 100 * times[i] / max(total, 1e-9))
    log.info(_line())
    log.info("  %3s  %-58s %12s", "", "Total", format_duration(total))
    log.info(_line("="))


def run(root, data_dir, output_dir):
    log = setup_logging()
    n = len(STAGES)
    log.info(_line("="))
    log.info("  Business entity resolution pipeline (Amazon ML Challenge 2026, team !COYS!)")
    log.info(_line("="))
    log.info("  data       %s", data_dir)
    log.info("  output     %s", output_dir)
    log.info("  workspace  %s (temporary)", root)
    log.info("  stages     %d in %d phases", n, len({p for p, _, _, _ in STAGES}))
    log.info("  started    %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    ctx = mp.get_context("spawn")
    total = Timer()
    times = []
    phase = None
    for i, (ph, name, _, _) in enumerate(STAGES):
        if ph != phase:
            phase = ph
            log.info("")
            log.info(_line("="))
            log.info("  %s", ph.upper())
            log.info(_line("="))
        log.info("[%2d/%d] %s", i + 1, n, name)
        t = Timer()
        p = ctx.Process(target=_run_stage, args=(i, root, data_dir, output_dir))
        p.start()
        p.join()
        times.append(t.seconds)
        if p.exitcode != 0:
            _summary(log, times, total.seconds, failed=i)
            raise RuntimeError(f"stage '{name}' failed (exit code {p.exitcode})")
        log.info("[%2d/%d] %s: done in %s  (pipeline %s, %d%% of stages)", i + 1, n, name, t, total, 100 * (i + 1) // n)
        log.info(_line())
    log.info("")
    _summary(log, times, total.seconds)
    for f in ["matching_results.tsv", "candidate_pairs.tsv"]:
        path = os.path.join(output_dir, f)
        rows, pairs = _output_stats(path)
        log.info("  %-22s %s S1 entities, %s pairs (%.2f per entity)", f, f"{rows:,}", f"{pairs:,}", pairs / max(rows, 1))
    log.info("  written to %s", output_dir)
    log.info(_line("="))
