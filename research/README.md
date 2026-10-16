# Research code

This folder keeps the code exactly as it was written during the competition: one file per experiment, in the order the
ideas were tried. The final pipeline in [`entity_resolution/`](../entity_resolution) was assembled from it and produces
the same output files; use that package to run the solution. This folder is kept as the record of how each result in
[docs/experiment_log.md](../docs/experiment_log.md) was obtained.

```
research/
├── src/
│   ├── run_pipeline.py          ← the original end-to-end driver (caches every step; --from_step / --to_step / --list)
│   ├── er_common.py             ← paths, IO, metric, cross-fitting folds, output writer
│   ├── prep_v1.py, prep_v3.py   ← normalization
│   ├── exp01_*.py … exp39_*.py  ← one file per experiment (pipeline stages and research experiments)
│   ├── utils/validate_submission.py
│   ├── tools/                   ← reusable analysis tools (leave-one-country-out, loss breakdown, decoding checks, …)
│   └── analysis/                ← one-off analyses and audits (see analysis/INDEX.md)
└── scripts/                     ← run_expNN.sh: the exact command chain of each experiment
```

## Which experiments form the final pipeline

| Stage | Experiment modules | Final package |
|---|---|---|
| Normalization | `prep_v3.py` (imports `prep_v1.py`) | `preprocessing.py` |
| Blocking | `exp01_block.py`, `exp01_match.py` / `exp03_dense.py` | `blocking/` |
| Pair features | `exp07_tokfeat2.py`, `exp09_consensus.py`, `exp13_rolefeat.py` | `features/` |
| Stage-1 / stage-2 matchers | `exp13_rolefeat.py` (+ `exp05_tokfeat.py`, `exp02_stack.py`), `exp15_compact.py` | `models/gradient_boosting.py` |
| Cross-encoders | `exp06_crossenc.py`, `exp17_cefill.py` | `models/cross_encoder.py` |
| Stacking | `exp10_combine.py`, `exp28_gbstack.py` | `models/stacking.py` |
| France rule, decoding | `exp11_rolerule.py`, `exp19_align.py` | `postprocessing/` |
| Post-decoding rules | `exp31_sibprior.py`, `exp32_acro.py`, `exp33_descswap.py`, `exp35_acrocand.py`, `exp37_candgate.py` | `postprocessing/` |

The other `exp*.py` files are experiments that were tested and not kept: translating French words, self-training,
importance weighting, a zero-shot LLM judge, French type alignment, a 4th cross-encoder, and so on. The experiment log
describes each one and its result.

## Running it

```bash
cd research/src
python run_pipeline.py --data ../../dataset --work ../../work --out ../../output
```

Every step caches its results under `--work` (about 40 GB), so a run can resume with `--from_step <name>`. The default
configuration is the submitted one (exp35, 3 cross-encoders, LB 0.98891); `USE_CE4 = True` in `run_pipeline.py` adds the
4th cross-encoder (exp36, LB 0.988929).

Each `scripts/run_expNN.sh` reproduces one experiment's command chain. The scripts expect the virtual environment at
`.venv/` in the repository root (Windows layout `.venv/Scripts/python.exe`; adjust `PY=` on Linux/macOS) and write caches
to `work/` (override with `ER_WORK_DIR`). Most experiments build on the caches of the main pipeline, so run it first. The
analysis scripts in `src/analysis/` read the same caches.
