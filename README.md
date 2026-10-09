# Business Entity Resolution at Scale: Amazon ML Challenge 2026

**Top 50 of 32,000+ teams (89,000+ registrations).** Team **!COYS!**: Rithvik Achutuni, Gautam Gandhi, Deepak Pandey, Manas Inamdar.

For every business record in a reference source, find all records in two other noisy sources that describe the same business. The scale is 1.73M reference businesses against about 10M candidate records. Training data covers the US and India; the test set also contains **France**, with no labels at all.

| | |
|---|---|
| Best public leaderboard score (macro F0.5) | **0.988929** (first submission: 0.978934) |
| Candidate set | **3.80 candidates per business** (down from 18.6 after blocking), with 98.4% of true pairs kept |
| Hardware | One laptop: Intel i5-12450H, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) |

📄 **Read the full story:** [docs/project_report.md](docs/project_report.md). It covers how the approach evolved, every experiment, what worked, what didn't, and why.

---

## Approach

```
 Raw TSVs (S1 reference, S2/S3 records)
   │ 1. Normalization: transliteration, legal forms, aliases, native-script dictionary, address canonical forms
   │ 2. Blocking: two-tower char-3-gram embeddings (contrastive) + our own GPU IVF search → ~18.6 candidates per business
   │ 3. 124 pair features: fuzzy scores, word alignment, house-number arithmetic, noise-word scores,
   │                       candidate consensus, per-country word roles (learned without labels)
   │ 4. Stage-1 XGBoost (cross-fitted, trained in a "dense world" that matches test density)
   │ 5. Learned candidate filter → 3.95 per business → stage-2 XGBoost with competition features
   ├── US / India → 6. four fine-tuned cross-encoders on the uncertain band → 7. GBDT stacker with "rival" features
   └── France     → 8. word-role rule for noise words (statistics from the country's own unlabeled data)
   │ 9. Assignment decoding (each record → its best business) + count-matching rules + final candidate gate (3.80)
   ▼
 matching_results.tsv + candidate_pairs.tsv
```

**What made the difference:**
- **A validation world that matches the test.** We drop 20% of training businesses so their records become distractors, as in test. Its score matched the leaderboard for US/India to within 0.0001.
- **Leave-one-country-out testing** (train on the US, test on India, and vice versa), our labelled stand-in for the unseen country.
- **Word roles instead of words.** Whether a word *replaces* another word at the same address (generator noise) or is *appended* at another address (a sister company) is learned per country from unlabeled data. This transfers to French words the model never saw.
- **Label-free count matching.** For each pair pattern, we compare how often it is accepted on test with how often it is true in validation. This found test-only distractor families and missed French acronym matches.
- **A learned candidate filter.** It cut candidates by 79% with no loss in recall or F0.5.
- **Cross-encoders + a gradient-boosted stacker with rival features** for the training countries.

## Results

| Submission | Main change | Public LB |
|---|---|---|
| exp01 | Embedding blocker + 65-feature XGBoost | 0.978934 |
| exp05 | Token-alignment and house-number features | 0.982907 |
| exp10 | Consensus features + cross-encoder (US/India) | 0.985008 |
| exp11b | French word-role rule | 0.986449 |
| exp14r | Word-role features in the model | 0.986936 |
| exp29gf | Compact candidates + 3 cross-encoders + GBDT stacker with rivals | 0.987891 |
| exp35 | Count-matching rules (US siblings, French acronyms, French descriptor siblings) | 0.98891 |
| **exp36** | **+ 4th cross-encoder** | **0.988929** |

Validation (US/India) rose from 0.98490 to 0.99054. The full list of submissions, including probes and failures, is in [docs/experiment_log.md](docs/experiment_log.md).

## Tech stack
Python 3.11 · **polars** · numpy · **rapidfuzz** · **PyTorch 2.9** (CUDA 12.8) · **XGBoost 3.2** (GPU) · scikit-learn · Hugging Face **transformers** · anyascii

Pretrained models (all within the competition's license and size rules, pinned revisions, downloaded automatically):
- `intfloat/multilingual-e5-small` (MIT): cross-encoders 1–2;
- `bert-base-uncased` (Apache-2.0): cross-encoder 3;
- `intfloat/multilingual-e5-base` (MIT): cross-encoder 4.

Everything else (the blocker, the matchers, the stackers) is trained from scratch on the competition data.

---

## Repository layout

```
├── src/
│   ├── run_pipeline.py        ← final pipeline: raw data → both output files
│   ├── er_common.py           ← paths, IO, metric, cross-fitting folds, output writer
│   ├── prep_v1.py, prep_v3.py ← normalization
│   ├── exp01_*.py … exp39_*.py← one file per experiment (pipeline stages and research experiments)
│   ├── utils/validate_submission.py   ← official submission validator
│   ├── tools/                 ← reusable analysis tools (leave-one-country-out, loss breakdown, decoding checks, …)
│   └── analysis/              ← one-off analyses and audits (see analysis/INDEX.md)
├── scripts/                   ← run_expNN.sh: the exact command chain of each experiment
├── docs/
│   ├── project_report.md      ← the full project story
│   ├── methodology.md         ← methodology document submitted with the final package
│   ├── experiment_log.md      ← every experiment: hypothesis, change, result, conclusion
│   ├── france_problem.md      ← everything tried for the unseen country
│   └── dataset_description.md ← our notes on the dataset
└── requirements.txt
```

**Which modules form the final pipeline:**

| Stage | Modules |
|---|---|
| Normalization | `prep_v3.py` (imports `prep_v1.py`) |
| Blocking | `exp01_block.py` (embedding + GPU IVF search), `exp01_match.py` / `exp03_dense.py` (pruning, base features) |
| Pair features | `exp07_tokfeat2.py` (tokens, noise words), `exp09_consensus.py` (consensus), `exp13_rolefeat.py` (word roles) |
| Stage-1 / stage-2 matchers | `exp13_rolefeat.py` (+ `exp05_tokfeat.py`, `exp02_stack.py`), `exp15_compact.py` (candidate filter + stage 2) |
| Cross-encoders | `exp06_crossenc.py`, `exp17_cefill.py` |
| Stacking | `exp10_combine.py` (logistic), `exp28_gbstack.py` (GBDT with rival features) |
| France rule, decoding | `exp11_rolerule.py`, `exp19_align.py` |
| Post-decoding rules | `exp31_sibprior.py`, `exp32_acro.py`, `exp33_descswap.py`, `exp35_acrocand.py`, `exp37_candgate.py` |

The other `exp*.py` files are experiments that were tested and not kept: translating French words, self-training, importance weighting, a zero-shot LLM judge, French type alignment, and so on. [docs/experiment_log.md](docs/experiment_log.md) describes each one and its result.

---

## Reproducing the submission

### 1. Environment (Windows, Linux or macOS; an NVIDIA GPU is required)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
On Windows without symlink rights, set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` before the first run (the pretrained models are downloaded from the Hugging Face hub on first use).

### 2. Data
Place the competition data in `dataset/` (it is not part of this repository):
```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

### 3. Run
```bash
cd src
python run_pipeline.py --data ../dataset --work ../work --out ../output
```
- **Outputs:** `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The validator runs automatically at every decoding step.
- **Caches:** every step caches its results under `--work` (about 40 GB at peak).
- **Resume or partial runs:** resume an interrupted run with `--from_step <name>`. List the steps with `python run_pipeline.py --list`, and stop early with `--to_step <name>`.
- **Runtime:** about 11 hours on the laptop above. The long steps are the embedding models, the training-side search, and the four cross-encoders.
- **Determinism:** every entry point fixes its seeds and uses deterministic GPU kernels, so reruns on the same hardware and software reproduce the same files. Other GPUs may differ in the 4th–5th decimal. XGBoost results depend on the thread count; we used the default.
- **Memory:** 16 GB RAM is enough because the steps run one at a time. Avoid running other heavy jobs alongside.

### Running individual experiments
Each `scripts/run_expNN.sh` reproduces one experiment's command chain. These scripts expect the virtual environment at `.venv/` (Windows layout `.venv/Scripts/python.exe`; adjust `PY=` on Linux/macOS) and write caches to `work/` (override with `ER_WORK_DIR`). Most experiments build on the caches of the main pipeline, so run the pipeline first. The analysis scripts in `src/analysis/` read the same caches.

---

## Team
Rithvik Achutuni · Gautam Gandhi · Deepak Pandey · Manas Inamdar
