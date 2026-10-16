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
   ├── US / India → 6. three fine-tuned cross-encoders on the uncertain band → 7. GBDT stacker with "rival" features
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
| **exp35** | **Count-matching rules (US siblings, French acronyms, French descriptor siblings): final submitted code** | **0.98891** |
| exp36 | + 4th cross-encoder (best score, not packaged) | 0.988929 |

Validation (US/India) rose from 0.98490 to 0.99050.

**Why the final code is exp35 and not exp36:** the 4th cross-encoder (multilingual-e5-base, 278M parameters) gained only +0.00004 on validation and +0.00002 on the leaderboard, within run-to-run noise, while adding about 4 hours of runtime and the largest model of the pipeline. We submitted the simpler three-cross-encoder pipeline. The full list of submissions, including probes and failures, is in [docs/experiment_log.md](docs/experiment_log.md).

## Tech stack
Python 3.11 · **polars** · numpy · **rapidfuzz** · **PyTorch 2.9** (CUDA 12.8) · **XGBoost 3.2** (GPU) · scikit-learn · Hugging Face **transformers** · anyascii

Pretrained models (all within the competition's license and size rules, pinned revisions, downloaded automatically):
- `intfloat/multilingual-e5-small` (MIT): cross-encoders 1–2;
- `bert-base-uncased` (Apache-2.0): cross-encoder 3;
- `intfloat/multilingual-e5-base` (MIT): cross-encoder 4, tested in exp30/exp36 only (not used by the final code).

Everything else (the blocker, the matchers, the stackers) is trained from scratch on the competition data.

---

## Repository layout

```
├── main.py                        ← entry point: raw data → both output files
├── entity_resolution/             ← the final pipeline
│   ├── config.py                  ← every setting (seeds, model sizes, thresholds, rule parameters)
│   ├── pipeline.py                ← the ordered list of stages; each stage runs in its own process
│   ├── data_io.py                 ← reading the TSVs, writing and validating the output files
│   ├── preprocessing.py           ← normalization of names and addresses
│   ├── blocking/
│   │   ├── embedding_model.py     ← hashed char-3-gram two-tower encoder, contrastive training
│   │   ├── vector_search.py       ← GPU IVF nearest-neighbour search
│   │   └── candidate_generation.py← search channels, pruning, dense validation world
│   ├── features/
│   │   ├── pair_features.py       ← 65 base features (fuzzy scores, numbers, legal forms, ranks)
│   │   ├── token_features.py      ← 37 word-alignment, noise-word and house-number features
│   │   ├── consensus_features.py  ← 9 features over a business's other candidates
│   │   ├── role_features.py       ← 13 per-country word-role features (no labels)
│   │   └── competition_features.py← features comparing a pair with its rival candidates
│   ├── models/
│   │   ├── validation.py          ← dense validation world, labels, F0.5 threshold search
│   │   ├── gradient_boosting.py   ← cross-fitted XGBoost matchers, candidate filter
│   │   ├── cross_encoder.py       ← cross-encoder training and scoring on the uncertain band
│   │   └── stacking.py            ← logistic stack and GBDT stack with rival features
│   └── postprocessing/
│       ├── rules.py               ← French word-role rule and the count-matching rules
│       └── decoding.py            ← assignment decoding, candidate gate, output files
├── utils/validate_submission.py   ← official submission validator
├── research/                      ← the original experiment-by-experiment code (see research/README.md)
├── docs/
│   ├── project_report.md          ← the full project story
│   ├── methodology.md             ← methodology document submitted with the final package
│   ├── experiment_log.md          ← every experiment: hypothesis, change, result, conclusion
│   ├── france_problem.md          ← everything tried for the unseen country
│   └── dataset_description.md     ← our notes on the dataset
└── requirements.txt
```

---

## Running the pipeline

### 1. Environment (Windows, Linux or macOS; an NVIDIA GPU is required)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
The pretrained models are downloaded from the Hugging Face hub on first use (pinned revisions). On Windows without symlink rights, set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

### 2. Data
Place the competition data in `dataset/` (it is not part of this repository):
```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

### 3. Run
```bash
python main.py --data dataset --output output
```
- **Outputs:** `output/matching_results.tsv` and `output/candidate_pairs.tsv`, checked with the official validator at the end.
- **Progress:** the console shows the current phase and stage, progress bars inside each stage (normalization, blocker epochs, search, feature parts, boosting rounds, cross-encoder batches), the time of every stage, and a summary table with the time per stage and the total at the end.
- **Intermediate data:** normalized tables, models, candidates and feature layers are kept in a temporary folder (about 40 GB at peak) that is deleted when the run ends. Use `--temp-dir <folder>` to place it on a drive with enough space.
- **Runtime:** about 7 hours on the laptop above. The long stages are the embedding blocker, the training-side search and the three cross-encoders.
- **Memory:** 16 GB RAM is enough because the stages run one at a time, each in its own process. Avoid running other heavy jobs alongside.
- **Determinism:** every stage fixes its seeds and uses deterministic GPU kernels, so reruns on the same hardware and software give identical files. Other GPUs may differ in the 4th–5th decimal.

---

## Team
Rithvik Achutuni · Gautam Gandhi · Deepak Pandey · Manas Inamdar
