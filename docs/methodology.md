# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** !COYS!  
**Team Members:** Rithvik Achutuni, Gautam Gandhi, Deepak Pandey, Manas Inamdar  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source 2 / Source 3 records to Source 1 entities with a scalable cascade:

1. A learned character-n-gram two-tower embedding blocker with GPU IVF search.
2. A learned pair filter that cuts the candidate set to **≈3.80 candidates per Source 1 entity** (6.59M pairs for 1.73M entities) without measurable recall loss.
3. A two-stage cross-fitted XGBoost matcher. Its uncertain band is re-scored by three fine-tuned cross-encoders and combined by a gradient-boosted stacker that compares each pair with the record's competing entities.
4. Assignment decoding (each record goes to at most one entity), followed by four targeted rules found by label-free count matching (one for the US, three for the unseen country).

**Public leaderboard: 0.98891.**

Our main contributions are:

- **A validation world that reproduces the test's density** (20% of Source 1 entities dropped). Its score matches the leaderboard for the training countries.
- **"Word-role" features, learned per country without labels**, that make the matcher transfer to the unseen country (France). They are complemented by structural rules whose training-country truth we measured on labels.

---

## 2. Methodology

### 2.1 Problem Analysis

**Name noise:**

- legal-suffix changes, abbreviations, digit-for-letter typos (5tar, 8lue);
- stop words dropped; word-order changes;
- acronyms ("PC" for "Pornic Compagnie");
- random brand names at the same address ("Quonex");
- native-script transliterations (Devanagari, Kannada) in India.

**Address noise:**

- street-type abbreviations; "No." / "#" / "N°" prefixes;
- component reordering; house-number typos; missing components;
- French department instead of region.

**Missing fields:** 2.7% of test Source 2/3 records have no address. When such a record's name is shared by several Source 1 entities, it cannot be placed. This single case is **half of the remaining error** in the training countries (Section 5).

**The generator adds words to names in two roles, with a separate vocabulary per country** (measured on the training labels):

- **Noise words** _replace_ a descriptor at the same address, e.g. "Obrien Lion LLC" → "Obrien LLC Services". US/India: center, services, partners. France: fils, associés.
- **Family words** are _appended_ to the full name of a sister company, usually at another house number, e.g. "Falcon Corp" → "Falcon Group". A true match only 0.2% of the time.

**Distractors:**

- sister companies (family words);
- legal-form siblings at a neighbouring number;
- records whose Source 1 entity is absent from the test ("orphans").

**Test density:** the test has 5.8 Source 2/3 records per entity vs 4.7 in training. It behaves like the training data with ~20% of entities removed, so about 40% of test records are distractors.

**Unseen country (France):**

- small name and street vocabulary: 15.2% of French entities share their exact address with another entity, vs 4–5% in US/India;
- its own noise and family vocabulary;
- a France-specific distractor type: sibling organisations ("Gilbert Sport" vs "Gilbert Amis" at the same address).

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned filter + two-stage gradient-boosted classifier + cross-encoder re-scoring (hybrid)

**Core Innovations:**

1. A **dense validation world** that reproduces the test's density: cross-fitted by entity, 20% of entities dropped. It predicts the training-country leaderboard score to within ≈0.0002.
2. **Unlabeled, per-country word-role statistics** as features. For every word, in each split and country:
    - how often it replaces vs is appended;
    - how often it sits at the same house number;
    - how over-represented it is in Source 2/3 names.

    These let the matcher recognise French noise words it has never seen (for example, "fils" swaps: 1% accepted without them, 90% with them).

3. A **compact candidate set** (learned filter) with unchanged recall.
4. A **gradient-boosted stacker with competition ("rival") features**: each cross-encoder score, and the matcher's probability, minus the best score among the same record's other candidate entities. Cross-encoders score pairs independently; the rival features tell the stacker whether a competing entity looks even better (+0.0002 F0.5 on validation, stable across seeds).
5. **Every transfer assumption is tested on a held-out training country before it touches the unseen one.** Several plausible France rules (type alignment, importance weighting, quantile mapping, self-training) were rejected this way.
6. **Label-free count matching.** The true-record noise mix is the same in train and test, so for every fine pair type (name relation × house-number relation × legal form) we compare "accepted pairs per 1,000 entities" on test with "true accepted pairs per 1,000 entities" on validation. An excess marks a test-only distractor type, a deficit a missed match type. This found the US legal-form sibling family and the French acronym, descriptor-sibling and acronym-candidate pockets (Section 4, post-decoding rules).

---

## 3. Candidate Generation (Blocking)

**Blocking keys / method:**

1. **Normalization** (`prep_v3.py`):
    - lower-casing, accent folding and unicode transliteration;
    - legal-suffix canonicalization;
    - digit-for-letter typo repair;
    - street-type, unit and ordinal canonicalization;
    - city-suffix removal;
    - Indian state codes and renamed cities;
    - French department → region;
    - alias splitting ("X doing business as Y");
    - a native-script → Latin token dictionary mined from the training ground truth.
2. **Two-tower embedding blocker** (`exp01_block.py`):
    - hashed character 3-gram + word EmbeddingBag towers for name and address (128-d);
    - trained contrastively from scratch on training pairs, one model per entity fold (cross-fitting).
3. **GPU IVF top-k search per country** (`exp01_block.py search`): four channels.
    - name top-20, address top-20, joint (name + address cosine) top-30;
    - a **reverse** channel: each Source 2/3 record retrieves its top Source 1 entities, so every record reaches a candidate list.
4. **Pruning:** per entity, the top-15 by joint cosine, plus the top-2 entities of each record. This gives ≈18.6 pairs per entity.
5. **Learned pair filter** (`exp15_compact.py`): the stage-1 matcher (cross-fitted on training entities) keeps pairs with p > 0.01 in training countries and p > 0.003 in unseen countries. **This filtered set is `candidate_pairs.tsv` and is exactly what the final matcher scores.**

6. **French acronym candidate channel** (`exp35_acrocand.py`): pruning ranks by name + address cosine, and an acronym record ("CC" for "Caducee Comite") has almost no name similarity, so in dense French streets 20% of the exact acronym matches were pruned away. Records whose name is the initials of exactly one Source 1 entity at the same house number and street are added back as candidates (3,572 pairs).

7. **Final gate** (`exp37_candgate.py`): every stage after the stage-2 matcher only acts on pairs with stage-2 p > 0.01. The cross-encoders and stackers re-score 0.01 < p < 0.99, the unseen-country rules need p > 0.01, and the decoding thresholds are ≥ 0.6. Pairs at or below 0.01 can therefore never be accepted and leave the candidate set; the accepted matches are unchanged.

**Candidate pairs generated (test):** 6,591,700 for 1,732,544 Source 1 entities:

- **3.80 per entity**;
- before the learned filter: 32.2M pairs (18.6 per entity); after the filter 3.95; after the final gate 3.80.

**How true matches are kept** (dense validation world, labels):

| stage                    | candidates per entity | true-pair recall |
| ------------------------ | --------------------- | ---------------- |
| after pruning            | 19.0                  | **98.43%**       |
| after the learned filter | 3.65 (train world)    | **98.42%**       |
| after the final gate     | 3.61 (train world)    | **98.41%** (the dropped true pairs are never accepted) |

- The final F0.5 is unchanged by the filter. A filter that only drops pairs the matcher would never accept costs nothing, while tighter embedding cut-offs lose recall (top-3: 6.0 per entity, recall 97.56%).
- The unseen country gets a lower filter threshold (0.003): no match accepted by the unfiltered pipeline is lost.
- The 1.6% of true pairs that are never retrieved are dominated by records **without an address whose name is shared by several entities** (86.5% address-less). No matcher can place those.

---

## 4. Matching Model

**Features used** (≈142 in stage 2):

- **Name:** rapidfuzz ratio / token-sort / token-set / partial / Jaro-Winkler on normalized and core names; token Jaccard; first-token and suffix agreement; name length; alias match.
    - IDF-weighted token alignment with abbreviation awareness ("frs" ~ "freres").
    - Noise score of unmatched tokens (Source 2/3 over-representation, per split).
- **Address:** fuzzy scores on the normalized address; numeric-token overlap; first house number equal / edit distance / relative difference / prefix; component counts; street-token IDF alignment.
- **Embedding:** name, address and joint cosines; ranks within the entity and within the record; margin to the record's second-best entity.
- **Consensus (exp09):** whether a record's differing word or house number recurs among the entity's other candidates; identical-core cluster size.
- **Word roles (exp13):** replace-vs-append share, same-house-number share, over-representation and frequency of the extra and missing words, computed per split and country without labels.
- **Stage 2:** stage-1 probability competition features (rank / gap / margin within entity and record, expected number of other matches) and consistency with the entity's confident candidates. These are computed only over the filtered candidate set.

**Model type:**

- Two-stage XGBoost (depth 8, early stopping), 2-fold cross-fitted by Source 1 entity, trained in the dense world.
- Re-scoring of the uncertain band (0.01 < p < 0.99) by fine-tuned cross-encoders on "name ; address" text pairs, 2-fold cross-fitted:
    - two **multilingual-e5-small** models (MIT, 118M parameters, frozen embeddings), trained on different bands;
    - one **bert-base-uncased** model (Apache-2.0, 110M parameters) for architecture diversity.
- A **gradient-boosted stacker** (XGBoost, depth 5, 500 rounds, 2-fold cross-fitted) over [logit p, the three CE scores, 12 structural features (house-number agreement, name/address cosines, missing address, name frequency, …), rival features]. A cross-fitted logistic stacker [logit p, CE scores, interactions] is the fallback for the few pairs without every CE score.

| stacking step (dense validation, training countries) | F0.5        |
| ---------------------------------------------------- | ----------- |
| stage-2 matcher alone                                | 0.98858     |
| + cross-encoder 1 (logistic stack)                   | 0.98985     |
| + cross-encoders 2 and 3 (logistic stack)            | 0.99015     |
| GBDT stacker + structural features                   | 0.99032     |
| **+ rival features (final)**                         | **0.99050** |

A fourth cross-encoder (multilingual-e5-base, 278M) reached 0.99054: noise-level, so it is not part of the submission.

- The cross-encoder is used **only for training countries**: on French pairs whose truth the leaderboard established, it was anti-correlated with the truth.

**Unseen countries** (no labels):

1. **Word-role rule:** a single-word swap at the same house number, where the word has a noise role in that country's own statistics, is accepted. Its training-country precision is 99.9% (US) / 98.5% (India).
2. **Post-decoding rules** (each found by label-free count matching; each re-decodes with the same thresholds):
    - **US legal-form siblings** (`exp31_sibprior.py`): same name, the entity without a legal form, the record with one, another house number ("Maid Tavern" 21402 vs "Maid Tavern LLC" 21411). The test contains 5–12× more of them per entity than the validation world, and the excess sits at p ≤ 0.98 / 0.95. Those pairs are demoted (p × 0.5; 9,156 test pairs).
    - **French acronyms** (`exp32_acro.py`): the record name is the initials of the entity name, same house number, unique acronym entity. The only pair type that stays near-certain on a held-out country (93–100% true in every probability bin); 2,515 pairs accepted.
    - **French descriptor siblings** (`exp33_descswap.py`): one descriptor swapped for another at the same house number ("Pessac Club" → "Pessac Ecole") is a sibling organisation. The vocabulary is learned from the country's own records: frequent words that are not over-represented in Source 2/3 names. 2,134 accepted pairs demoted.
    - **French acronym candidates** (`exp35_acrocand.py`): the candidate channel described in Section 3, accepted at p = 0.95.
3. **Rejected: type alignment.** We considered accepting pair types that are 96–99.7% true in the training countries (brand names at the same address, identical names at another number, typo swaps, …) when the model rejects them. Tested on a held-out training country, this _hurt_ (−0.0007 to −0.0042): within a type, the pairs the model rejects are the genuinely doubtful ones, and out-of-country probabilities stay informative. The final pipeline therefore uses the matcher's probability plus the word-role rule only.

**Threshold selection method:**

- Assignment decoding: each Source 2/3 record keeps only its best entity.
- The F0.5-optimal threshold is chosen on the dense out-of-fold predictions: 0.7 for training countries, plus a second threshold of 0.6 for entities with no other match.
- Unseen countries use 0.9, chosen on the cross-country benchmark (train on one country, evaluate on the other): out-of-country probabilities are over-confident.

---

## 5. Results & Error Analysis

- **F0.5 (macro), dense validation world, training countries:** **0.99050**.
- **Public leaderboard: 0.98891.** It decomposes (via a France-emptied diagnostic submission) into ≈0.9905 for US/India and ≈0.978 for France (15% of the test entities).

**Leaderboard progression:** 0.978934 (blocker + matcher) → 0.982907 (token features) → 0.985008 (cross-encoder) → 0.986449 (French word-role rule) → 0.986936 (word-role features) → 0.987891 (GBDT stacker, rival features, compact candidates) → **0.98891** (post-decoding rules).

**Loss decomposition** (validation, oracle fix of each error category; base 0.98842):

| error type                                                              | pairs | F0.5 gain if fixed                                           |
| ----------------------------------------------------------------------- | ----- | ------------------------------------------------------------ |
| **records without an address whose name is shared by several entities** | ≈107K | **+0.0055** (half of all loss, unresolvable by name/address) |
| other true pairs never retrieved                                        | 17K   | +0.0017                                                      |
| other true pairs rejected                                               | 30K   | +0.0020                                                      |
| false positives                                                         | 18K   | +0.0026                                                      |

**Common false positives:**

- records of entities absent from the test (orphans) that carry the same name as a remaining entity;
- sister companies at a neighbouring number with a different legal form;
- brand-name records at an address shared by several businesses;
- in France, sibling organisations at the same address ("X Sport" vs "X Amis").

**Common false negatives:**

- address-less records with common names (ambiguous by construction);
- native-script (Kannada / Devanagari) names with partial addresses;
- brand-name records with partial addresses;
- multi-digit house-number typos.

---

## 6. Conclusion

- A density-matched validation world, cross-fitted two-stage gradient boosting and cross-encoder re-scoring give ≈0.990 F0.5 on the training countries.
- A learned filter keeps the candidate set at ≈4 per entity.
- For the unseen country, the key was to describe _what role a word plays_ instead of _which word it is_. This role is learned per country from unlabeled records, so the matcher and a small set of label-validated structural rules transfer to French data.
- Lesson: measure everything on a validation world whose density matches the test, and test every transfer assumption on held-out countries before trusting it on the unseen one.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`:

- **Entry point:** `run_pipeline.py --data <dataset> --work <scratch> --out <output>`. It runs every step in order and resumes with `--from_step`.
- **Shared code:** `er_common.py` (paths, IO, metric, writer).
- **Preprocessing and blocking:** `prep_v3.py` (and `prep_v1.py`, which it imports); `exp01_block.py` (embedding blocker + IVF search); `exp01_match.py` (pair features).
- **Validation world and features:** `exp03_dense.py` (dense validation world, cross-fitting, decoding); `exp05_tokfeat.py`, `exp07_tokfeat2.py`, `exp09_consensus.py`, `exp13_rolefeat.py` (feature layers and model stages); `exp02_stack.py` (stage-2 features).
- **Compaction and cross-encoders:** `exp15_compact.py` (learned filter + stage 2 on the filtered set); `exp06_crossenc.py` (cross-encoder training and scoring); `exp17_cefill.py` (cross-encoder coverage); `exp10_combine.py` (stacking + decoding).
- **Stacking:** `exp28_gbstack.py` (gradient-boosted stacker with rival features).
- **Unseen countries and decoding:** `exp11_rolerule.py` (word-role rule); `exp19_align.py` (final decoding; its alignment option is disabled).
- **Post-decoding rules:** `exp31_sibprior.py` (US legal-form siblings), `exp32_acro.py` (French acronyms), `exp33_descswap.py` (French descriptor siblings), `exp35_acrocand.py` (French acronym candidate channel).

See `README.md` for the environment, hardware and per-step runtimes. All models are trained from scratch on the provided training data, except the pretrained multilingual-e5-small (MIT) and bert-base-uncased (Apache-2.0) encoders, which are fine-tuned. No external data, lookups or APIs are used. The unlabeled test records are used transductively only for per-split statistics (IDF, word roles).

### B. Additional Results

The full experiment log (≈20 experiments, including negative results and leaderboard probes) is summarised in the project's `EXPERIMENTS.md`.
