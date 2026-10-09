# Experiment Log — Amazon ML Challenge 2026 (Business Entity Resolution)

**Metric:** macro F0.5 per S1 entity (precision-weighted; singletons score 1 only if the prediction is empty).
**Validation ("OOF"):** 2-fold cross-fit by S1 over the *whole* train world. Blocking runs over the full train S2/S3 pool, exactly like test. From exp03 on, validation uses a *dense world* (see exp03), because the normal world proved optimistic.
**Target:** LB > 0.99 (user, 09-27; almost all of the top 50 are ≈0.99). **Current best LB:** **0.987891** (exp29gf, 09-27). Submission limit: 5/day.

## Scoreboard

| Exp | Idea (one line) | OOF (normal world) | OOF (dense world k80) | Public LB | Status |
|---|---|---|---|---|---|
| **exp33** | exp32 + **French descriptor-swap demotion** (same house number, one S1 descriptor swapped for another, e.g. "Pessac Club" → "Pessac Ecole"; label-free vocabulary of 43 words; 2,134 accepted pairs demoted) | – | = | _probe_ | siblings (LB: rejected ones ~100% false); expected +0.0002..+0.0003 over exp32 |
| **exp32** | exp31 + **French acronym rule** (record name = initials of the S1 name, same house number, unique acronym S1, R's best S1, p in (0.1, 0.9] → 0.95; 2,515 flips) | – | = | _probe_ | LOCO true rate 0.93–1.00 in every p bin > 0.1; expected +0.0001 |
| **exp31** | exp29gf + **prior-shift correction for test-only legal-form siblings** (US: same name, S1 without legal form, record with one, other house number → demote if p ≤ 0.98 near / 0.95 far) | – | 0.99035 (cost of the rule on the dense OOF, −0.00015) | _probe_ | test: 9,156 accepted US pairs demoted, ≈6.9K of them false by count matching; Monte Carlo LB effect **+0.00082** |
| exp30g | 4-CE GBDT stack with rival features (+ e5-base CE exp26) | – | 0.99054 (+0.00004 over exp29g) | – | e5-base CE OOF AUC 0.8969 / 0.8892; 4-CE logistic exp27c 0.99017. Noise-level gain; not used |
| exp01 | embedding blocker (GPU IVF) + 65-feature XGB + assign/threshold decoding | 0.98490 | – | **0.978934** | submitted |
| exp02 | + stage-2 stacking: stage-1 p competition + group-consistency features | 0.98685 | – | ≈ exp01 ("almost the same") | submitted: OOF gain did not transfer |
| exp03 | train and validate in a test-density world (S1 dropout 20%) + stage 2 | – | 0.98576 (exp01 models: 0.98318 @0.75) | – | not submitted: dense-world gain has the same form as exp02's |
| exp04 | French department→region canonicalization at test time + exp03 models (thr 0.7) | = exp03 | 0.98576 | **≈0.979** | same as exp01 (US/India gains cancelled by France) |
| exp04_fr90 | exp04 with a stricter threshold (0.9) for the unseen country (France) | = | = | **≈0.980** | **best**: a stricter unseen-country threshold gains ≈+0.001 (≈+0.007 on France) |
| exp05 | + token-alignment and house-number-difference features (dense world); France thr 0.9 | – | **0.98764** | **0.982907** | **best**; LB gain (+0.003) > dense gain (+0.0019), since it also helped France |
| exp05_fr95 / exp05_fr80 | exp05 re-decoded with France thr 0.95 / 0.8 | – | = | not submitted | LOCO curve flat between 0.9 and 0.95, expected ±0.0005; not worth a submission |
| exp06 (full) | + multilingual cross-encoder re-scoring the uncertain band, CE used for all countries | – | 0.98954 | not submitted | CE does not transfer across countries (LOCO −0.0009), so it is risky for France |
| **exp06_exp05** | CE stacker for seen countries (US/India); France keeps exp05 p (thr 0.9) | – | 0.98954 (US/India) | held | expected ≈0.9845: not enough to justify a submission (user: only submit near 0.99); kept in `runs/` |
| exp07 | prep_v3 normalization (mined gaps) + noise-word and abbreviation features; full from-scratch deterministic run (`work_v3`) | – | 0.98787 (stage 2) | – | +0.0002 in-country only; base for exp08/09 |
| exp08 | exp06-style cross-encoder on exp07 (v3 world), seen countries only | – | 0.98959 | – | CE +0.0017 over exp07 |
| exp09 | cluster-consensus features (does a record's differing word or number recur among other candidates?) | – | 0.98808 (stage 2) | – | logloss −7% (stage 1), F0.5 +0.0002 in-country; LOCO avg +0.0002 (no transfer gain) |
| exp10 | exp09 stage 2 + exp08 CE stacker (seen countries); France = exp09 p @0.9 | – | **0.98965** | **0.985008** | +0.0021 over exp05 (≈ the in-country gain × 0.85 + a little France) |
| exp10_diag | exp10 with every France row empty (US/India identical) | – | – | 0.849712 | **diagnostic**: France ≈0.96, US/India ≈0.9896 on test (see decomposition) |
| exp11a | exp10 + word-role rule for France, noise words fils/associés only (13,944 French rows gain a match) | – | = | 0.985847 | **+0.00084**: fils/associés are noise words |
| exp11b | exp10 + word-role rule for France, all rule words (fils, associés, groupe, développement, france; 25,185 rows) | – | = | **0.986449** | **best, +0.00144**; groupe/développement/france swaps are noise too (+0.0006 more) |
| exp12 / exp12r | role translation of French words into training vocabulary (fils/associés→services, groupe/développement/france→partners, family words→holdings), test side recomputed in work_v4; r = + exp11 rule on top | – | = | **not submitted** | +24.1K / −5.7K French accepts vs exp11b; merging distinct family words into one token made sister-company distractors look like duplicates, and ~4K+ were accepted |
| exp12s / exp12sr | exp12 with pure noise words only, each onto a distinct training noise word (fils→center, associés→service); dual-use handled by the exp11 rule (sr); family words untouched | – | = | not submitted | sr vs exp11b: +7,012 accepts (mostly "Et Fils"/"& Associés" appended at the same address), −2,341 mixed drops; expected ≈ +0.0001 |
| exp14 / r / rc / rcd | exp13 + exp08 CE (seen; dense **0.98975**) / + exp11 rule / + descriptor swaps / + dual-use words appended at the same address | – | 0.98975 | **rcd: 0.983376 (−0.0031)** | France 3.248 / 3.358 / 3.376 matches. rcd bundled 2 untested rules: +33,062 French matches (descriptor swaps + dual-use appends), consistent with them being mostly **sibling organisations** (≈−0.0044 if all wrong). Mistake: never bundle untested hypotheses into a submission |
| exp26 / exp30 | 4th CE (multilingual-e5-base) + 4-CE GBDT stack with rival features | – | fold-0 CE AUC 0.8969 (≈ bert 0.8991) | – | _running_; adopt only if dense > 0.99050 |
| exp29t80 | exp29gf with France threshold 0.8 | – | = | **0.987799** | −0.00009: French p in (0.8, 0.9] ≈65% true; 0.9 is France's optimum |
| **exp29gf** | exp28f + **rival features** in the GBDT stacker (score minus the best score among the record's other S1 candidates, for p and each CE) | **0.987891** | **0.99050** (+ t_empty) | **best LB** (09-27 12:50); in the zip | +0.00018 dense over exp28c (two seeds); France identical; kit tail byte-identical |
| **exp28f** | exp24 inputs with a **GBDT stacker** over [logit p, 3 CE scores, 12 structural features] (logistic stack as fallback) + rule + two-threshold | – | **0.99032** (+ t_empty) | superseded by exp29gf | +0.00017 dense over exp24c |
| exp29t85 / exp29t95 | exp29gf with the France threshold 0.85 / 0.95 (rule p_set 0.96 for 0.95); US/India identical to exp29gf | – | = | _optional probes_ | France 3.278 / 3.213 matches per S1 (exp29gf 3.254). LOCO optimum between 0.85 (two-stage) and 0.93–0.95 (single-stage); expected ±0.0001–0.0002. (Replace the exp24 versions, deleted.) |
| **exp24f** | exp20f + **third CE (bert-base-uncased)** in the stack | – | **0.99015** (+ t_empty) | _new safe candidate_ | BERT OOF AUC 0.8991 / 0.8914; stacker weight ≈0.32 (like e5 v2); +0.00017 dense; expected ≈0.9875–0.9877 |
| **exp20f** | exp15 + CE v1 (filled) + **CE v2** stacked + two-threshold decode (seen, t_empty 0.6) + exp11 rule (France) | – | **0.98998** (+ t_empty) | _safe candidate_ | all parts validated (labels or LB); expected ≈0.9873–0.9876 |
| exp17cr | exp15cr with CE coverage filled for exp15's band (254K test / 88K train pairs scored by the saved exp08 fold models) | – | 0.98985 | _to submit (#1 on 09-27)_ | test: US 3,333 / India 683 rows change; expected ≈+0.0002 over exp15cr |
| exp19 | exp17cr + **France alignment (H5)**: accept French no-tie pairs of types ≥96% true in-country (brand / acronym at the same address, identical name at a far number or near + same legal form, address-less unique name) | – | = | **withdrawn** | 19,615 French flips. The same alignment applied out-of-country with labels (LOCO) **hurts** by −0.0007 to −0.0042 |
| exp16r | exp15cr + self-trained stage 2 for France (pseudo-labels incl. rule flips), unseen thr 0.98 | – | = | not submitted | France −14.8K / +3.7K matches, mixed audit (drops usually-true patterns) |
| exp15cr | exp14r recipe on a compact candidate set: ANN → stage-1 filter (p1 > 0.01 seen / 0.003 unseen) → stage 2 retrained on it | – | 0.98982 | _to submit_ | **3.95 candidates per S1 (was 18.56)**, dense F0.5 +0.00007 |
| **exp14r** | exp13 role-feature model + exp08 CE (seen countries) + exp11 rule; LB-validated parts only | – | 0.98975 | **0.986936 (best)** | +0.00049 over exp11b: the role model helps France too (my estimate was 0.9867, range 0.9860–0.9873) |
| exp11c | exp11b + French **descriptor** swaps at the same house number (club/école/amicale/comité/amis/sportive/parents/centre/…; 25,816 flips in 9.5% of French S1) | – | = | _to submit (09-26 last slot)_ | France empty share 0.0625 → 0.0571 (US/India 0.0576–0.0581, train singleton rate 0.0558) |
| exp06_ood | as above, but France uses the CE with a transfer-calibrated weight (OOD stacker, CE coef 0.22) | – | = | not submitted | differs from exp06_exp05 in only 9,272 French pairs (0.5% of S1): LB effect ≈±0.0002, not informative enough |

## Key findings so far (read these first)
0. **LB decomposition:** dense-world validation puts US/India at ≈0.985, so LB 0.980 implies **France ≈0.95**, even with the 0.9 threshold. France (15% of test) is the main gap to the top.
1. **Each S2/S3 record belongs to at most one S1.** Competition features (the record's rank and margin across S1s) and assign decoding are the strongest signals.
2. **The test world is denser than train.** Test has 5.73–5.82 S2/S3 records per S1 vs 4.68 in train; the matches per S1 are about the same, so ~40% of test records are distractors vs 26% in train. Expected-F estimated from model probabilities drops on test for every country (US −0.003, India −0.0025, France −0.011). This is the likely reason normal-world OOF over-estimates LB (0.9849 → 0.9789) and why exp02's +0.002 did not transfer.
3. **France (15% of test, never seen in training) is the hardest part.** Each record has 11 competing S1s (vs ~5), 22.6% of best pairs have a margin below 0.2 (vs 8–15%), and address similarity is lower. French data has families of near-duplicate entities ("Lions Compagnie SA" vs "Lions Compagnie SCI" on the same street with a different number).
4. About 1.6% of GT pairs never reach the matcher. They are mostly empty-address records with generic names ("Family Clinic"), which are unresolvable by design.

---

## exp00 — Preprocessing (`prep_v1.py` → `data/cache/prep_v1/`, ~5 min)
- **Normalization:** anyascii (diacritics, scripts), lowercase, punctuation removal, `&`→and.
- **Names:**
  - junk removal (`(ID: 123)`, `#123`, `| www.x.com`);
  - alias split (`X f/k/a|aka|dba|formerly|t/a Y` → Y is the real name);
  - domain and hashtag detection; honorifics stripped;
  - legal suffixes canonicalized into `name_sfx`; `name_core` = suffix-free name with duplicate tokens removed.
- **Mined native-script maps** (train GT only): 551,230 of 551,240 native names align token-for-token with their S1 name. That gives 1,347 tokens (प्राइवेट→private, प्रा→pvt) and 16 state names (महाराष्ट्र→maharashtra).
- **Addresses:** abbreviations → short canonical forms (street→st, rue→r, saint→st, …); US/India states → codes; `null` and `N°` handled; numbers extracted with leading zeros stripped.

## exp01 — Blocking + matcher (`exp01_block.py`, `exp01_match.py`) · OOF 0.98490 · **LB 0.978934**
**Blocking.**
- Model: two-tower EmbeddingBag over hashed char-3grams and words (2^19 buckets, dim 128, IDF-weighted), trained contrastively on GT pairs (in-country batches, hard-negative sorted batches, singletons and distractors as negatives), 3 epochs. One model per S1 fold, so train candidates are out-of-fold; test uses the fold0 model.
- Search: GPU IVF in torch (k-means ~1000 vectors per cluster, nprobe 32). Channels: name@20, addr@20, joint@30, reverse R→S1@3.
- Pruning: keep a pair if it is in the S1's top-15 by joint cosine **or** the S1 is among the record's top-2.
- Size and recall: raw union 160M pairs (72.7 per S1, recall 0.9899) → pruned 38.5M (17.4 per S1, recall 0.9838). The perfect-matcher upper bound is F0.5 0.9952.

| Held-out recall by epoch | US joint@30 | US union | India joint@30 | India union |
|---|---|---|---|---|
| 1 | 0.9791 | 0.9893 | 0.9757 | 0.9874 |
| 3 | 0.9806 | 0.9895 | 0.9770 | 0.9872 |

| Pruning (India) | per S1 | recall |
|---|---|---|
| top-5, record's best S1 | 5.8 | 0.9755 |
| top-15, record's top-2 (**chosen**) | 17.6 | 0.9832 |
| all candidates | 74.3 | 0.9887 |

**Matcher.**
- 65 features: rapidfuzz name and address scores, house-number overlap, suffix agreement, record flags, name-frequency counts, embedding cosines and ranks, competition features.
- Model: XGBoost depth 8, eta 0.15, 30% of negatives, 2-fold cross-fit.
- Top features: r_margin2, r_rank_j, r_gap_j, b_nnum, num_tset.

**Decoding (OOF).**

| threshold | 0.5 | 0.7 | **0.75** | 0.8 | 0.9 |
|---|---|---|---|---|---|
| plain threshold | 0.98195 | 0.98441 | | | |
| assign + threshold | 0.98329 | 0.98480 | **0.98490** | 0.98487 | 0.98388 |

The expected-F0.5 prefix decoder scored 0.98449 and separate S2/S3 thresholds scored 0.98487; neither beat assign + threshold.

**Breakdown:** India 0.9855, US 0.9843, singletons 0.978, S1 with one true match 0.955.

**Errors:**
- 32K false positives; 72% are distractors that copy an S1's address or name with ±1 house number.
- 147K false negatives inside the candidates (many empty-address records) plus 124K missing from the candidates.

**Dead ends:**
- Sparse TF-IDF top-k (sparse_dot_topn): more than 7 min per 50K queries.
- Exact GPU brute force: 8e12 scores per channel, memory-bandwidth bound, hours.
- Full-data embedding model: slow (VRAM spill) and not needed.

**Pitfall:** predicting before the test features had finished left all US rows empty, and the validator still passes. Always check per-country coverage.

## exp02 — Stage-2 stacking (`exp02_stack.py`) · OOF 0.98685 · LB ≈ exp01
- **Hypothesis:** the true records of one entity resemble each other, so an empty-address record with the same name as a confident candidate should be accepted.
- **Change:** 18 features computed from stage-1 OOF p:
  - rank, gap, sum and count>0.5 within the S1;
  - rank, margin and second-best within the record;
  - similarity (name, address, numbers) to the top-3 other confident candidates ("anchors").
- **Result (OOF):** 0.98685 (+0.0020); logloss 0.011 → 0.008. Singletons 0.978 → 0.988, S1 with one true match 0.955 → 0.946. Best threshold 0.7.
- **Verdict:** the leaderboard did not move. The gain lives in the normal-world validation and does not survive the denser test world. This motivated exp03.

## exp03 — Dense-world training/validation (`exp03_dense.py`, `scripts/run_exp03.sh`) · dense OOF 0.98576
- **Hypothesis:** the model and threshold are tuned in a world with 26% distractors, but test has ~40%. Removing 20% of train S1 entities turns their records into distractors (41% share), which reproduces test density with the most realistic kind of distractor.
- **Change:** candidates from the cached `cand_v1` restricted to kept S1s; ranks, pruning, competition features and S1 name counts recomputed in that world (`feat_v3/k80s0`); stage 1 and stage 2 retrained with cross-fitting; evaluated on kept S1s only.
- **Results (dense world, kept S1 = 1.77M):**

| model | best thr | F0.5 | F0.5 @ submitted thr |
|---|---|---|---|
| exp01 models (normal-world trained) | 0.8 | 0.98339 | 0.98318 @0.75 |
| stage 1 dense-trained | 0.75 | 0.98417 | |
| stage 2 dense-trained | 0.7 | **0.98576** | |

- Density explains only about −0.0017 of the OOF→LB gap (0.9849 → 0.9832). Dense training adds +0.0008 at stage 1; stage 2 adds +0.0016, the same form of gain that did not transfer in exp02.
- **Verdict:** a useful, more honest validation world, but it does not explain the LB. That led to the LOCO analysis below.

## LOCO analysis (`tools/loco.py`): the key finding
Train stage 1 on one country and score the other, in the dense world, with fixed 900 rounds and no target peeking:

| | in-country cross-fit | trained on the other country | best thr (unseen country) |
|---|---|---|---|
| India | 0.98473 | **0.95393** (US-trained) | ≥0.9 (edge of grid) |
| US | 0.98380 | **0.97066** (India-trained) | ≥0.9 |

- The matcher does **not transfer across countries**. On an unseen country it over-predicts (singletons score 0.88–0.92) and needs a much higher threshold.
- France is exactly this case. It is consistent with the LB decomposition: if US and India on test are ≈0.983, LB 0.979 implies France ≈0.955.
- This also explains why exp02/exp03 did not move the LB: they improve in-country fit, not transfer.
- **Robustness sweep** (`logs/loco_sweep1.log`; 60% of source-country S1 for training, a fixed 35% of target-country S1 for eval):

| config | US→India @0.7 | US→India best | India→US @0.7 | India→US best |
|---|---|---|---|---|
| base (all 65 features, depth 8) | 0.94091 | **0.95015**@0.95 | 0.95940 | **0.97110**@0.95 |
| monotone constraints | 0.91907 | 0.93829@0.98 | 0.95416 | 0.96676@0.95 |
| monotone, depth 6 | 0.92478 | 0.94053@0.97 | 0.95855 | 0.96704@0.93 |
| monotone, no count/length features | 0.92286 | 0.92692@0.9 | 0.94604 | 0.95099@0.9 |
| no count/length features | 0.94345 | 0.94926@0.93 | 0.95994 | 0.96422@0.9 |
| monotone, no count/length/cosine | 0.92030 | 0.92483@0.85 | 0.94403 | 0.94820@0.85 |

  Regularization does *not* help transfer; every restricted feature set is worse. The only consistent lesson is the threshold: an unseen country needs ≈0.9–0.95 instead of 0.7.
- Caveat: LOCO (one training country) is harsher than the France case (two training countries; France is Latin-script with US-like addresses), so the France optimum is likely between 0.7 and 0.95. It will be probed on the LB.
- Memory note: the first LOCO version held 33M×67 rows plus copies and thrashed (8.5 GB RSS, 0.3 GB free). It now keeps compact float32 arrays (~5 GB).

## exp04 — French region/department canonicalization (`exp04_frnorm.py`, `scripts/run_exp04.sh`)
- **Finding:** French S1 addresses end in the *region* (95%). S2/S3 end in the region (33%), the *department* (31%: Nord, Gironde, Loire-Atlantique, Pas-de-Calais), or nothing. US/India never had this conflict (states → codes).
- **Change:** map every French region and its departments to one region token (`frhdf`, `frnaq`, `frpdl`, …). This is generic normalization like the US state codes. Train contains none of these names (10 rows with "nord"), so only the test side is recomputed (`prep_v2/test → emb_v2 → cand_v2 → feat_v4`) and scored with the existing exp01/exp03 models.
- **Effect on France (each record's best S1):** median address cosine 0.82 → 0.904 (train 0.91); median address token-set 94 → 99 (train 100); competing S1s per record 11.1 → **5.7** (train ≈5.4); share with margin < 0.2: 22.6% → 20.2%.
- **Predictions:** France pairs +24K / −7K (≈3% of French pairs). Model-estimated France F0.5 rises 0.9797 → 0.9805 (exp01 models) and 0.9816 → 0.9823 (exp03 models). US/India are unchanged.
- Outputs: `runs/exp04/output` (exp03 models) and `runs/exp04_exp01/output` (exp01 models).
- The French threshold matters most: 2.9% of French predicted pairs have p in (0.7, 0.9], vs 1.3–1.5% for US/India. `exp04_fr90` (France thr 0.9; `tools/decode_variant.py`) removes ≈26K French pairs: 3.43 → 3.33 matches per S1 and 5.4% → 5.85% empty.

---

## Loss analysis after exp04 (`tools/loss_breakdown.py exp03 0.7 0.8 0`, dense world, F0.5 0.98576, loss 0.0142)
| error type | S1 affected | loss share |
|---|---|---|
| FN among candidates (matcher rejected a true pair) | 4.6% | 0.0053 (37%) |
| true matches missing from the candidates | 4.8% | 0.0049 (35%) |
| FP on non-singletons | 1.4% | 0.0032 (23%) |
| FP on singletons | 0.08% | 0.0008 (5%) |
- Blocking misses (95K pairs, 1.55%): 80% have an empty R address, and 62K of those have a name shared by several S1s. They are mostly unrecoverable, with at most ~30K potentially recoverable. **Not the lever.**
- Borderline pairs (p between 0.5 and 0.97) are dominated by **"family" distractors** in all countries: the same address or street with one name word swapped ("Recherche Club" vs "Recherche Pharmacie", "Thompson Academy" vs "Thompson Realty"), or a house number off by a small amount (1704 vs 1708). Whole-string fuzzy scores treat these like typos. This motivates exp05.

## exp05 — Token-alignment and house-number features (`exp05_tokfeat.py`, `scripts/run_exp05.sh`) · dense 0.98764
- **Hypothesis:** explicit features separating *a typo* from *a swapped word*, and *a digit deletion* from *an offset house number*, cut matcher FN/FP. Being language-agnostic, they should also help France.
- **Change:** 21 new features (`feat_v5`, 60 µs per pair, ~8 min per split on 8 processes).
  - Name and address tokens aligned both ways by Jaro-Winkler: weakest-token similarity, #unmatched tokens (<0.85), max IDF of an unmatched token, IDF-weighted coverage.
  - House numbers: first-number log |difference| and relative difference, prefix/suffix (digit-deletion) flag, digit edit distance, #S1 numbers missing in R.
  - Stages 1 and 2 as in exp03 (dense world); test uses exp04 French-normalized features; countries unseen in training get threshold `--unseen_thr`.
- **Results (dense world, kept S1):**

| model | best thr | F0.5 | India | US | singletons | 1 match |
|---|---|---|---|---|---|---|
| exp03 stage 1 | 0.75 | 0.98417 | 0.98473 | 0.98380 | 0.97742 | 0.95308 |
| **exp05 stage 1** | 0.8 | **0.98662** | 0.98683 | 0.98648 | 0.98513 | 0.95962 |
| exp03 stage 2 | 0.7 | 0.98576 | 0.98620 | 0.98548 | 0.98582 | 0.94196 |
| **exp05 stage 2** | 0.7 | **0.98764** | 0.98786 | 0.98750 | 0.99049 | 0.95121 |

- Stage-1 logloss 0.0111 → 0.0089. New features in the top 20 by gain: hn_edit (#3), tk_b_nun, atk_b_nun, tk_b_idfun, hn_logdiff, hn_prefix.
- **LOCO** (`logs/loco_exp05.log`): US→India best 0.95015 → **0.96163**; India→US best 0.97110 → **0.97782**. The gain is larger out-of-country than in-country, so the features generalize. Unseen-country optimum is still thr ≈0.93–0.95, with a flat curve from 0.9 to 0.95.
- **Test (France thr 0.9):** France 3.15 matches per S1 (exp04_fr90: 3.33), 6.5% empty. US/India unchanged (3.38/3.36). The new model is itself much stricter on French word-swap pairs: even at thr 0.7, France gets 3.26 matches per S1.
- **Submissions:** `runs/exp05/output` (France 0.9, primary), `runs/exp05_fr95/output`, `runs/exp05_fr80/output` (probes).

## Analysis after exp05 (LB 0.982907)
- **Implied France score:** if US/India on test ≈0.986 (dense 0.9876 minus the usual small gap), LB 0.9829 implies France ≈0.966. France is 15% of the entities but ~30% of the LB loss.
- **Remaining dense loss 0.0124** (`tools/loss_breakdown.py exp05 0.7 0.8 0`): FN in candidates 0.0044, out of candidates 0.0049 (mostly unrecoverable), FP 0.0031.
- **Remaining US/India errors look close to irreducible.** Empty-address records with a near-identical name (p 0.3–0.6) are correctly uncertain: the same pattern is wrong about as often, e.g. "H 5 U Harbors" belongs to another same-named S1. Random aliases at addresses shared by two S1s are also ambiguous.
- **France has 2.5× more uncertain best-pairs than the US** (0.25 vs 0.10 per S1 with p between 0.3 and 0.9). Inspection shows three types: (1) empty address with an identical name, which is ambiguous; (2) a random alias at the exact address; (3) **family variants at the same address with one French word swapped** ("Solidair Sportive" vs "Solidair Loisirs" p 0.77; "Vins Primaire" vs "Vins Sportif"), while noise ("Elevate (France) Sportif SAS" vs "… SAS Cie") gets 0.99. Type 3 is *semantic*: generic word (noise) vs content word (a different business). Our features cannot know this for unseen French vocabulary. This motivates exp06.
- **Value of fixing the band:** exp05 p in (0.02, 0.98) covers 1.7% of train pairs (559K; 763K on test, 250K of them French). A perfect classifier on that band gives dense F0.5 **0.9947** vs 0.9876.

## exp06 — Multilingual cross-encoder on the uncertain band (`exp06_crossenc.py`, `scripts/run_exp06.sh`) · _running_
- **Model:** `intfloat/multilingual-e5-small` (MIT, 118M) as a pair classifier on lowercased raw "name ; address" text pairs, max 96 tokens (mean 45). The embedding matrix is **frozen**: 96M of the 118M parameters, 1.8× faster (728 pairs/s), 2.1 GB VRAM, and unseen French/Hindi tokens keep their pretrained meaning.
- **Data:** exp05 stage-2 p in (0.01, 0.99): 725K train pairs (45% positive, OOF), 976K test pairs (France 250K ≈ 1 per S1; US/India ≈ 0.5 per S1).
- **Protocol:** 2-fold S1 cross-fit, 2 epochs, AdamW 3e-5, linear warmup/decay. A logistic stacker on [logit(p2), CE logit, product] is cross-fitted and replaces p only inside the band. Transfer check: CE trained on the US band only, applied to the India band.
- **Results (dense world):**
  - The CE alone on the band has OOF AUC 0.904 / 0.900 (fold 0 / 1) vs stage-2 p 0.942 / 0.940. It is weaker alone but **complementary**: the stacker coefficients are [logit p2 0.83, CE 0.61, product 0.04].
  - **Stacked: 0.98954 vs exp05 0.98764 (+0.0019).** India 0.98786 → 0.99010, US 0.98750 → 0.98917, singletons 0.99049 → 0.99435, one-match S1 0.95121 → 0.96106. Best thr 0.7; plateau 0.65–0.8.
- **Transfer check (critical):**
  - A CE trained on the US band only has AUC 0.743 on the India band (0.90 in-country).
  - With the US-fitted stacker (CE weight 0.6), India drops **0.98786 → 0.98694 (−0.0009)**.
  - With a stacker fitted in the out-of-country regime (cross-fit on India), the CE weight is 0.22 and India reaches 0.98831 (+0.00045). So there is some signal, but only at a small weight.
  - Conclusion: the CE learns partly country-specific patterns, so do **not** trust it at full weight for France.
- **Test:** France changes 2.5–3× more than US/India under the full CE (0.088 vs 0.03 changed pairs per S1), which is why the full variant is not submitted.
- **Why we expect exp06_exp05 to improve the LB, and by how much** (France is identical to exp05, so only US/India change):
  - Dense OOF gain +0.00190, bootstrap 95% CI [+0.00184, +0.00195] over S1, so sampling noise is negligible.
  - The gain is broad, not a few lucky S1s: 1.93% of S1 change score; 1.43% improve vs 0.50% get worse (≈3:1).
  - On test, US/India predictions change at 0.032 pairs per S1 vs 0.020 in OOF (a denser, harder world): the same direction, larger volume.
  - The exp02-type failure (the gain not transferring because France got worse) is excluded because France is untouched.
  - Estimate: ΔLB ≈ 0.85 (US/India share of test S1) × ΔF(US/India) ≈ +0.0016, up to ≈+0.0025 if the higher test change rate keeps the 3:1 win ratio. **Point estimate ≈0.9845 (range ≈0.9840–0.9855).** Confidence that it improves on 0.982907: high but not certain. The main risk is a shift between the dense train world and test US/India.
- **Submissions:**
  - `runs/exp06_exp05/output`: CE for US/India, France = exp05 (**submit**).
  - `runs/exp06_ood/output`: France with the OOD-calibrated CE stacker (**#2, France probe**).
- Training cost: ~25 min per fold (2 epochs × 351K pairs at 728 pairs/s, frozen embeddings, 2.1 GB VRAM); scoring 3.3K pairs/s.

## Reproducibility (for the final code submission)
- `scripts/reproduce_best.sh predict`: regenerates the current best TSV **byte-identically** from the saved artifacts (verified: two regenerations give the same SHA-256; re-scoring the test band from the saved cross-encoders gives max |diff| 0.0 vs the saved scores).
- `scripts/reproduce_best.sh full`: the exact command chain from the raw TSVs, including the steps that were run by hand. GPU training (embedding towers, cross-encoder) is not bit-exact, so this path reproduces the result up to tiny GPU noise.
- `runs/MANIFEST_best.sha256`: checksums of every artifact the best submission depends on (blocker models, native-script maps, exp05 XGB models, exp06 cross-encoders and stacker, test predictions, output TSVs). Verify with `sha256sum -c runs/MANIFEST_best.sha256`. **These files are never deleted during cleanup.**
- `er_common.write_outputs` sorts pairs by (s1, r) and uses an order-stable group_by (since 09-26). Earlier files had the same pair sets but a nondeterministic ID order within lists.

## Determinism (needed for "from scratch → exact same predictions") · `tools/determinism_check.py`
- Each stochastic component was trained twice on a sample. By default only the **IVF k-means** differed (float `index_add_` atomics, max diff 6e-5). The embedding towers, XGBoost GPU hist and cross-encoder fine-tuning were already bit-identical.
- Fixes (09-26):
  - `er_common.set_determinism(0)` (seeds, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cudnn deterministic, `torch.use_deterministic_algorithms`) is called at the start of every entry point (exp01_block, exp03–exp06).
  - The cross-encoder loads with `attn_implementation="eager"` (memory-efficient attention backward has no deterministic kernel), from the pinned HF revision `614241f…` so a fresh download gets identical weights.
  - After the fixes all four components are bit-identical across reruns.
- **Consequence:** the current artifacts (exp01–exp06) were trained *before* these fixes, so a from-scratch rerun would reproduce them only up to tiny k-means/attention noise. **The final submission will therefore be produced by one clean from-scratch run** of the final pipeline (`scripts/reproduce_best.sh full` in a fresh `ER_WORK_DIR`, base model downloaded by the code). That exact code and output go into the final zip, so re-running the code reproduces the submitted TSV exactly (same hardware/software stack).

## Gap analysis before exp07 (target ≈0.99)
- **Where the gap to the top is:**
  - With exp06, US/India on the dense world are 0.9895, about where the top teams' *overall* LB is. Their ceiling (a perfect matcher on our candidates) is ≈0.995.
  - France (15% of test) is ≈0.966 by LB decomposition. A leaderboard at 0.989 is impossible with a France that weak, so **France is most of our gap** (≈0.003–0.004 LB). The rest is in-country.
- **Token-substitution mining** (`tools/subst_mining.py`; token differences in 400K pairs: confident French test pairs, and US/India GT pairs):

| gap | evidence (per 400K pairs) | where |
|---|---|---|
| alias phrases not split: "doing business as", "formerly known as", "also known as", "née" | ~10K / 7.5K / 7.6K train records; alias words end up inside `name_core` | all |
| generic words inserted or swapped in by the noise (center, services, service, partners, enterprises) vs *content* words that mean a different business | "center" extra in 12K US GT pairs; French "*→services" swaps | all |
| French `&` written as "et"; legal forms SCI/EI; abbreviations frs/fs/cb/svc; digit typos 5ARL/5tar/8lue/lnc | 1.1K / 5.4K / ~800 / ~400 | FR (and US/IN typos) |
| address designators (unit/apt/#/pmb/po box, door/h.no/plot/flat/shop, "N°"→no) | "unit" missing 43K (US); "no" extra 45K (FR), 27K (IN) | all |
| city-type suffixes (city, cdp, twp, county, town of, borough) | "city" extra 10K (US) | US |
| ordinals 7th↔seventh, 2nd↔2th | ~2K | US |
| French street types (cours→crs, quai→q, résidence, passage, lotissement) and bis/ter | ~3K | FR |
| India state codes TS↔TG, Kerala↔Keralam; renamed cities (Calcutta, Bombay, Bengaluru); Laxmi↔Lakshmi, Jai↔Jay | 2.3K / 1K / ~3K / ~1.5K | IN |

## Decision-rule check (`tools/decode_opt.py`, exp05 dense OOF)
- Per-S1 expected-F0.5-optimal subset selection (Monte Carlo over Bernoulli(p) labels, including "predict nothing") gives 0.98773 vs 0.98764 for the global threshold 0.7: **+0.0001, not a lever**. The assignment + global threshold rule is already near-optimal given the probabilities.

## Operational lessons
- **Never run two GPU jobs at once** on the 4 GB RTX 3050. Embedding training (≈3.9 GB) plus XGBoost GPU spills VRAM into shared memory, and both slow down 5–10×: blocker epoch 297 s → 755 s, and stage-1 XGB made no progress for 10 min. GPU phases are scheduled sequentially; CPU-only phases (features) can overlap with one GPU job.

- **Disk:** the drive filled up during v3 stage-2 features (0 bytes free). Freed ~13 GB: v3 `*.npz` hashed features (regenerable), the old world's *train-side* layers (feat_v3, feat_v5/train, feat_v5s2/train, cand_v1/train, exp05 s1_oof) and the isolated exp07 check. The exp06_exp05 predict path is intact (`sha256sum -c runs/MANIFEST_best.sha256`: 17/17 OK).

## exp07 — prep_v3 normalization + token features v2 (`prep_v3.py`, `exp07_tokfeat2.py`, `scripts/run_world_v3.sh`) · _running_
- **Hypothesis:** each gap above lowers similarity for every pair that contains it, blurring matches and non-matches, and French pairs are hit hardest. Fixing them at the source plus telling the model *what kind* of word is unmatched should lift both in-country accuracy and France.
- **Change:**
  - `prep_v3`: all the fixes in the table, applied identically to S1/S2/S3 and train/test. The French region canonicalization from exp04 is folded in.
  - `exp07` features: exp05's 21 token/number features recomputed with **abbreviation-aware** alignment, plus 12 **noise-word** features: max/min noise score and #noise-like / #content-like among unmatched name and address tokens.
  - Noise score of a word t: ns(t) = log((#R names with t + 1) / (#S1 names with t + 1)) − log(N_R/N_S1), computed per split from its own records (no labels, like IDF). French words get scores from the French test data.
- **Run:** everything from the raw TSVs in a fresh `work_v3/` (blocker retrained, candidates re-searched, dense world, stages 1/2), with `set_determinism` on. If it becomes the best, it *is* the reproducible final pipeline.
- **Isolated check of the feature layer** (old prep_v1 world, same candidates as exp05; `exp07_old`, CPU XGB eta 0.1):
  - Stage-1 logloss 0.00877 / 0.00831 vs exp05 0.00889 / 0.00860.
  - **Dense F0.5 0.98681 @0.8 vs exp05 stage 1 0.98662 (+0.0002)**; singletons 0.98513 → 0.98624.
  - New features rank high by gain (atk_b_ncont #8, tk_b_ncont, tk_b_nsmax/nsmin), but the in-country F0.5 gain is small. The in-country matcher is saturating (stage 1 ≈0.987).
  - The value for France (transfer) is to be measured with LOCO.
- **v3 world run:** the first attempt died with a RAM OOM in the train search while the CPU XGB check was also running (both ≈5–6 GB). It was resumed from the search (`scripts/run_world_v3_resume.sh`) with nothing else running. **Rule: one heavy job at a time.**
- **v3 blocking** (retrained towers on prep_v3 text): pair recall **0.98979 vs 0.98992** (v1); per channel name 0.680 / addr 0.901 / joint 0.963 / rev 0.929 (v1: 0.681 / 0.903 / 0.962 / 0.928). The learned char-n-gram blocker was already robust to these surface variants, so normalization matters for the matcher, not for recall.
- **exp06 loss breakdown, in-country after the CE** (dense, F0.5 0.98954, loss 0.01046): out-of-candidate FN 0.00453 + 0.00031 (**46%**), FN inside candidates 0.00365, FP non-singleton 0.00165, FP singleton 0.00032.
- **v3 stage 1** (prep_v3 + exp07 features, dense k80s0): logloss 0.00866 / 0.00794. **F0.5 0.98692 @0.8** (India 0.98701, US 0.98685, singletons 0.98643, one-match S1 0.96004).
  - Compare: exp07 features in the old world 0.98681; exp05 0.98662.
  - **Normalization adds +0.0001 in-country; the whole exp07 upgrade adds +0.0003.** In-country accuracy is saturated at the matcher level, so the remaining levers are the cross-encoder (+0.0019) and France.
  - Test noise table: "participations" (a French *family-variant* word, e.g. "X Participations") gets ns 8.0 because it appears mostly in distractor records. It could look like appended noise to the model, which is a France FP risk to check.
- **v3 stage 2:** F0.5 **0.98787 @0.75** (exp05 0.98764; +0.0002). India 0.98804, US 0.98776, singletons 0.99211, one-match S1 0.95084. Test: France 3.13 matches per S1 (thr 0.9), US/India 3.36–3.38.

- **French family-word check** (predicted French pairs, thr 0.9, where the R name has a family word the S1 name lacks; exp05 → v3):
  - participations 1 → 0; holding 2 → 2; développement 874 → 106; services 7697 → 7650.
  - **groupe 881 → 2139.** "Groupe" is ambiguous: sometimes noise (the French "Group"), sometimes a separate family business ("Financement Groupe" vs S1 "Financement Energie" at the same address). exp09's consensus features are designed to tell the two apart.

## Two-stage LOCO (`tools/loco2.py`, v3 world): does stage 2 hurt an unseen country?
- Hypothesis from LB history: stage 2 helped in-country but never moved the LB (exp02, exp04), so perhaps it hurts France.
- Test: the full production pipeline (stage-1 cross-fit, then stage 2 on OOF p1) trained on one country and applied to the other.

| out-of-country | stage 1 best | stage 2 best | stage 2 @0.9 | singletons (s2) | one-match S1 (s2) |
|---|---|---|---|---|---|
| US→India | 0.96063 @0.9 | **0.96197** @0.85 | 0.96136 | 0.923 | 0.871 |
| India→US | 0.97782 @0.9 | **0.97863** @0.85 | 0.97852 | 0.963 | 0.927 |

- **The hypothesis is rejected:** stage 2 slightly *helps* out-of-country, and France's setup (stage 2, thr 0.9) is near its out-of-country optimum (0.85–0.9).
- **The real out-of-country failure is false matches on singletons (0.92–0.96 vs 0.99 in-country) and one-match S1 (0.87–0.93 vs 0.95):** look-alike distractor businesses are accepted. This motivates exp09.

## exp09 — Cluster-consensus features (`exp09_consensus.py`) · _running_
- **Idea:** a distractor business ("Recherche Pharmacie" next to S1 "Recherche Club") has its *own* ~3–4 records, all carrying the differing word and the same other house number. Noise changes individual records independently. So: is a candidate's differing token **shared by other candidates of the same S1** (→ separate entity) or unique (→ noise)? Structural, not vocabulary-based, so it should transfer to France.
- **Features (9):**
  - extra-name-token support (all candidates / other source only / #tokens with support ≥2);
  - support of S1 tokens missing in R;
  - identical-core cluster size;
  - house-number support for R's number vs S1's number, and whether they are equal;
  - extra-address-token support.
- Smoke test (400K v3 pairs, 2.4 s): nx_sup_max **0.672 for non-matches vs 0.053 for matches**; ax_sup_max 2.51 vs 0.35; hn_r_eq_s1 0.13 vs 0.75.
- **Results (dense world):**
  - Stage-1 logloss 0.00810 / 0.00741 vs exp07 0.00866 / 0.00794 (**−7%**, the largest logloss drop since exp05). F0.5 **0.98714 @0.8** (exp07 0.98692).
  - Stage 2 0.98808 @0.75 (exp07 0.98787). New features in the top 20: hn_sup_r (#13), nx_n2 (#16).
  - As with every in-country change since exp05, better calibration barely moves F0.5: the remaining in-country errors are the genuinely ambiguous ones.
  - The real test is out-of-country:

| two-stage LOCO (best thr) | exp07 stage 1 | exp09 stage 1 | exp07 stage 2 | exp09 stage 2 |
|---|---|---|---|---|
| US→India | 0.96063 | 0.96139 | 0.96197 @0.85 | 0.96170 @0.8 |
| India→US | 0.97782 | 0.97867 | 0.97863 @0.85 | 0.97922 @0.85 |
| **average** | 0.96923 | 0.97003 | 0.97030 | **0.97046** |

  - Singletons improve out-of-country (stage 2 US→India 0.923 → 0.932), but one-match S1 do not.
  - **Conclusion:** better features no longer close the out-of-country gap (in-country ≈0.988 vs out-of-country ≈0.970). Every feature change now moves both by ≈0.0002, so the gap needs a *domain-adaptation* technique, not features. Next: covariate-shift importance weighting and per-country rank normalization (both use only unlabeled target features).
- **exp10 (exp09 + exp08 CE, seen countries):** stacker coefficients [0.81, 0.58, 0.04]. **Dense 0.98965** (India 0.99014, US 0.98932, singletons 0.9946, one-match S1 0.96136), the best in-country so far, but only +0.0001 over exp08 (0.98959). Test: France 3.125 matches per S1 (thr 0.9), US 3.415, India 3.372.

## Domain-adaptation checks for France (single-stage LOCO, v3 features + consensus; `logs/loco_iw.log`, `logs/loco_rank.log`)
Evaluated on 35% of the target country's S1, trained on 60% of the source country's S1, 900 rounds, no target peeking:

| config | US→India best | India→US best | average |
|---|---|---|---|
| base | 0.96298 @0.93 | 0.97852 @0.93 | 0.97075 |
| importance weights, clip 5 | **0.96899** @0.97 | 0.97731 @0.97 | **0.97315 (+0.0024)** |
| importance weights, clip 10 | 0.96807 @0.98 | 0.97682 @0.98 | 0.97245 |
| importance weights, clip 20 | 0.96877 @0.98 | 0.97640 @0.98 | 0.97259 |
| per-country rank normalization | 0.87235 | 0.87068 | **broken** |

- **Importance weighting** (a domain classifier on unlabeled features; weight = odds of looking like the target):
  - The classifier separates the countries almost perfectly, so most weights sit at the clip floor.
  - It still gives +0.006 on the harder direction and −0.0012 on the easier one: average +0.0024.
  - The best threshold moves to 0.97, because the weights change the effective class balance.
- **Rank normalization** destroys calibration (label prevalence differs between countries). Rejected.
- **Honest scale check:** France is 15% of test, so even +0.0024 out-of-country is ≈+0.0004 LB.

## Where we stand (09-26, 11:00) and the open question
- **In-country is saturated:** dense 0.98965 (exp10). Normalization, token/noise and consensus features each moved it by only +0.0001–0.0002; the cross-encoder gave +0.0017.
- **Out-of-country** (the France proxy) stays ≈0.018 below in-country, and features no longer close it.
- **Unknown that decides the next step:**
  - LB 0.9829 (exp05) vs dense 0.9876 leaves a 0.005 gap. It is either (a) France weak (≈0.966, my working assumption) or (b) test US/India harder than the dense world, with France fine. These need different fixes.
  - **Diagnostic:** submit exp10 and exp10 with France emptied. LB(exp10) − LB(diag) = 0.15 × (F_France − s_France), with s_France the French singleton share (≈0.06). This gives France's score within about ±0.005, and US/India's test score within ±0.001.
- **Expected LB of exp10:** US/India ≈ dense − 0.002 ≈ 0.9877 → ×0.85 ≈ 0.8395. France unknown: at ≈0.966 the total is ≈0.984–0.985; at ≈0.98 it is ≈0.987.

## exp11 — Word roles: why France loses matches (`exp11_rolerule.py`)
**Where France is uncertain.** About 10% of accepted French pairs are in the uncertain band (0.3–0.98), vs 3.5% for US/India. The matched-side name adds a word the base name lacks:
- groupe 6.5%, france 6.0%, associés 5.4%, fils 5.0%, cie 3.1%, développement 2.9% of uncertain French pairs;
- each below 0.1% of confident French pairs.

**The generator adds words in two roles, with a separate vocabulary per country** (training labels, assigned pairs whose names differ by one extra word):
- **Family / sister-company distractor:** the full base name plus the word, usually with another house number ("Falcon Corp" → "Falcon Group", 3179 → 3200).
  - US: holdings, group, north/east/west/south/central/downtown/uptown/metro.
  - India: enterprises, public, industries, exports, overseas, ventures, infratech, holdings, group, solutions.
  - True-match rate **0.2%**. Nothing dropped from the base name in 94–97% of cases.
- **Noise:** the word *replaces* a descriptor and the address stays ("Obrien Lion LLC" → "Obrien LLC Services").
  - US: center, services, service. India: the same + partners.
  - US "partners" is dual-use: 18% swaps (true), 82% appends (distractors).

**Per-word structure in test** (swap share / same house number / model's mean p on swap-with-same-number pairs):

| word | swap share | same number | mean p (swap + same number) |
|---|---|---|---|
| US center / services | 0.83 | 0.63 | **1.00** |
| US partners | 0.18 | 0.13 | 0.98 |
| US holdings / group | 0.01 | 0.01 | – |
| France services | 0.77 | 0.76 | 0.98 |
| **France fils** | 0.76 | 0.85 | **0.27** |
| **France associés** | 0.76 | 0.80 | **0.50** |
| **France groupe / développement / france** | 0.17–0.20 | 0.20 | **0.79 / 0.31 / 0.56** |
| France international / holding / participations / distribution | ~0.01 | 0.02 | – (family, correctly rejected) |

**Rule** (country-agnostic, no labels). An assigned pair is accepted when all of these hold:
- the names differ by one S1 word swapped for one R word;
- the first house number is equal;
- the swapped-in word has a noise role in its own split and country: swap share ≥ 0.1, R-over-representation > 0.1, ≥ 200 occurrences.

**Rule on training labels:**
- true-match rate **0.9990 (US, 171,759 pairs) and 0.9849 (India, 96,881)**;
- the in-country model already accepts 99% (US) / 92% (India) of these; its rejections are real exceptions (true-match rate 0.42–0.48).

**Rule on test:**
- US: 0.9% rejected;
- **France: 27,945 of 36,881 rule pairs (76%) are rejected**, and their p is spread over 0.01–0.9, i.e. word-driven, not confident.

**Applied to unseen countries only** (p := max(p, 0.95) when p > 0.01). US/India rows are byte-identical to exp10.

**Expected on the LB:**
- if the flipped pairs are true matches: about **+0.0008 (exp11a) / +0.0015 (exp11b)**;
- if they are distractors: about −0.002 / −0.004.

**Cross-country check (US→India, labels): not informative.** US and India share their noise words (center/services/partners), so the rule barely fires: 1,297 flips with a true-match rate of 0.41, which are genuine exceptions. It also shows the danger of low swap shares: India words with a swap share of 0.10–0.13 (agencies, motors, steel) are sister-company words (true-match rate 0). France is the unique case of an unfamiliar noise vocabulary.

**LB results (09-26):**

| submission | LB | vs exp10 |
|---|---|---|
| exp10_diag (France emptied) | 0.849712 | – |
| exp11a (fils/associés) | 0.985847 | **+0.00084** |
| exp11b (all five words) | **0.986449** | **+0.00144** |

The rule works, and the dual-use words (groupe/développement/france) add another +0.0006.

**LB decomposition** (France weight 0.14975; true singleton share 0.0558 in both training countries, assumed the same for France):
- **US+India on test ≈ 0.9896**, exactly the dense validation (0.98965). The dense world is accurate, and **the whole remaining gap is France**.
- **France: exp10 ≈0.960 → exp11a ≈0.965 → exp11b ≈0.969**, still ≈0.02 below US/India.
- **Ceiling:** France at the US/India level gives LB ≈0.9896. 0.99 would need France ≈0.992, above in-country.
- Every +0.01 on France is +0.0015 on the LB.

## exp12 — Role translation of French words (`exp12_roletrans.py`, work_v4) · not submitted
- **Map (all roles):**
  - fils/associés → services;
  - groupe/développement/france → partners;
  - international/holding/participations/distribution/enterprises/trading → holdings;
  - "5as" (the SAS typo, appended at the *same* address, hn_eq 0.85) excluded by a house-number guard.
- **Recompute:** test side only, with unchanged exp09 models. Blocking volume is unchanged (France 4.86M vs 4.87M pruned candidates).
- **Result vs exp11b:** France 3.300 matches per S1 (exp11b 3.229); +24,109 / −5,731 accepted pairs.
- **Audit of the new accepts:**
  - 7,162 are appended word + different house number, the sister-company signature ("Ets Rural SARL, 135" → "Ets Rural Participations S.A.R.L., 136").
  - About 4,000 carry pure family words.
- **Cause:** mapping several distinct family words onto ONE token ("holdings") turns a sister-company family ("X Participations", "X International", "X Holding") into identical records. That distorts the consensus, frequency and noise statistics.
- **Fix → exp12s:** translate only pure noise words, each onto a distinct training noise word. Leave family words (already rejected) and dual-use words (exp11 rule) alone.

## Remaining France gaps: structural pattern comparison (after exp11b)
**Method:** for country-agnostic patterns (name difference × house-number relation), compare the dense-world training true-match rate with the French acceptance rate. Caveat: the test contains extra distractor types, so a gap is only a lead.

**1. Identical name, different house number (31K French pairs; train true 0.962, France acc 0.305). No action.**
- Split by legal form and number distance, US training shows a **sister-entity distractor**: same name, *neighbouring* number (≤12), different legal form. True-match rate 0.127 (legal form missing on one side: 0.603).
- US test has 15–25× more of them per S1 than the dense world, and the model rejects them (mean p 0.11) while US still scores ≈0.9896.
- France behaves like the US (mean p 0.14), so this is probably correct.
- India's training labels say these are mostly true (0.90); its model accepts ~60%.

**2. One-word swap with a frequent French *descriptor* word, same number (≈17K pairs; mean p 0.24–0.62). Open, LB probe candidate.**
- Words: club, école, amicale, comité, amis, sportive, centre, parents, société, fêtes, collège, primaire.
- In US/India, same-number swaps are 99% true (280K US), but they are typos spread over rare tokens. No frequent descriptor↔descriptor swaps exist there, so the labels cannot decide.
- Arguments for true: every swap type in US/India is true, even at a different number (0.952), and France's confirmed noise words use the same swap mechanism.
- Argument against: these words are *not* over-represented on the R side (nsc −0.2, unlike noise words), which is also consistent with distractors built as "base + other descriptor".
- **Break-even true rate ≈0.67.** If true: ≈+0.001–0.0013 LB; if distractors: ≈−0.002.
- **Structural argument for the same-number subset:** descriptor swaps sit at the same house number 40% of the time (noise words ~80%, sister-company words ~2%). That makes the whole population a ≈50/50 mixture, but the same-number subset ≈97% noise.
- **Probe = exp11c.** After the flips, France's empty share (0.0571) matches US/India, a consistency signal.

## Where the in-country loss is (09-26 evening; dense validation, exp09 stage 2 = 0.98808)
**Oracle decomposition** (each fix alone):

| fix | F0.5 | gain |
|---|---|---|
| add true pairs missing from candidates | 0.99294 | **+0.0049** |
| accept true pairs the matcher rejected | 0.99289 | +0.0048 |
| remove false positives | 0.99050 | +0.0024 |

**Candidate recall is 98.4%.** 95,803 true pairs (1.57%) never reach the matcher:
- **35% were retrieved but pruned** (embedding rank 15–80).
- **65% were never retrieved**, and 86.5% of those have an **empty address** on the S2/S3 side.
- **Recall by address:** records with no address 71.7% (4.4% of true pairs); records with an address 99.67%.

**Most of it is unrecoverable:** an address-less record whose name is shared by more than 20 kept S1s cannot be placed (41.8K of the misses). When such records *are* retrieved, the matcher accepts:

| S1s sharing the name | acceptance |
|---|---|
| 1 (unique) | 94.7% |
| 2 | 25% |
| 3–5 | 11% |
| more | ≈0 |

- **Recoverable:** ≈13.5K unique-name misses, plus part of the pruned ones. Worth about **+0.0006–0.001**.
- **No leakage:** IDs and file order are random (corr 0.0001), and name / address / country are the only columns.

**Implication for 0.99:** our US/India on test ≈0.9896. A team at 0.99 with France ≈0.985–0.99 has US/India ≈0.990–0.991, so their main edge is **France (≈0.985–0.99 vs our ≈0.969)**: cross-country generalization, not in-country accuracy.

## exp13 — Word-role features in the model (`exp13_rolefeat.py`, feat_v13)
**Idea:** give the model the word-role information itself, so it learns "noise-role word swapped at the same address → match" from US/India and applies it to any vocabulary.

**Role table** per split and country, model-free:
- built from near-duplicate candidate pairs (names differ by one extra R token plus at most one missing S1 token) where the S1 is the R's top embedding candidate;
- per word: swap share, same-house-number share, log rate;
- the same statistics for a word as the missing S1 token;
- document frequency in S1 names and R over-representation.

**Pair features:** 13, min/max over the extra and the missing tokens.

**The tables recover the roles without labels:**

| role | example words | swap share |
|---|---|---|
| noise | center, services, fils, associés | 0.71–0.84 |
| family | holdings, enterprises, participations | 0.00–0.05 |
| dual-use | partners, groupe | 0.16–0.25 |
| French descriptors | club, comité | 0.98–0.99 (same number 35–40%) |

**Test:** cross-country benchmark, same samples as the baseline (0.96298 / 0.97852), then the in-country stages (`scripts/run_exp13.sh`).

## Deep-dive 09-26 night: why we are not at ~0.99 (all measured with labels unless stated)
**LB decomposition** (exp10_diag):
- US+India on test ≈ **0.9896–0.990**, the same as dense validation;
- France ≈ **0.966–0.969** (exp11b; about +0.003 after exp14r).
- The assumption behind it (France singleton share like US/India) is supported: 4.8% of French S1 have no plausible candidate (best p < 0.05) vs 5.1% US / 5.3% India.
- Top-3 was ≈0.989 on 09-26. With US/India ≈0.9896, reaching it needs France ≈0.986.

**In-country error anatomy** (exp13 OOF, base 0.98842, oracle fix per category):

| error | pairs | F gain if fixed |
|---|---|---|
| true match not in candidates | 95,803 | +0.0049 |
| — of which address-less R with a shared S1 name | 62,807 | +0.0032 |
| true match in candidates but rejected | 74,693 | +0.0043 |
| — of which address-less R with a shared S1 name | 44,241 | +0.0023 |
| FP: R is a pure distractor | 6,839 | +0.0011 |
| FP: R's S1 was dropped (orphan) | 7,382 | +0.0010 |
| FP: R belongs to another S1 | 3,634 | +0.0005 |

**Address-less R with a name shared by ≥2 S1s = 0.0055, half of all loss. Irreducible.**
- Records per S1 per source vary widely (1–4 per source, no quota to exploit).
- IDs and file order are random.
- Test count: 265,506 address-less R (2.7%). 81,873 are ambiguous (France 17,287, India 37,518, US 27,068); exp11b accepts only 11,592 of them.
- **Dropping the accepted ambiguous ones hurts:** precision is 0.855 (2 same-name S1) / 0.797 (3–5) / 0.734 (6+), vs a removal break-even of ≈0.72. F changes −0.00021 / −0.00004 / 0.
- Unique-name address-less: precision 0.966; dropping them costs −0.0061.

**Levers checked (no hyperparameter is worth more than +0.0002):**

| lever | result |
|---|---|
| candidate pool 15 → 30 per S1 | oracle +0.00017 (pruned true pairs sit at embedding rank 30–80) |
| all non-ambiguous missed candidates (pruned + never retrieved) | oracle +0.0017 |
| two-threshold decode (S1 with no accepted match: t_empty; others: t_main), in-country | best t_main 0.75 / t_empty 0.6 = 0.98856 (**+0.00014**) |
| same decode, cross-country | a lower t_empty *hurts*: lone candidates of empty S1 are mostly distractors out-of-country |
| France threshold | 0.9 is at the cross-country optimum (0.90–0.93) |
| ensemble exp09 + exp13 stage-2 OOF (weight 0.3/0.7) | 0.98857 (**+0.00015**) |
| reverse descriptor rule (reject accepted French descriptor swaps) | the model already rejects them (< 1K accepted) |

**The cross-encoder is anti-correlated on France.** On LB-established French truths (CE logits on exp08 test_ce):
- known-true fils/associés swaps: CE mean **−3.9** (4% > 0.5);
- known-false descriptor swaps: CE mean −0.77 (37% > 0.5).

It memorised the US/India noise vocabulary. It stays off for France.

**New constraint (organisers' update, 09-26 night):** `candidate_pairs.tsv` counts in the final ranking, and **a smaller candidate set per S1 ranks higher**. Ours: **18.56 per S1** (median 17, p90 24, max 434; 32.2M pairs) for ~3.4 true matches per S1. So bigger candidate pools are out, and compaction becomes a goal.

## exp15 — Compact candidate set (`exp15_compact.py`)
**Goal:** the organisers' update ranks a smaller `candidate_pairs.tsv` higher, without losing F0.5.

**Dense validation** (exp13 OOF; final F0.5 with stage-2 p restricted to the kept pairs):

| candidate generation | per S1 | recall | F0.5 |
|---|---|---|---|
| current: ANN top-15 + R-rank ≤ 2 | 19.02 | 0.9843 | 0.98842 |
| ANN only, S1 top-3 / 5 / 8 / 10 + R top-1 | 6.00 / 6.79 / 9.18 / 11.03 | 0.9756 / 0.9766 / 0.9786 / 0.9797 | 0.98784 / 0.98797 / 0.98810 / 0.98816 |
| **ANN + stage-1 filter p1 > 0.001 / 0.005 / 0.01 / 0.03** | **3.88 / 3.71 / 3.65 / 3.57** | 0.9843 / 0.9842 / 0.9842 / 0.9838 | **0.98842 (all)** |
| stage-1 top-k per S1 (+ each R's best S1), k = 3–6 | 6.0–7.2 | 0.981–0.984 | 0.98842 |

**Test** (exp14r accepted pairs that each filter would drop):

| filter | US | India | France |
|---|---|---|---|
| p1 > 0.01 | 4.02 per S1, drops 0 | 3.75, drops 10 | 4.10, drops 9 |
| p1 > 0.003 | 4.19, drops 0 | 3.92, drops 0 | 4.39, drops 0 |

**Chosen:** p1 > 0.01 for training countries and p1 > 0.003 for unseen countries (stage 1 is less calibrated out of country). That gives **≈3.95 per S1 vs 18.56 (−79%)**.

**Pipeline, kept honest:** `candidate_pairs.tsv` = exactly the pairs fed to the matcher.
1. ANN blocking (exp01_block, ≈19 per S1).
2. Learned pair filter (exp13 stage-1 GBDT, cross-fitted OOF on train).
3. Candidate set.
4. Stage 2 **retrained** on the filtered set, with its competition and consistency features computed only over kept pairs.
5. exp08 CE stack (seen countries), then the exp11 rule (unseen countries).

**Results:**
- **Kept:** train (dense world) 6.44M of 33.6M pairs (2.92 per S1, was 15.21); **test 6.84M of 32.2M (3.95 per S1, was 18.56)**. 62,686 test S1 (3.6%) have no candidate left; the validator accepts empty candidate rows.
- **Stage 2 on the filtered set:** logloss 0.0338 / 0.0324 (on harder negatives, not comparable); **F0.5 0.98858 @0.75** (exp13 on all candidates: 0.98842). Removing hopeless pairs sharpens the competition features.
- **+ CE (exp15c):** **0.98982 @0.7** (exp14: 0.98975).
- **exp15cr (+ exp11 rule):** France 3.254 matches per S1 (exp14r 3.248), empty 0.0612; US / India 3.394 / 3.376. It differs from exp14r in US 7,350 / India 4,407 / France 6,120 rows.
- **Candidate file:** 3.95 per S1, median 4, p90 6, **max 63 (was 434)**; 6.84M pairs (−79%).
- **Expected LB:** ≈ exp14r (0.9869–0.9872). It becomes the base for the final package, because candidate size now counts in the ranking.

## Two-threshold decode on the CE-stacked exp15c (seen countries)
Grid on `runs/exp15c/oof_combined.parquet` (now saved by `exp10_combine.py eval`):

| | F0.5 |
|---|---|
| single threshold 0.7 | 0.98982 |
| **t_main 0.75, t_empty 0.6** | **0.98990 (+0.00008)** |

Validated but tiny: fold it into the final candidate rather than spend a submission on it.

## exp16 — Self-trained stage 2 for unseen countries (`exp16_selftrain.py`) · not recommended
**Threshold dependence** (LOCO single stage, base → st):

| threshold | US → India | India → US |
|---|---|---|
| 0.9 | 0.96283 → 0.96164 (−0.0012) | 0.97803 → 0.97885 (+0.0008) |
| 0.98 (both directions peak) | 0.96305 → **0.96503** | 0.97659 → **0.98091** |

The self-trained model is over-confident, so the unseen threshold is **fixed a priori to 0.98** (not tuned on France). The rule's p_set is raised to 0.99 to stay above it.

**Pseudo-labels on France** (exp15cr):
- positives: assigned and p > 0.98, **plus the exp11 rule flips** (LB-validated, so the model can learn the dual-use words from French data);
- negatives: p < 0.02, sampled 0.3.

**Model:** stage 2 trained on all labeled train pairs (filtered set) + French pseudo-labels, 150 rounds (≈ exp15's best iteration of 131). It re-scores every French candidate; US/India stay as exp15cr. The exp11 rule is then re-applied (exp16r).

**Results:**
- **Training data:** labeled pairs = one S1 fold (3.07M; the full 6.1M × 142 features runs out of RAM in XGBoost); pseudo-labels 818K positive (16.5K rule flips) + 40K negative.
- **Effect on France:** the uncertain share drops from ≈10% to **6.6%**. At 0.98, France has **3.211** matches per S1 (exp15cr 3.254), empty share 0.0629 (0.0612). The rule now flips only 593 pairs.
- **Audit vs exp15cr (France):** −14,841 / +3,728 accepted pairs.
  - The drops include patterns that are usually true in training: x1m2 same number 5,263 (true 0.971), x0m0 missing number 4,249 (0.917), **x0m0 same number 719 (0.999)**, and only a few descriptor swaps (club/école).
  - The adds are mostly dual-use words *appended* at the same address (groupe/développement/france, ≈3K), the pattern suspected false in exp14rcd.
- **Verdict:** at the a-priori 0.98 threshold the self-trained model is too conservative on France, and the LOCO gain (+0.0017) does not clearly transfer. **Not a candidate.** At most a labelled probe with an uncertain sign.

## 09-27 00:45 — Diagnosis for the ≥0.9895 goal (no new submissions yet)
**Target arithmetic.**
- Our US/India on test ≈0.990. That can't be far below any team, because half the in-country loss is irreducible.
- A top team at ≈0.989 therefore has France ≈**0.982**; ours is ≈0.970. **France is the game.**
- Reaching 0.9895 needs France ≈0.986 (+0.016); reaching 0.99 needs France ≈0.989 plus in-country gains.

**Hypotheses tested today (all without submissions):**

| hypothesis | result | verdict |
|---|---|---|
| France misses a *volume* of matches (blocking) | R coverage 100% in all countries. With-address R acceptance: France 59.3% vs US 59.0% / India 58.1%. Address-less R: France 42.2% vs 53–58% (40% of French address-less R have ambiguous names, vs 25–33%) | rejected: France makes *swaps* (FN and FP), not a volume loss |
| French names collide more | S1 names shared by ≥2: France 50%, US 40%, India 53% | no |
| **French addresses are shared by different businesses** | S1 exact address shared by ≥2: **France 15.2%**, US 4.0%, India 4.9% (small street pool) | true, but **not the driver**: uncertain pairs per S1 are 0.48 (shared address) vs 0.43 (unique) |
| exact name + exact address is mishandled in France | 232K pairs, 99.94% accepted; the 134 exceptions are legal-form siblings sharing an address ("Lille Amis EURL" / "Lille Amis SARL") | handled |

**France is uncertain everywhere:** uncertain pairs (0.1 < p < 0.97) per S1 are France 0.43, US 0.25, India 0.11.

**Rejected French band (0.3, 0.9]: 46,669 pairs.** Top patterns (training truth in brackets):

| pattern | pairs | note |
|---|---|---|
| one-word swap, same number (0.995) | 9,925 | includes the descriptor siblings |
| swap + missing word, same number (0.973) | 8,785 | mostly **brand names / acronyms at the exact same address**: "Quonex", "Noviriza", "AF" = Association des Français, "PC" = Pornic Compagnie. A true noise type in training, 6.8K pairs |
| identical name, address-less R | 7.2K | |
| identical name, same number but **different street** | 3.2K | same-named business elsewhere in town; plausibly distractors |

Each pattern alone is worth ≈+0.0002 LB, far too small for the goal.

**Systematic cause (SHAP, stage-1 exp13, one-word swaps at the same number, R's top embedding candidate):**

| country | mean p | mean margin (log-odds) |
|---|---|---|
| US | 0.912 | 7.72 |
| India | 0.747 | 4.36 |
| **France** | **0.629** | **2.58** |

Features that pull France down vs US:
- **r_margin2 −1.09** (the gap between the R's best and second-best S1 in embedding space);
- rl_e_hneq_min −0.52, tk_b_idfun −0.37, ncl_partial −0.27, cos_j −0.27.

Quantiles of `r_margin2` for each R's top candidate (10% / 25% / median / 75% / 90%):

| country | quantiles |
|---|---|
| US | 0.33 / 0.61 / 0.79 / 0.96 / 1.10 |
| India | 0.08 / 0.41 / 0.69 / 0.87 / 1.03 |
| **France** | **0.05 / 0.26 / 0.51 / 0.74 / 0.88** |

Raw cosines are similar across countries; only the *margin to the runner-up* shifts. France's small name and street pool packs businesses together in embedding space, so every French record looks "contested".

**Hypothesis H1:** the margin/gap shift is a nuisance, so mapping the target country's density features onto the source distribution (quantile mapping, no retraining, unlabeled features only) improves transfer.
- Test: LOCO with the same trained model, with and without the mapping (`tools/loco.py --configs qmap`; variants margin / cos / cos+idf; samples 40% / 20%).
- If it helps (India shows a milder version of the shift), apply to France.
- **Result: H1 rejected** (same trained model, target features mapped):

| mapping | US → India | India → US |
|---|---|---|
| none | 0.96424 @0.95 | 0.97840 @0.93 |
| margin only | 0.96470 @0.98 | 0.97645 |
| all embedding (cos) features | 0.95914 (singletons 0.928 → 0.868) | 0.97587 |
| cos + idf | 0.95602 | 0.97618 |

  Mapping makes the model over-confident on distractors: embedding closeness carries **real ambiguity signal**, not a nuisance shift. Do not normalize it for France.

**In-country errors** (exp15c OOF, R with an address):
- FN 14,511; **true pair lost to another S1 (wrong assignment) 22,671**; FP 6,262.
- They are diverse: native-script names, brand names with partial addresses, house-number typos, digit-for-letter typos. No single systematic hole.

**H3 (French descriptor swaps are records of a dropped co-located sibling, so the swapped-in descriptor recurs among the S1's other candidates): rejected.**
- 34,334 of 34,367 descriptor swaps have support 0. They are single records, not orphan clusters.

**H4 (field-level contrast between an R's best and runner-up S1 would fix wrong-S1 assignments): rejected.** Of the 22,671 in-country "true pair lost to another S1" cases:

| case | count |
|---|---|
| R address-less **and** both S1 have the same name | 18,257 |
| R has an address, but both S1 share its house number (brand name at a shared address) | 2,073 |
| R address-less, names differ | 1,144 |
| house number matches neither S1 | 939 |
| the true S1 matches the house number and the chosen one does not | **only 65** |

This is genuine ambiguity: no field signal is left.

**France's uncertainty is mostly not ambiguity.** Uncertain pairs (0.1 < p < 0.97) per S1, split by whether a tie exists among the R's candidates:

| | France | US | India |
|---|---|---|---|
| **no tie** | **0.345** (89.5K pairs, mean p 0.60) | 0.176 | 0.052 |
| name tie | 0.059 | 0.029 | 0.027 |
| address tie | 0.010 | 0.002 | 0.002 |

The no-tie French pool (96K pairs incl. address-less R), current acceptance @0.9:

| type | pairs | mean p | accepted | status |
|---|---|---|---|---|
| descriptor swap, same number | 21.1K | 0.45 | 6% | LB: false |
| noise/family word, same number | 20.2K | 0.84 | 85% | LB: noise true |
| **brand/acronym, same number** | **16.6K** | 0.76 | **44%** | unknown in France; 0.97 true in train |
| **identical name, different number** | **14.4K** | 0.39 | **9%** | unknown |
| identical name, address-less | 5.2K | 0.86 | 66% | likely true |
| identical name, same number, different street | 3.1K | 0.51 | 15% | likely distractor |

**Ceiling of per-type France fixes:** a wrongly rejected match costs ≈0.085 of one S1, and France is 15% of test. Flipping *all* unknown groups correctly is worth ≈+0.008 France = **+0.0012 LB**, so France alone cannot reach 0.9895. In-country gains are needed too.

**CE coverage gap (the in-country lever).** exp08 scored the band of the old exp07 model. Share of exp15's uncertain pairs that have a CE score:

| exp15 p band | train | **test (US/India)** |
|---|---|---|
| (0.2, 0.9) | 98.9% | **89.2%** |
| (0.05, 0.95) | 97.4% | **78.7%** |
| (0.01, 0.99) | 86.2% | **67.0%** |

About 250K uncertain test pairs get no CE score, so the test file gets less of the CE gain than validation says. **exp17** scores the missing pairs with the saved exp08 fold models (out-of-fold on train, fold-average on test), then refits the stacker.

**exp17 — CE coverage fill** (`exp17_cefill.py`, `scripts/run_exp17.sh`):
- **Scored:** the missing band pairs with the saved exp08 fold cross-encoders; train out-of-fold, test fold-averaged.
- **Coverage:** CE scores train 697K → 785K, **test 997K → 1,251K**.
- **Dense validation:** 0.98982 → 0.98985 (validation coverage was already 97%).
- **Test:** US 3,333 rows (+1,022 / −2,515 matches) and India 683 rows change. Expected ≈+0.0002 LB.

**exp19 — France alignment, H5** (`exp19_align.py`).
- **Hypothesis:** French no-tie pairs of structural types that are near-certain in-country are under-accepted, because the model's confidence does not transfer.
- **Excluded:** France-specific distractor types (descriptor swaps: LB false; legal-form siblings).
- **Types, with in-country truth on the dense OOF** (p > 0.05, no tie; the in-country model already accepts 96–98.6% of them):

| type | in-country true rate |
|---|---|
| brand / acronym at the same number and street | 0.987 (346K) |
| identical name, far number (legal form equal or missing) | 0.986 (499K) |
| identical name, near number, same legal form | 0.960 (50K) |
| identical name, address-less, unique name | 0.982 (105K) |

- **France (exp17cr):**

| type | pairs | mean p | accepted @0.9 |
|---|---|---|---|
| brand | 83,099 | 0.928 | 86.5% |
| identical name, far | 15,963 | 0.467 | 39.8% |
| identical name, near | 3,245 | 0.594 | 41.9% |
| identical name, address-less | 18,178 | 0.957 | 91.3% |

- **Flips** (p in (0.05, 0.9] → 0.95): **16,181** (brand 9,515 / far 3,755 / near 1,372 / address-less 1,539). France 3.254 → 3.316 matches per S1, empty 0.0612 → 0.0587.
- **Expected:** ≈+0.0007 LB if French truth ≈ in-country; break-even at a true rate of ≈0.72.
- **Extension** (same hypothesis; wider scan of no-tie types, in-country truth vs French acceptance):

| type | in-country true rate | France accepted |
|---|---|---|
| identical name, same number, different street (street noise) | **0.997** (196K) | 74.4% (8.9K) |
| typo swap (edit distance ≤ 2), same number | **0.995** (216K) | 93.3% (17.5K) |
| words only dropped, same address | 1.000 | 96% (fine) |
| other name change, same address | 0.996 | 63% (14.7K) |

  The first two are added (**id_street, typo_swap**). "Other name change" is left out: it mixes in the appended dual-use words, suspected false (exp14rcd).
- **Final exp19:** 19,615 flips (brand 9,515 / id_far 3,755 / id_near 1,372 / id_noaddr 1,539 / id_street 2,269 / typo_swap 1,165). France 3.329 matches per S1, empty 0.0580, **now the same as US/India**. Expected ≈+0.0009 LB if French truth ≈ in-country.

**exp18 / exp20 — cross-encoder v2 + stacking:**
- **CE v2:** same frozen-embedding e5-small recipe, trained on exp15's band (0.01, 0.99), 3 epochs. OOF AUC 0.8950 / 0.8862 vs stage-2 p 0.9388 / 0.9370 on this hard band; the same profile as v1.
- **Double stacker** (638K train pairs with both scores): coefficients [logit p 0.84, ce1 0.07–0.14, ce2 **0.42–0.50**]; v2 carries most of the CE signal on exp15's band.
- **Dense 0.98998** (exp17c single CE: 0.98985) = **+0.00013**, plus t_empty +0.00008 at decode.
- **Decision rule** (set in advance): a small gain means CE capacity is not the bottleneck, so a bigger CE (download) is not a priority.
- **exp20f** = exp20c + rule + two-threshold (+370 seen-country pairs); France 3.254 / 0.0612, India 3.377 / 0.0578, US 3.392 / 0.0574. **Safe candidate.**

**Importance weighting with the current feature layers (LOCO, 40% / 25% samples, clip 5): rejected.**
- US → India 0.96422 → 0.96331; India → US 0.97841 → 0.97762.
- It hurt in both directions; the word-role features already absorb what it added in the exp09 era.
- exp21_iw.py (the production version) is not used.
- **Every validated transfer technique has now been tried:** role features (adopted), self-training (mixed), reweighting, quantile mapping, type alignment (all negative). France is at the ceiling of this model family.

**Expected-F0.5 decoder** (`tools/expf_decode.py`; per S1, choose the top-k of its assigned pairs maximizing the plug-in expected F0.5, with an "accept nothing" option worth the singleton probability):
- best 0.99003 vs threshold 0.98998 (+0.00005), about the same as the two-threshold rule. **Decoding is at its optimum.**

**Errors vs the CE band (exp20c OOF):**
- FP 11,899, of which only 1,577 with stage-2 p > 0.99 (outside the band).
- FN 42,745, of which only 224 with p < 0.01.
- Widening the CE band is worth ≈+0.00004 at most.

**Checkpoint 03:20.** The pipeline is at its ceiling: expected LB ≈0.9875 (exp20f).
- Twelve ideas were tested and rejected with labels tonight: blocking volume, co-location, margin normalization, orphan clusters, field contrast, type alignment, reweighting, exact-name key, French blocking, ensembling, hyperparameters, expected-F decoding.
- France is 15% of the score: +0.001 LB needs France +0.0067, and French probabilities are calibrated.
- In-country: +0.001 LB needs ≈25% of the reducible errors fixed; half of all in-country loss is unresolvable.
- The only sizeable lever left is more language-model capacity in the cross-encoder. A larger multilingual model needs a download (user permission).

**exp23 / exp24 result:**
- BERT CE OOF AUC 0.8991 / 0.8914.
- Triple stacker coefficients: logit p ≈0.82, e5 v1 ≈0, **e5 v2 0.24–0.31, bert 0.32–0.33**.
- **Dense 0.99015** (exp20c 0.98998): architecture diversity helps.
- exp24f = exp24c + rule + two-threshold (+343 pairs), validator PASS. Kit: `USE_CE3 = True` if adopted.

**exp23 (progress notes):**
- fold 0 OOF AUC **0.8991** on exp15's band (e5-small v2 0.8950, v1 0.8973 on its own band).
- Slow: bs 32, ≈25 min per epoch per fold; scoring runs ≈60–80 min per fold at ≈7 GFLOP per pair.
- **exp24** (queued) stacks all three CEs.

**exp23 (design):** a third CE on the locally cached **bert-base-uncased** (Apache-2.0, 110M; English only, fine because CEs are used for US/India only), for architecture diversity in the stack.

**exp22 — Stage-2 XGBoost hyperparameters (filtered set; never tuned since exp01): not a lever.**
- depth 10 / eta 0.05: 0.98864 (best iterations 180–212, logloss 0.03392 / 0.03244).
- depth 6 / eta 0.05: 0.98856.
- vs depth 8 / eta 0.1: 0.98858.
- Stage 2 is saturated.

**French blocking quality: fine.** R with an exact (name_core, first house number) match to exactly one S1 of its country (almost surely true):

| country | pairs | retrieved | S1 is R's top-1 |
|---|---|---|---|
| France | 538K | **99.25%** | 98.71% |
| India | 1.29M | 96.87% | 98.42% |
| US | 1.09M | 99.84% | 99.82% |

The hashed n-gram embedding transfers to French text. France's loss is in the matcher's (calibrated) uncertainty on noisy variants, and inherent to the French data (co-location, generic names, sibling organisations).

**Exact-name key for address-less R (unique exact name_core match to one kept S1): rejected** (dense validation, exp20c):
- 113,697 such pairs, true rate 0.922. 99.5% are already candidates and 103,820 are already accepted.
- The rule would *add* 9,874 pairs (R not accepted anywhere) that are only **35.6% true** (F0.5 −0.00064). The 568 outside the candidate set are 0.35% true.
- The blocker does not miss unique-name matches; the earlier "unique-name misses" are R whose own name is misspelt.

**exp18 — cross-encoder v2 (design):** same frozen-embedding e5-small recipe, trained on exp15's band (638K train / 1.03M test pairs), 3 epochs. It will be stacked *with* exp17's CE (`exp10_combine --ce_run2`: a second stacker for pairs with both scores, and `--t_empty` for the two-threshold decode). The single-CE path of the patched combine reproduces 0.98985 exactly. _running_

**Remaining French uncertainty after exp19:**
- p in (0.05, 0.3] 46.1K; **(0.3, 0.9] 33.9K**; (0.9, 0.97] 54.8K (incl. rule and alignment flips at 0.95).
- The rejected band is mostly one-word swaps at the same number (12.3K, chiefly descriptor swaps, LB false), identical-name address-less R *with* name ties (5.8K, ambiguous), identical name at another number that is sibling-like (3.9K), and small types.
- **No further clean near-certain type is left to align.**

**Plan / tally (09-27 01:45):**

| candidate | contents | expected LB |
|---|---|---|
| exp17cr | compact + CE fill | ≈0.9871–0.9874 |
| exp19 | + France alignment (probe) | ≈+0.0009 |
| exp20 | + CE v2 stacked + two-threshold | ≈+0.0002–0.0003 in-country |
| **stack** | | **≈0.9884–0.9887** |

**H5 tested out-of-country with labels, before any submission (LOCO exp13 predictions, same six types, same flip rule): rejected.**

| | base (best threshold) | + alignment | at threshold 0.9 |
|---|---|---|---|
| US → India | 0.96543 | 0.96475 | 0.96470 → 0.96245 |
| India → US | 0.97938 | 0.97778 | 0.97938 → 0.97515 |

- Out-of-country, the *types* are still mostly true (0.81–0.995), but the flipped subset (the pairs the model rejects within a type) is true only 0.62 (US → India) / 0.46 (India → US).
- That rate **tracks the model's p**: flipped pairs with p 0.05–0.3 are 0.31 / 0.14 true, 0.3–0.6 are 0.51 / 0.32, 0.6–0.8 are 0.68 / 0.51, 0.8–0.95 are 0.85 / 0.76.
- **Lesson:** out-of-country probabilities are informative and roughly calibrated (slightly over-confident, hence the ≈0.9–0.95 threshold). Blanket type-level corrections hurt; only targeted fixes of a specifically identified failure (the unknown French noise words, exp11: LB +0.0014) help. **exp19 withdrawn.**

Ensemble check (0.7 × exp15 + 0.3 × exp09 on the filtered pairs):
- base 0.98858 → 0.98868; under the CE stack 0.98985 → **0.98990 (+0.00005)**.
- The CE already captures it, and it would mix exp09's weaker France predictions in. **Not adopted.**

Reaching 0.9895 needs ≈+0.001 more. If the leaderboard confirms the alignment is strongly positive, the riskier types become worth a probe.

**Package:**
- `submission_kit/final/` holds run_pipeline.py (full chain from the raw TSVs, with a switchable CE v2 / t_empty / alignment), README, requirements and the methodology doc.
- `make_submission.py --run final --out_dir <final output>` packs 17 source files.

**Cleanup:** removed superseded runs (exp11a/11c, exp12×4, exp14rc/rcd, exp16/16r, exp10_diag, exp08_exp05, loco_pred), output TSVs of exp10 / 11b / 14 / 15 / 15c, and feat_v7s2. 33 GB free.

## exp28 — GBDT stacker (`exp28_gbstack.py`)
**Dense OOF (band pairs with all 3 CE scores: 638K; 2-fold cross-fit):**

| stacker | F0.5 |
|---|---|
| logistic (exp24c) | 0.99015 |
| GBDT on [logit p, 3 CE] | 0.99024 |
| **GBDT + 12 structural features** | **0.99032** |

- Structural features: num1_eq, cos_a/n, ad_tset, ncl_tset, nc_tset, r_margin2, num_jacc, sfx_eq, b_addr_empty, a/b_s1cnt.
- The stacker learns *when* each CE is reliable (house numbers agree, address missing, how common the name is).
- Test: 768,900 seen-country pairs re-stacked; pairs without every CE score keep the logistic p; France untouched.
- **exp28f** = exp28c + rule + two-threshold (+321): India 3.376 / 0.0577, US 3.397 / 0.0573. Validator PASS.
- **Kit:** steps `gbstack_eval` / `gbstack_predict` added; the tail reproduces exp28f byte-for-byte; zip rebuilt.

**exp29g — rival (competition) features in the GBDT stacker (09-27 09:20):**
- **Hypothesis:** stage 2 gets its biggest gains from competition features (a pair vs the record's other S1 candidates, exp02), but the CE scores were never compared across a record's candidates. So add, per record R, *score minus the best score among R's other S1 candidates* (null if R has no other scored pair), for logit p and for each CE (`exp28_gbstack.py --comp r`).
- **Dense validation (same 2-fold cross-fit):**

| stacker | F0.5 |
|---|---|
| exp28c (GBDT + 12 structural) | 0.99032 |
| **+ R-side rivals (4 features)** | **0.99051 (+0.00019)**; India 0.99122, US 0.99004 |
| + R- and S1-side rivals (8) | 0.99047 (the S1 side adds nothing: an S1 has several true records) |

- **Robustness:** GBDT seed 7 gives 0.99032 → 0.99050 (the same +0.00018).
- **Rejected follow-ups:** a second stacking round (rivals of the stacked p): 0.99039 (hurts); the strongest rival S1's structural features (num1_eq, cos_a, cos_n) + the record's candidate count: 0.99051 (= no gain). Both code paths removed; the `r` path is verified unchanged (identical test_pred).
- **exp29gf** (= exp29g + rule + two-threshold): France byte-identical to exp28f; ≈7.8K US/India rows differ; candidate_pairs identical; validator PASS. Expected ≈ +0.00016 LB over exp28f.
  - Regenerated with the kit's XGBoost thread count (6; histogram sums depend on it: 4 threads gave 0.99051, 6 give **0.99050**), +338 rescue pairs.
- **Decode re-check on exp29g OOF** (t_main × t_empty): 0.7 / none 0.99050, 0.7 / 0.6 0.99050, 0.75 / 0.6 0.99052, lower t_empty worse. The rival-feature stacker already handles S1s without an accepted match, so the rescue is now neutral. It stays in the kit (harmless; a change would be noise-level).
- **Kit:** `GB_COMP = "r"` adds `--comp r` to both gbstack steps. The tail (`--from_step stack_eval`) reproduces exp29gf **byte-for-byte**. Zip rebuilt with exp29gf (SHA-256 of both TSVs checked).

**Checks 09-27 ~10:00 (no change adopted):**
- **Raw-level look at in-country wrong-S1 errors on address-less records** (exp29g OOF): only 1,453 accepted FPs. Sampled cases are genuinely identical names (e.g. "Kozhikode Care Private Limited" at two S1s). Irreducible, confirmed.
- **Second candidate filter on stage-2 p** (test, exp15 p2):

| p2 filter | candidates per S1 |
|---|---|
| none (current) | 3.95 |
| > 0.01 | 3.80 |
| > 0.02 | 3.72 |
| > 0.05 | 3.63 |

  F0.5 would be unchanged: such pairs are never accepted. But −4% is small against the ~3.4 true matches per S1, and it adds a filtering stage whose competition features would have to be recomputed. Not adopted.

**H5 per type (09-27 10:05; LOCO exp13 predictions, no-tie typed pairs, true rate by p bin):**

| type | US→India: 0.05–0.3 / 0.3–0.6 / 0.6–0.8 / 0.8–0.9 | India→US: same bins |
|---|---|---|
| brand | 0.27 / 0.44 / 0.56 / 0.69 | 0.24 / 0.43 / 0.64 / 0.79 |
| id_far | 0.42 / 0.68 / 0.77 / 0.85 | 0.06 / 0.20 / 0.30 / 0.40 |
| id_noaddr | 0.07 / 0.60 / 0.82 / 0.92 | 0.51 / 0.56 / 0.69 / 0.86 |
| id_street | 0.36 / 0.52 / 0.73 / 0.80 | 0.28 / 0.50 / 0.79 / 0.86 |
| typo_swap | 0.09 / 0.12 / 0.47 / 0.69 | 0.11 / 0.22 / 0.50 / 0.65 |

- No type is reliably above the ≈0.72 flip break-even below p 0.8, even the brand names that look obviously true by eye (a raw sample of French p 0.5–0.9 pairs is full of them).
- The model's out-of-country p stays calibrated *within* each type. Confirms the H5 rejection type by type.

**Margin-aware decode for an unseen country (LOCO exp13 predictions):**
- Accept only if p − p(second-best S1) > 0.6: India 0.96470 → 0.96481, US 0.97938 → 0.97955 at thr 0.9. Consistent but tiny: +0.00002 LB at France's weight. Not adopted.
- Lone-candidate rescue (p > thr − 0.1 and runner-up < 0.02): −0.001 to −0.003, as before.

**Cleanup (09-27 10:10):** removed superseded work_v3 runs (exp15cr, exp17cr, exp19, exp20, exp20c/cr/f, exp24cr/f, exp28cr, exp10, exp11b, exp14) and the old world's feat_v4 / feat_v5 / feat_v5s2 layers. 29 GB free.
- Kept: the CE runs (exp08/17/18/23/26), exp13/15, exp24c (fallback), exp14r (best LB), exp28f (previous candidate), exp29* and the kit-tail runs (final*).

**Interruption (09-27 10:04–12:05):** the laptop slept, and every process paused, including the e5-base fold-0 scoring. Keep-awake was enabled afterwards.
- New ETA: e5-base fold 1 finishes ≈15:00; exp27 (4-CE logistic) → exp30 (4-CE GBDT + rival) runs automatically after it (`scripts/run_exp30.sh`, log `logs/run_exp30.log`), ≈15:20.
- Deadline for adopting it: 16:30 (kit update + byte check + zip).

**12:40 — France self-training retry: not feasible today.**
- **Evidence:** held-out-country self-training gains best-vs-best +0.0008 (US→India) / +0.0025 (India→US), i.e. ≈ +0.00025 LB at France's weight. The exp16 audit on France was mixed.
- **Cost:** it needs ≈6–8 GB RAM for XGBoost on 3M × 142 features. Only 4.2 GB is free while the e5-base CE trains, so it would risk an out-of-memory crash, and after the CE finishes (≈14:45) there are not the 2–3 h needed.
- exp29gf submitted as an LB anchor.

**Package maintenance (09-27 09:45):**
- `Documentation_template.md`: now describes the three CEs, the GBDT stacker (with a stacking-gain table) and rival features. The withdrawn France alignment is now listed as a *rejected* transfer rule; before, it wrongly appeared as part of the pipeline.
- Kit README: pretrained models (e5-small, bert-base with pinned revisions), step table (ce3, gbstack), runtime ≈11 h.
- `exp06_crossenc.py pairs --test_countries US,India` restricts the test band as exp26 did. Verified: identical train/test band parquets.
- The kit has `USE_CE4` (e5-base, HF id + pinned revision), **off** until validated.

## LB 09-27 evening: exp35 = **0.98891** (best; +0.00102 over exp29gf)
exp35 = **exp29gf** (3-CE GBDT stack with rival features; run chain exp29gf → exp31 → exp32 → exp33 → exp35, thresholds from exp29g) + exp31 US legal-form sibling prior + exp32 French acronym rule + exp33 French descriptor-swap demotion + exp35 French acronym candidate channel. (The 4-CE variant of the same rules is exp34, not submitted.) Expected 0.9891–0.9892; landed at the low end. The bundle is net positive; its parts are not separable from this one score. 4 → 3 submissions left (the organisers added 2).

## LB 09-27: exp29gf = **0.987891** (previous best; previous 0.986936)
- **Decomposition:** US/India on test ≈ dense 0.9905, which puts **France ≈ 0.973** (exp14r ≈0.971). Almost all of the top 50 are ≈0.99.
- **How far France can move:** assigned French pairs per p-bin: (0.7, 0.8] 6.8K, (0.8, 0.85] 4.5K, (0.85, 0.9] 6.3K = 17.6K (0.068 per S1; US 0.028, India 0.013).
  - Lowering France's threshold to 0.7 changes the LB by 0.0102 × (0.285·t − 0.2), where t = their true rate. That is **at most ≈ +0.0005** (t = 0.9), and 0 at t = 0.7.
  - With the ceiling of perfectly resolving French uncertain pairs (+0.0012 LB), **the uncertain band cannot close a 0.002 gap**. Top teams must differ elsewhere: confident French pairs, candidates, or in-country beyond our estimate. Not identifiable without labels.
- **Probe built:** exp29t80 (France threshold 0.8; US/India = exp29gf), `work_v3/runs/exp29t80/output/matching_results_exp29t80.zip`. It tests whether French p in (0.8, 0.9] is mostly true (France under-matching) or not.

## LB 09-27: exp29t80 (France threshold 0.8) = **0.987799** (−0.00009 vs exp29gf)
- It added ≈10.8K French matches (p in (0.8, 0.9]). Solving 0.00625 × (0.285·t − 0.2) = −0.00009 gives **t ≈ 0.65**: French p is over-confident, as the held-out-country tests showed, and **0.9 is at or near France's optimum**.
- **France is not under-matching.** Structural comparison on test (exp29gf decode):

| country | S2/S3 records per S1 | accepted per S1 | share of records accepted |
|---|---|---|---|
| France | 5.53 | 3.254 | 0.588 |
| US | 5.76 | 3.397 | 0.590 |
| India | 5.82 | 3.378 | 0.580 |

  - The lower French match count comes from 4% fewer records per S1; the acceptance share is the same.
  - The matches-per-S1 tail is *lighter* in France (≥7 matches: 2.8% vs 3.5–3.6%), so there is no pile-up of false matches either.
- **Conclusion:** France's ≈0.017 gap to US/India is *swaps*: wrong pairs accepted and true pairs rejected in similar volume, with calibrated but less discriminative probabilities. Thresholds and rules cannot fix that. Only a matcher that discriminates better on French pairs can, which needs French labels (unavailable) or a transfer method that passes the held-out-country test (none did).
- A stricter-threshold probe (exp29t95) is expected ≈ −0.0001 → not worth a slot.

## exp26 — fourth cross-encoder, multilingual-e5-base (`scripts/run_exp26.sh`, resumed by `run_exp26_resume.sh`) · _running_
- **Why:** the CE stack gains came from *diversity*: e5-small v2 +0.00013, bert-base +0.00017. The open question was whether a larger model (278M vs 118M; 12 layers × 768 vs e5-small's 12 × 384) also *reads* the uncertain pairs better.
- **Recipe:** same as exp18/23: frozen embeddings, "name ; address" pairs from exp15's band (0.01, 0.99), 2 epochs, bs 16 (bs 32 ran out of GPU memory), 2-fold cross-fit. Test band restricted to US/India (768,900 pairs).
- **Fold 0 OOF AUC 0.8969**, vs e5-small v2 0.8950, bert-base 0.8991, stage-2 p 0.9388. The larger model reads the hard pairs **no better** than the small ones.
- **Hypothesis now:** capacity is not the bottleneck. At best it adds a little diversity: expected ≤ +0.0001 dense, like the exp20→exp24 step or less.
- **Decision rule, set in advance:** exp30 (4-CE GBDT + rival) replaces exp29gf only if its dense F0.5 > 0.99050. Otherwise exp29gf stays, and the kit's `USE_CE4` stays off (it would add ≈4 h of runtime for nothing).
- **Timeline:** slow, ≈2 h per fold for training + scoring, plus 2 h lost to the laptop sleeping. Fold 1 ends ≈14:45; exp27 → exp30 run automatically (≈15:05).

## Web search (09-27 07:45, user suggestion)
- **Generic queries** (cross-lingual entity matching, domain adaptation) return NER literature, nothing for business-record ER transfer.
- **Public repositories of other participants** exist, e.g. [AvinashMalladi/amazon-ml-challenge-2026-entity-resolution](https://github.com/AvinashMalladi/amazon-ml-challenge-2026-entity-resolution).
  - Approach: inverted-index blocking (top-12 per entity), a 14-feature LightGBM, strict country partitioning, French diacritics stripped. "F0.5 > 0.98", no LB score.
  - A simpler version of our design, with nothing new for France. Read for ideas only; no code used.
- Top teams do not publish during the contest, so further searching is not worth the time.

**Typo-embedding idea (user suggestion):**
- French typo-type swaps are already accepted at 93.3% (in-country 99.4%): ≈1.2K French pairs affected.
- Fuzzy string features and the subword CEs already cover typos, so a typo-embedding model targets a small pocket. Not pursued.

## Pretrained model downloads (standing user permission 09-27: MIT / Apache-2.0, ≤ 8B parameters, ≤ 5 GB)

| model | license | parameters | disk | pinned revision | purpose |
|---|---|---|---|---|---|
| intfloat/multilingual-e5-small (09-26) | MIT | 118M | 0.5 GB | 614241f622f53c4eeff9890bdc4f31cfecc418b3 | cross-encoders v1/v2 |
| bert-base-uncased (already cached) | Apache-2.0 | 110M | 0.4 GB | 86b5e0934494bd15c9632b12f734a8a67f723594 | cross-encoder v3 (exp23) |
| **Qwen/Qwen2.5-1.5B-Instruct** (09-27 06:20) | **Apache-2.0** | 1.5B | 3.09 GB | **989aa7980e4cf806f80c7fef2b1adb7bc71aa306** | exp25: zero-shot multilingual judge for France |
| **intfloat/multilingual-e5-base** (09-27 06:55, local folder `models/multilingual-e5-base`) | **MIT** | 278M | 1.11 GB | **d128750597153bb5987e10b1c3493a34e5a4502a** | stronger in-country cross-encoder |

## exp25 — Zero-shot LLM judge (`exp25_llm.py`, Qwen2.5-1.5B-Instruct) · rejected
**Idea:** a zero-shot multilingual instruct LLM is not fine-tuned on US/India, so it has none of the vocabulary bias that makes the CEs mis-score French noise swaps. Score = logit(Yes) − logit(No) of the first answer token.

**CPU smoke test (8 hand-picked French pairs):**

| case | score |
|---|---|
| exact name, same address | +6.4 |
| brand name at the same address | +6.0 |
| noise swap "Fnath Club SARL" → "Fnath SARL Développement" | +5.6 |
| typo | +1.4 |
| noise swap "Lille Ecole EURL" → "Lille EURL Et Fils" | −0.6 (wrong) |
| descriptor swap "MM Club" / "Mm Comite" | −1.9 (right) |
| descriptor swap "Gilbert Sport" / "Gilbert Amis" | +4.5 (wrong) |
| sister company "Ets Rural" → "Ets Rural Participations", next number | +3.4 (wrong) |

Weak on the distinctions France needs.

**Labeled validation (GPU, 15–30 pairs/s):**
- **In-country uncertain pairs** (3,000, p in (0.05, 0.95), exp20c OOF): AUC **0.52** (random) vs model p 0.836. A cross-validated stack [logit p, llm] gives AUC 0.8362 vs 0.8364 and logloss 0.4975 vs 0.4972: **no signal**.
- **French LB-established groups:** noise-word swaps (true) mean +3.52, 94% "yes"; descriptor swaps (mostly false) mean +2.75, 86% "yes". It barely separates them and says "yes" to anything similar.
- **Verdict:** a 1.5B zero-shot judge cannot resolve these synthetic hard pairs, and a larger LLM does not fit the 4 GB GPU (7B at 4-bit ≈4.5 GB VRAM; the Apache 4-bit Qwen-7B is 5.6 GB on disk). **Rejected; model deleted.**

**Per-source match quota (09-27 09:00, train GT): none.** Matches per S1 per source: S2 mean 1.77, S3 mean 1.89, max 5 / 6; the (S3, S2) count pairs spread over (1,1) 270K, (2,1) 251K, (1,2) 223K, (2,2) 208K, (3,1) 140K, … So a count constraint cannot resolve which same-named S1 an address-less record belongs to.

## Label-free test-vs-OOF count matching (goal LB > 0.99)
Analysis scripts in `src/analysis/` (`census*.py`, `sib*.py`, `acro*.py`, `scan*.py`, …); production code: `exp31_sibprior.py`.

**Checks that confirmed earlier conclusions (no action):**
- IDs / file order: no leak (mod-k agreement = chance, Spearman ≈ 0, file-position corr 0.001).
- The 20K with-address in-country misses are random brand names ("Kornexxylo") on noisy copies of the S1 address. An exact-address key recovers only 979 of 397K exact-address non-candidate pairs (0.25% true): rejected.
- Address-less records with a shared name: the true S1 wins on raw-name similarity only 51–57% of non-tied cases (chance ≈ 50%). Records do copy a sibling record's exact noisy name somewhat more often for the true S1 (27% vs 16%), but that is mostly the record-count prior the model already has. Irreducible, confirmed.
- French no-candidate records (26% of French R) are sister companies (…Participations / Holding / International) and legal-form siblings: no French blocking hole.
- French (0.1, 0.7] band: dominated by descriptor swaps at the same address (LB: false). No hidden recall pool.

**Noise-type census (accepted pairs per 1K S1, by name relation × house-number relation):**
- US test accepted matches the US train truth type by type (identical 2039 vs 2063, swap1 571 vs 586, domain 228 vs 228, reorder 113 vs 113, alias 67 vs 68): the generator's true-record noise mix is the same in train and test.
- France differs by design, not by error: acronyms 48.8 per 1K (US 2.8, India 9.5); French domain-name records have a different house number in 2.3% of cases (US 11.9%, India 20%), so the French generator rarely changes house numbers.

**Acronyms (no acronym feature exists in the pipeline):**
- Acronym = the record's name is one 2–5 letter token equal to the initials of the S1 name words (with or without articles).
- Same house number, unique acronym-matching S1, R's best S1: in-country 99.86% true (8,577 pairs); **held-out country (LOCO) US→India 93–100% and India→US 100% true in every p bin > 0.1**. It is the only fine type that stays near-certain out of country (brand, domain, tag, identical-name types all track p, as found before).
- France: 14,879 such pairs; 2,691 have p in (0.05, 0.9] and are rejected → rule candidate (≈ +0.0001 LB).

**Main finding — test-only legal-form siblings (US):**
- Per fine group (name type × number relation × legal-form side × |Δnumber|), test accepted per 1K S1 vs dense-OOF accepted TP per 1K S1. Every group agrees within ±1–3 except one family: **identical name_core, S1 without legal form, record with one, different first house number** ("Maid Tavern" 21402 → "Maid Tavern LLC" 21411).

| US, that family, \|Δnumber\| | OOF candidates / 1K | OOF true rate | OOF accepted TP / 1K | test candidates / 1K | test accepted / 1K |
|---|---|---|---|---|---|
| ≤ 2 | 19.8 | 0.12 | 1.74 | 46.0 | **6.30** |
| 3–12 | 9.8 | 0.09 | 0.63 | 113.4 | **4.27** |
| 13–100 | 5.8 | 0.49 | 2.58 | 41.0 | **3.72** |
| > 100 | 31.4 | 0.72 | 21.8 | 40.1 | **24.0** |

- The test generator adds 5–12× more of these siblings than the dense world has (the dense world only simulates orphans). The model accepts some of them; the excess over the OOF true count sits at p ≤ 0.98 (near) / p ≤ 0.95 (far), and above that test = OOF.
- India shows no such excess (the model already rejects its siblings). France has no OOF reference.
- US accepted total: test 3397.1 vs OOF 3373.6 per 1K S1 (+23.5; this family +11.1, address-less identical names +6, rest small); India −6.6.

**exp31 (`exp31_sibprior.py`)**: demote (p × 0.5) US pairs of this family with p ≤ 0.98 (|Δ| ≤ 12) or p ≤ 0.95 (|Δ| > 12), then the unchanged exp29gf decode.
- No-op check: with the rule off the output is byte-identical to exp29gf.
- Dense OOF cost: 3,919 accepted pairs demoted (3,578 true) → F0.5 0.99050 → **0.99035** (US 0.99003 → 0.98977).
- Test: 9,156 accepted US pairs demoted (7,535 US rows change; France/India byte-identical; candidates unchanged). By count matching ≈ 2.2K of them are true, ≈ 6.9K false → expected **+0.0005..+0.0008 LB**; worst case (all true) ≈ −0.0005.
- Monte Carlo (P(true) per demoted pair = OOF TP / test count in its (|Δnumber| bucket, p bin) cell; other accepted pairs assumed true): **+0.00082 LB** (sd 0.00001). Requiring disjoint number sets (to skip postcode / apartment parse artefacts) does not separate true pairs (90% vs 92% true in the OOF) → not used.
- Implication for the decomposition: before exp31 the US test was ≈0.002 below its dense validation, so France ≈ **0.976** (not 0.973).
- Checked and benign: the US address-less excess (+6 per 1K) comes from the smaller US test world (S1 names shared by ≥2: dense 45.4%, mean group 18.6; test 40.0%, 12.2); India's test world is larger (53.1% / 18.7 vs 51.9% / 16.4) and shows the mirror deficit. The US "partners" swaps moved from p > 0.98 to 0.7–0.98 in the test (test role statistics), but their accepted total is unchanged (17.6 vs 17.9 per 1K). No group has true pairs pushed below the threshold (largest: 0.4 per 1K).

**exp32 (`exp32_acro.py`)**: French acronym rule on top of exp31. 2,515 French flips (France 3.254 → 3.263 matches per S1). Samples are exact-address acronyms ("CC" ← "Caducee Comite EURL", 13 Rue Leonard Lenoir). Expected +0.0001.

**Word roles in France (R-over-S1 representation, label-free):** injected noise words are over-represented in S2/S3 names (associés 77×, développement 11×, groupe 4.7×, services 2.4×, fils 1.6×, france 1.2×); every descriptor sits at 0.80–0.91× (club, école, amicale, comité, … and also "compagnie", "service"). So a descriptor-for-descriptor swap is a sibling organisation.

**exp33 (`exp33_descswap.py`)**: the model still accepts 2,134 French same-number descriptor↔descriptor swaps (p > 0.9: 761 / 614 / 312 / 447 in (0.9, 0.95] / (0.95, 0.98] / (0.98, 0.99] / > 0.99), e.g. "Pessac Club SARL" → "Pessac Ecole SARL". The rejected ones were ~100% false on the LB (exp14rc/rcd), and the model has no French vocabulary, so its p inside this type is uninformative → demote them (p × 0.5).
- "compagnie" is excluded from the vocabulary: records add "Cie" to the legal suffix (a filler like "& Fils"), and the model accepts 51% of descriptor → compagnie swaps (descriptors 5–20%). Most French "word added, same address" accepts are just Cie → Compagnie (true).
- Other number relations add only 20 accepted descriptor swaps: not extended.
- Expected +0.0002..+0.0003 over exp32 (break-even: 70% of the demoted pairs true).

**exp30 (4-CE GBDT, e5-base CE):** dense 0.99054 (+0.00004 over exp29g). Noise-level. **exp34** = exp30gf + the exp31/32/33 rules (same counts: 9,159 / 2,515 / 2,134), for that base.

**Checks after exp31 (15:20):**
- Per-S1 accepted-count distribution, test vs dense OOF prediction: US empty 5.77% vs 5.75%, k=3 24.42% vs 24.44%, … (India identical too). US S1s without a legal form: test mean 3.4015 (exp29gf) → 3.3691 (exp31) vs OOF 3.3628; S1s with a legal form unchanged (3.3944 vs 3.3818, the benign world-size effect). US/India after exp31 ≈ dense validation.
- France per-S1 distribution is shifted left (k=1 7.0% vs 6.1%, k=2 19.1% vs 17.8%, k≥5 21.6% vs 23.4%; mean 3.255 vs 3.38): consistent with a ~4% French recall deficit spread over the uncertain band (needs better discrimination, no rule).
- French (0.8, 0.9] band (the t80 probe, ≈66% true overall): 10,842 pairs = 1,672 descriptor swaps (false) + 1,119 acronyms (true, exp32) + 8,051 others → others ≈75% true, barely above the ≈71% break-even (≈ +0.00005): not probed.
- Brand names at the same house number: the S1 address being unique (no other S1 at it) matters. p (0.8, 0.9]: unique 0.92 in-country, 0.71 / 0.87 LOCO; shared 0.46 / 0.31 / 0.34. France has 1,827 unique-address brand pairs there → ≈ +0.00002: not worth a probe.
- No US/India group has true pairs pushed below the 0.7 threshold (largest mid-band surplus 0.4 per 1K).
- S1-level completion out of country (LOCO, unseen threshold 0.9): candidates of S1s with 1 accepted match are 0.68 / 0.72 true in (0.8, 0.9] (break-even ≈0.6–0.67) → ≈ +0.00003 for France; empty S1s 0.28 / 0.32 (a French rescue would hurt, as found before); k ≥ 2 below break-even. Not used.

**Step 1 audit (prep_v3, 09-27 16:00):** no score-relevant defect. Measured on labels: "Shree/Sree" S1 vs "Shri/Sri" record (the title strip removes only Shri/Sri) 791 true pairs, recall 0.948 = consistent pairs 0.949; dotted "D.B.A." not split 8,295 true pairs, recall 0.996 (split aliases 0.9999); French Cie (legal form) vs Compagnie (word) 1,682 French pairs, 1,327 accepted anyway; "house number = first number" differs while a number is shared in 11% of true pairs (India 20%) but the model also has number-set features (France 0.4%); CB→club / FS→fils collide with ~150 French acronyms; French apostrophes consistent (800 of 72K split). All would need a full re-run for < +0.0001.

**Step 2 audit (blocking, 09-27 16:45):**
- IVF vs exact search (5K train S1 per country, same fold model, same K's, forward channels): recall loss 0.06pp US / 0.29pp India.
- Funnel (dense world, labels): search union 98.98% → pruning 98.43% (33.4K true pairs lost, two thirds address-less) → filter 98.42%. Field-aware rescue "also keep the S1's top-K by address" adds pairs that are 0.2–0.9% true: rejected.
- **French acronym leak:** French acronym records whose initials match exactly one S1 at the same number + street: 18,018; search finds 94.6%, pruning keeps 79.3% (US/India ≈ 98.5%). Cause: pruning ranks by cos_n + cos_a, an acronym has cos_n ≈ 0, and dense French streets have many same-address competitors. The same join on labels is 99.93% (US) / 98.89% (India) true. → **exp35 (`exp35_acrocand.py`)**: adds the 3,724 missing pairs as candidates with p = 0.95 (skips 152 whose record is accepted elsewhere) → 3,572 added, France 3.255 → 3.269 matches per S1, validator PASS. Implemented as a separate candidate channel after the matcher: editing the pruning would invalidate every cached downstream layer (~11 h re-run), and these pairs are decided by the validated rule anyway, so all other pairs stay byte-identical.
- **French sibling acronyms:** at S1 addresses, acronym records whose letters differ from the S1's initials only in the last letter (e.g. "AM" next to "AF") vs exact matches: France 6,045 vs 19,252 (0.31), US 0.04, India 0.16. Descriptor siblings get acronyms too; with ≈12% first-letter collisions between descriptors, ≈700–800 of the exact French matches are sibling acronyms → the acronym rule is ≈96% precise in France (break-even ≈71%). exp32/exp35 stay positive (≈ +0.0001 each). In India the few pairs pruning lost were only 34% true (89) → the channel stays unseen-country only.
- Hand-off to Step 3: every blocking column is a matcher feature. rk_rev (reverse rank) is not recomputed in the dense world (stale for 38% of pairs) but carries 0.02–0.03% of stage-1/2 gain: harmless. r_margin2 (56% of stage-1 gain) and cos_j (16%) are consistent train vs test: records have 3.23 / 3.18 (US) and 3.28 / 3.22 (India) candidate S1s; true-match margin quantiles match (India 0.391/0.634/0.797 vs 0.401/0.635/0.794).

**Blocking recall target check (user: ">99%"; dense world, labels):** search union 98.98% (US 99.07 / India 98.84) → pruned 98.43% → final candidates 98.42%. Records WITH an address (95.6% of true pairs): 99.65%; WITHOUT an address (4.4%): 71.6%. By S1 name sharing, address-less true pairs reach the matcher 91% (unique) / 89% (2) / 74% (3–5) / 32% (6–20) / 3% (21+; 43K pairs; 8,112 names shared by ≥ 21 S1 cover 497K S1, e.g. "meridian" 572, "family center" 337). Adding every same-name S1 for address-less records: +1.40 candidates per S1, 1.3% precision, recall only 98.95%, and the matcher accepts 3% (6–20) / 0% (21+) of such true pairs even when they are candidates → not done. Score ceilings with today's candidates: perfect matcher **0.99524**; perfect except the ambiguous address-less records **0.99284**; current **0.99050** (US 0.99003 / India 0.99121).

**Step 3 audit (pair features, 124 stage-1 inputs = base 65 + tokens 37 + consensus 9 + roles 13):**
- No label leakage: labels are joined only inside the training functions; the four feature builders never read the ground truth.
- Train (dense world) vs test on exact copies (identical name core and address, record's best candidate; US 280K/165K, India 200K/204K pairs): of 248 feature×country checks only 9 shift (|SMD| > 0.1 or PSI > 0.1), all world-size effects: name-frequency counts (a/b_s1cnt, a/b_rcnt: US test world 663K S1 vs dense 1.06M, India 810K vs 707K), s1_ncand, nm_same_core, and r_margin2 in the US (+0.20 sd; India unchanged). Count features carry < 0.1% of the model gain. No computation bug.
- France vs US (test, exact copies): r_margin2 median 0.78 vs 1.03 (SMD −1.5), hn_sup_r / hn_sup_s1 6 vs 3 / 7 vs 4, fewer name tokens / numbers. Cause: 15% of French S1 share an exact address (US 4%) → the runner-up S1 is a co-located business with a high address similarity. A real data property (normalizing it was rejected before: it carries real ambiguity) → no change.
- Stage-1 gain concentration: r_margin2 56%, cos_j 16% (blocking-derived), rl_e_hneq_min 5%, s1_gap_j 4%.
- **Addendum 18:40, role feature `lrate` is not world-invariant** (the exact-copy drift check could not see it: exact copies have no extra words). lrate = log10((n+1)/N1): n = single-extra-word near-duplicate events of word t, N1 = S1 in the country. (1) Units: on train n comes from the dense world (80% of S1) but N1 counts all train S1 → train values log10(1.25) = 0.097 too low; divided by kept S1 the noise words match exactly (US center −0.898 train vs −0.905 test, services −1.027 vs −1.034). (2) Real world difference: family-role words (swap ≈ 0.01: holdings, group, north, valley; partners 0.17) are +0.24–0.27 higher on test, i.e. ~1.75× more such events per S1; their swap / same-number shares are unchanged or lower, so the extra events are sister-company records. Per-word median shifts (words with n ≥ 100): lrate +0.34 US / +0.22 India; swap, hneq ≤ 0.04 (shares are invariant). Effect on stage 1 (fold-0 model, 11.3M sampled test pairs): the units fix moves 330 pairs up / 550 down across p = 0.5 and 1,051 / 4,495 across 0.01; also removing the family shift drops 2,011 pairs below 0.5 (1,321 France, 681 US, 9 India) and lifts 8. Those are same-address records whose S1 descriptor was replaced by a family-type word ("Baumann Western Inc" → "Baumann Inc Partners", "Forca Amicale SARL" → "Forca SARL Groupe"); exp35 accepts 91% (France) / 93% (US) of them; in the US every CE calls them matches (median logit 3.8–4.5, ≥ 90% positive); in France the CEs are split (exp08 21% positive, exp18 71%, exp23 40%). **Decision: no code change.** The 3 lrate features carry 0.23% of the gain; either fix mostly lowers p1 for pairs the label-free CEs call matches, and it would force a stage-1 retrain and everything downstream. Revisit only if stage 1 is retrained anyway. For Step 4: stage 1 gives these same-address descriptor-swap matches only p ≈ 0.5.
**Step 4 audit (stage-1 matcher: `exp13_rolefeat.py train1` = `exp03_dense.cross_fit`; test p1 in `feats2` = `exp05_tokfeat._predict`), 09-27 19:10:** no score-relevant defect, no code change.
- Setup verified: dense world 33.57M pairs (6.02M true of 6.11M true pairs of kept S1); 2-fold by `s1_fold` (the same hash split in every later stage); all positives + 30% of negatives weighted ×1/0.3 (7.18M / 7.10M rows per fold); XGBoost hist GPU, depth 8, eta 0.1, subsample / colsample 0.8, early stopping 50 on a 3% slice of the other fold; OOF = each fold scored by the other fold's model; test = mean of both models. Labels enter only inside `cross_fit`.
- **Tree-count mismatch (real, harmless):** XGBoost 3.2 keeps the 50 trees built after the best iteration (580 / 644 trees, best 529 / 593). OOF uses `iteration_range=(0, best+1)`, but `_predict` (test p1, and also stage-2 test p) uses all trees. On test (1M sampled pairs) mean |Δp| 3e-4, max 0.09; 694 cross 0.01, 901 cross 0.003 (all-trees keeps 0.2% fewer). With labels (dense world, all-trees OOF vs stored OOF): filter recall identical at 0.001 / 0.003 / 0.01 / 0.03 (0.98431 / 0.98428 / 0.98417 / 0.98383), stage-1 F0.5 0.98752 = 0.98752 @0.8, OOF logloss 0.00747 = 0.00747. → Left as is: a fix would change every cached test prediction downstream for no measurable gain.
- Fold average on test vs single-model OOF: the two fold models differ by median 0.027 / mean 0.067 on the uncertain band (0.01 < p < 0.99); at the 0.01 filter the average keeps 215,416 of 1M sampled pairs vs 214,514 / 215,011 for either model alone. Standard, no change.
- OOF p1 over-predicts in the middle (bin means 0.55 / 0.75 / 0.86 vs true rates 0.49 / 0.66 / 0.80): harmless, the filter thresholds were chosen on OOF p1 and stage 2 re-learns the mapping.
- Stage 1 alone: assign decoding F0.5 0.98752 @0.8 (dense). Filter (Step 5) recall 0.98417 at 0.01 vs 0.98432 before filtering (−0.00015, ≈ 900 of 6.11M true pairs) at 3.65 pairs per kept S1.
- **Why a GBDT and not logistic regression (user question, measured):** same 124 features, same folds and weighting, challengers trained on a 10% sample of the stage-1 rows (≈ 715K per fold; both LRs converged), OOF on all 33.6M pairs (scratch `lr_vs_xgb.py`). Filter losses are at the same number of kept pairs as production p1 > 0.01 (6,444,127).

  | model | logloss | AUC | true pairs lost by the filter | stage-1 F0.5 |
  |---|---|---|---|---|
  | XGBoost, production (7.1M rows/fold) | 0.00747 | 0.999941 | 961 | 0.98752 |
  | XGBoost, same 10% sample | 0.00949 | 0.999905 | 2,832 | 0.98543 |
  | logistic regression, one-hot 32 quantile bins per feature + NaN bin | 0.01518 | 0.999737 | 15,463 | 0.97869 |
  | logistic regression, plain (median + missing indicators, standardized) | 0.02254 | 0.999448 | 34,535 | 0.97072 |

  Per-feature curves and proper sentinel/NaN handling are worth +0.0080 (plain → binned). Feature interactions are worth +0.0067 (binned LR → XGBoost on the same data). The extra data is worth +0.0021. AUC hides all of it (≥ 0.9994 everywhere) because the easy negatives dominate.
- Output to Step 5: `runs/exp13/stage1_fold{0,1}.json`, `s1_oof.parquet` (s1, r, y, p, src), and `feat_v13s2/{train,test}` (p1 + probability-competition + consistency features over all candidates; exp15 reads only p1 and recomputes the rest on the kept pairs).
**French transformation-type scan (09-27 ~21:00, user-requested; scratch `fr_scan.py`, `fr_census.py`, `fr_swap2.py`): nothing rule-worthy.**
- Each record's best pair was typed by name transformation (identical / reorder / spacing / acronym / append / delete / abbreviation / typo / one- or two-word swap / other) × house-number relation (same / different / record without number / without address).
- Out-of-country truth per type and p bin: LOCO exp13 stage-1 predictions (both directions, 2.25M best pairs). France volume: exp35 final p (943K best pairs).
- Out of country, p tracks truth within every type (as found for H5). Only one type clears "true rate ≥ 0.85 in both directions" in a band France rejects: identical name, record with an address but no number, p (0.7, 0.9]. That is 400 French pairs, OOC true 0.86 / 0.89, ≈ +0.00001 LB. No type with OOC true ≤ 0.65 among France's accepts.
- Accepted per 1K S1 on test (exp35):
  - France identical-name same-number +598 and different-number −341 vs the US/India average: the French generator rarely changes house numbers; identical-name totals agree.
  - acronyms +66: already handled by exp32/35.
  - two-word swaps at the same number 120 vs 32–60: checked with the exp33 descriptor vocabulary. Descriptor → other-descriptor multi-word swaps: 2,355 best pairs, only 8 accepted (the model already rejects them). The excess is ordinary noise (a descriptor next to a typo'd or brand word; 64K of 73K accepted).
- **Conclusion:** no French pocket with label or structural evidence strong enough for a rule. The remaining French gap is spread thinly across types, as the per-S1 distribution check suggested (≈4% recall deficit, no concentration).

**Step 5 audit (candidate filter + stage-2 matcher, `exp15_compact.py`), 09-27 20:15:** no score-relevant defect, no code change.
- Filter: train (dense) keeps 6.44M of 33.57M pairs (p1 > 0.01), recall 0.98417 vs 0.98432; test keeps 6.84M of 32.16M (3.95 per S1). Stage 2: 142 features = 124 + p1 + 8 p-competition (rank / gap / sum / count > 0.5 within the S1, rank / margin / runner-up within the record) + 9 consistency features (the record vs the S1's confident anchor candidates, recomputed on kept pairs only). Trained on 3.07M rows per fold, best iteration 131 (both folds). Top gain: p1, p_gap_s1, p_nhi_s1, cos_j. Alone: F0.5 0.98858 @0.75 (stage 1: 0.98752).
- Tree count (same helper as stage 1): 182 trees vs best 131. With labels, all-trees OOF vs stored: F0.5 0.98858 = 0.98858 (best thr), logloss 0.03291 vs 0.03287 → harmless. On test, 7.7K of 6.84M pairs cross 0.75.
- The 18 new features, train vs test, on exact copies that are the record's best candidate (US 792K / 495K, India 351K / 400K): no shift in US or India (|SMD| < 0.1, no missing-rate gap). France vs US-train: only the consistency features are slightly higher (SMD 0.13–0.17; French copies agree more with each other). A data property, not a bug.
- France's looser filter (0.003): 76,233 of 1.14M kept French pairs (0.29 per French S1) have p1 ≤ 0.01; stage 2 gives 1 of them p > 0.5, but the final exp35 accepts 85 (later rules). Tightening to 0.01 would cut 0.044 pairs per S1 overall (3.95 → 3.91) and lose those 85. The organisers' candidate-size weighting is unknown → kept.
- **Final code folder** `final_submission/code/business_entity_resolution/src/`: the 11 modules of Stages 1–3 (byte-identical in content to the code that produced the caches) + `run_pipeline.py` (steps prep → emb → search → pruning/base features → token / consensus / role layers; `--from_step`, `--to_step`, `--list`) + `utils/validate_submission.py`.

**Package integration (done in the final package):** the three rules are post-processing on the final decoded `test_pred.parquet`, so they append to `submission_kit/final/run_pipeline.py` after `final_decode`:
`("sibprior", ["exp31_sibprior.py", "predict", "--run", "final31", "--base", "final", "--metrics_run", "final_g"])`,
`("acro", ["exp32_acro.py", "--run", "final32", "--base", "final31", "--metrics_run", "final_g"])`,
`("descswap", ["exp33_descswap.py", "--run", "final33", "--base", "final32", "--metrics_run", "final_g"])`,
then copy `runs/final33/output/*.tsv` instead of `runs/final/output`. Add the three files to `make_submission.py`'s source list; check the tail byte-identity against the adopted run (exp33 on exp29gf, exp34 on exp30gf); candidate_pairs.tsv is unchanged by all three.

## Code layout
- `src/`: every pipeline and experiment module (`prep_v*.py`, `exp*.py`, `er_common.py`), `run_pipeline.py` (final pipeline), `utils/validate_submission.py`.
- `src/tools/`: reusable analysis tools (leave-one-country-out `loco.py` / `loco2.py`, loss breakdown, decoding checks, drift, determinism check).
- `src/analysis/`: one-off analyses referenced in this log (index in `src/analysis/INDEX.md`).
- `scripts/run_expNN.sh`: the exact command chain of each experiment.
- Caches (`<work>/data/cache/<layer>_vN`) and runs (`<work>/runs/<exp>/`: models, OOF, metrics, outputs) are created by the code and are not part of the repository.

## Open ideas (ranked)
1. Fix the France threshold for exp05 from the probe LBs. Then build a stage 2 that is aware of unseen countries (e.g. stage 2 trained on cross-country stage-1 predictions so it learns to correct out-of-country probabilities).
2. France robustness: features that do not depend on density or country; French abbreviation and department maps; leave-one-country-out validation as a proxy.
3. Better use of house-number evidence (distractors differ by ±1 house number).
4. Blocker: a larger k for the address channel, or a second embedding model, to recover the 1.6% missed pairs (limited upside).

## Final package
- The final package (`src/run_pipeline.py`) runs every step from the raw TSVs: the 4-cross-encoder configuration (`USE_CE4 = True`), then the rules exp31 → exp32 → exp33 → exp35 and the candidate gate exp37 (`FINAL = final37`).
- **Verified:** `--from_step stack_eval` on the cached intermediates reproduced the submitted files byte for byte (SHA-256 of both TSVs).
- **21:45 update: final candidate gate (`exp37_candgate.py`, package step `candidate_gate`, FINAL = final37).** Pairs with stage-2 p ≤ 0.01 can never be accepted by any later stage (CE band 0.01–0.99, France rules need p > 0.01 / 0.1, thresholds ≥ 0.6), so they leave candidate_pairs.tsv.
  - Test: 6,845,820 → 6,591,700 pairs (3.95 → **3.80 per S1**); 0 of 5,828,367 accepted pairs lost; matching_results.tsv byte-identical to exp35.
  - Validation: 3.65 → 3.61 per kept S1, candidate recall 0.98417 → 0.98412 (the dropped true pairs were never accepted).
  - Verified through the runner (final37 = exp37, SHA-256). Zip rebuilt as `!COYS!_submission.zip`; the previous one was renamed `!COYS!_submission_prev.zip`.
- **exp36** = exp35's rules on the 4-CE base (exp34 + acronym channel, thresholds from exp30g). 2,616 US/India rows differ from exp35, France identical, validator PASS. Expected +0.00003 LB (dense +0.00004). Upload file: `work_v3/runs/exp36/output/matching_results_exp36.zip`. If it scores above 0.98891, the package switches to `USE_CE4 = True` and must be re-verified.

## LB 09-27 21:50: exp36 = **0.988929** (best; +0.000019 over exp35). 4-CE base + the same rules.
Package switched to `USE_CE4 = True`. The tail reproduces exp30gf (`final`) and exp36 (`final35`) byte for byte; the gated candidate file comes from `final37`.

## exp38 — self-trained France stage 2 on the exp36 base (22:00, high-variance probe; `exp38_selftrain.py`)
- **Setup:** as exp16 (one labeled S1 fold, 3.07M rows + French pseudo-labels, 150 rounds). The pseudo-labels now also include the LB-validated French rule outcomes: exp32 acronym flips as positives (2,515) and exp33 descriptor siblings as negatives (30,987).
- **Pseudo-labels:** 821K positive, 47.8K negative.
- **Decode:** France at the a-priori 0.98 threshold. The rules are re-applied with p_set 0.99 (exp11 flips 687, exp32 853, exp33 demotes 30,987 of which 383 with p > 0.98, exp35 adds 3,421).
- **Result on test:** US/India byte-identical to exp36. France accepted 848,122 → 838,715 (−13,560 / +4,153); 3.233 matches per S1 (exp36 3.269).
- **Dropped:** identical-name address-less 4.4K (OOC truth 0.62), two-word swaps 2.5K (0.82), identical name at another number 1.8K (0.86), identical name at the same number 1.0K (0.99).
- **Added:** mostly one-word appends at the same number (1.5K, the pattern suspected false in exp14rcd).
- **Expected:** ≈0 ± 0.0005. The audit is mixed, as it was for exp16. Upload file: `work_v3/runs/exp38d/output/matching_results_exp38.zip`.


## Final result
- **Best public LB:** exp36 = **0.988929**.
- **Final standing:** team !COYS! finished in the **top 50** of 32,000+ teams (89,000+ registrations), earning Pre-Placement Interviews for the Amazon Applied Scientist Intern role.
