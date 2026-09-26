# Experiment Log — Amazon ML Challenge 2026 (Business Entity Resolution)

**Metric:** macro F0.5 per S1 entity (precision-weighted; singletons score 1 only if the prediction is empty).
**Validation ("OOF"):** 2-fold cross-fit by S1 over the *whole* train world. Blocking runs over the full train S2/S3 pool, exactly like test. From exp03 on, validation uses a *dense world* (see exp03), because the normal world proved optimistic.
**Target:** LB ≈0.989–0.99 (leaderboard top-3 ≈0.989 on 09-26; we are ≈53rd at 0.9829). **Current best LB:** **0.986936** (exp14r, 09-26 night). Submission limit: 5/day.

## Scoreboard

| Exp | Idea (one line) | OOF (normal world) | OOF (dense world k80) | Public LB | Status |
|---|---|---|---|---|---|
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
- **Model:** `intfloat/multilingual-e5-small` (MIT, 118M; downloaded with the user's OK) as a pair classifier on lowercased raw "name ; address" text pairs, max 96 tokens (mean 45). The embedding matrix is **frozen**: 96M of the 118M parameters, 1.8× faster (728 pairs/s), 2.1 GB VRAM, and unseen French/Hindi tokens keep their pretrained meaning.
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

## Workspace map
- `prep_v1.py`, `exp01_block.py`, `exp01_match.py`, `exp02_stack.py`, `exp03_dense.py`: one file per structural experiment. Earlier files are frozen and later ones import them.
- `er_common.py`: shared IO, metric, folds, submission writer. Paths can be overridden with `ER_DATA_DIR` / `ER_WORK_DIR`.
- `tools/`: analysis scripts (plus `loco.py` = leave-one-country-out and `decode_variant.py` = per-country threshold re-decoding).
  - `error_analysis.py`: FP/FN samples.
  - `prune_curve.py`: recall vs candidate count.
  - `blocker_misses.py`: pairs the blocker misses.
  - `drift.py`: train vs test feature distributions by country.
  - `expected_f.py`: F0.5 estimated from probabilities, per country.
- `scripts/run_expNN.sh`: the exact command chain of each experiment.
- `logs/`: stdout of every long run.
- `data/cache/<layer>_vN`: cached layers. `runs/<exp>/`: models, OOF, metrics, `output/*.tsv`.
- `make_submission.py` + `submission_kit/exp01/`: final zip builder (only needed for the final package).
- **Storage policy** (disk is tight). Keep:
  - `prep_v1`/`prep_v2`, the blocker models (`emb_v1/model_fold*.pt`), train candidates (`cand_v1/train`), and the feature layers the latest experiment uses;
  - the test predictions of the current best run (for re-decoding) and the best submitted TSV;
  - all small models, metrics and logs.

  Delete superseded feature layers, OOF/test prediction parquets and output TSVs of old runs. Hashed embedding features (`*.npz`) are deleted and regenerated with `exp01_block.py feats` (~3 min) when needed. The 09-25 cleanup freed 17 GB (cache ~30 → 12 GB).

## Open ideas (ranked)
1. Fix the France threshold for exp05 from the probe LBs. Then build a stage 2 that is aware of unseen countries (e.g. stage 2 trained on cross-country stage-1 predictions so it learns to correct out-of-country probabilities).
2. France robustness: features that do not depend on density or country; French abbreviation and department maps; leave-one-country-out validation as a proxy.
3. Better use of house-number evidence (distractors differ by ±1 house number).
4. Blocker: a larger k for the address channel, or a second embedding model, to recover the 1.6% missed pairs (limited upside).
