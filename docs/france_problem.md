# The France problem: everything we tried and what it gave

_Amazon ML Challenge 2026, business entity resolution. Compiled 2026-09-27 evening from `EXPERIMENTS.md` and the analysis notes. All "dense" numbers are validation with labels (US + India); all "LB" numbers are public leaderboard scores._

---

## 1. The problem in one paragraph

The training data contains only **US and India**. The test set also contains **France**: 259,452 of 1,732,544 test S1 entities, **15%** of the score. France has no labels at all. Every model we train learns US/India name and address conventions (their noise words, their sister-company words, their address formats) and applies them to French records it has never seen. On US and India the pipeline scores ≈0.9905. On France it has been 0.01–0.035 lower throughout the competition, and that gap is the largest single reason we are below the top of the leaderboard (≈0.99).

## 2. How we measure France without labels

| Tool | What it is | Used for |
|---|---|---|
| **LB decomposition** | LB ≈ 0.85 × (US/India score) + 0.15 × (France score). A diagnostic submission with every French row emptied (exp10_diag, LB 0.849712) showed that US/India on test ≈ dense validation, so France can be solved for from any LB score. | Tracking France's score across submissions (±0.003–0.005) |
| **LOCO** (leave-one-country-out) | Train on US only and score India, and vice versa, with labels. The only labelled imitation of "unseen country". Harsher than France (one training country instead of two). | Checking whether a transfer idea helps *before* submitting |
| **Label-free count matching** | For each fine pair type (name relation × house-number relation × legal form …), compare "accepted per 1,000 S1 on test" with "true accepted per 1,000 S1 in validation". The true-record noise mix is identical in train and test, so an excess or deficit points to a mistake. | Finding specific French (and US) error pockets without labels |
| **LB probes** | One targeted change per submission. | Settling questions no offline tool can (e.g. are descriptor swaps true?) |

## 3. France's score over time (LB decomposition)

| Stage | Best LB | France ≈ | What changed for France |
|---|---|---|---|
| exp01 / exp04 (09-25) | 0.978934 / ≈0.979 | ≈0.95 | nothing French-specific; model over-accepts in an unseen country |
| exp04_fr90 | ≈0.980 | ≈0.957 | stricter threshold (0.9) for unseen countries |
| exp05 | 0.982907 | ≈0.966 | token-alignment and house-number features (transfer well) |
| exp10 | 0.985008 | ≈0.960 | (in-country gains; France slightly worse) |
| exp11b (09-26) | 0.986449 | ≈0.969 | **word-role rule** for French noise words |
| exp14r | 0.986936 | ≈0.971 | word-role features inside the model |
| exp29gf (09-27) | 0.987891 | ≈0.973–0.976 | (in-country stacking gains) |
| **exp35** (09-27 evening) | **0.98891** | **≈0.978 (0.976–0.980)** | French acronym rule + acronym candidates + descriptor-swap demotion (+ US sibling rule) |

France has moved from ≈0.95 to ≈0.978. US/India sit at ≈0.9905. Closing France fully would add ≈+0.0016 LB. Reaching 0.99 with in-country unchanged needs France ≈0.987.

## 4. What is different about French data (diagnosis, all measured)

| Finding | Evidence | Consequence |
|---|---|---|
| French addresses end in a *region* in S1 but in a *department* (or nothing) in S2/S3 | 95% of S1 vs 33% of records use the region | fixed by region canonicalization (exp04) |
| French businesses are packed together | 15.2% of French S1 share an exact address (US 4.0%, India 4.9%); record's margin to its runner-up S1: France median 0.51 vs US 0.79 | every French record looks "contested"; the most important feature (`r_margin2`) is lower in France |
| The generator uses **French noise words** the model never saw | associés 77×, développement 11×, groupe 4.7×, services 2.4×, fils 1.6×, france 1.2× over-represented in records | true matches with these words were rejected (fixed by exp11) |
| French **descriptor siblings**: "Pessac Club" and "Pessac Ecole" at the same address are different organisations | descriptors are *not* over-represented in records (0.80–0.91×); LB rejected accepting them (exp14rcd) | demoted (exp33) |
| **Acronyms** are 17× more common | acronym records 48.8 per 1K S1 (US 2.8, India 9.5) | acronym rule + candidate channel (exp32, exp35) |
| French records rarely change the house number | domain-name records with a different number: France 2.3%, US 11.9%, India 20% | the model's number features mean slightly different things in France |
| France is **uncertain everywhere**, not missing volume | uncertain pairs per S1: France 0.43, US 0.25, India 0.11; share of records accepted France 0.588 = US 0.590 = India 0.580 | France's gap is *swaps* (wrong pair accepted, right one rejected), not a recall hole |
| French blocking is fine | exact (name, number) French pairs retrieved 99.25% (US 99.84%, India 96.87%) | nothing to fix upstream, except acronyms (exp35) |
| French probabilities are **over-confident but calibrated within types** | exp29t80 probe: French p in (0.8, 0.9] is ≈65% true; LOCO: out-of-country p tracks truth within every pair type | threshold 0.9 is France's optimum; blanket type-level corrections hurt |

## 5. Everything we tried

Legend: ✅ adopted (in exp35) · ❌ rejected · ⚪ built but not submitted · 🔬 diagnostic only

### 5.1 Normalization and features (help France through better, language-agnostic signals)

| # | Idea | Result | Status |
|---|---|---|---|
| exp04 | Canonicalize French regions/departments to one token (like US state codes) | France address cosine 0.82 → 0.904 (train 0.91); competing S1 per record 11.1 → 5.7; France pairs +24K/−7K | ✅ (folded into prep_v3) |
| exp05 | Token alignment (typo vs swapped word) + house-number difference features | LOCO US→India 0.950 → 0.962, India→US 0.971 → 0.978 (bigger gain out of country than in); LB +0.003 | ✅ |
| exp07 | prep_v3 normalization (French `et` = `&`, SCI/EI, frs/cb abbreviations, French street types, bis/ter) + **noise-word score** features computed from each split's own data | +0.0003 in-country; French word scores come from French test records | ✅ |
| exp09 | Cluster-consensus features (does the record's odd word recur among the S1's other candidates?) — structural, should transfer | LOCO average +0.0002 only | ✅ (in model) |
| exp13 | **Word-role features**: per-country tables learned from test data (does a word *replace* another at the same address = noise, or get *appended* at another address = sister company?) | LB +0.00049 (exp14r), helps France | ✅ |
| Regularization sweep | Monotone constraints, shallower trees, dropping count/length/cosine features to force "generic" behaviour | every restricted variant is **worse** out of country | ❌ |
| Typo-embedding model | A dedicated typo model | French typo swaps already accepted 93%; ≈1.2K pairs at stake | ❌ not pursued |

### 5.2 Thresholds and decoding for France

| # | Idea | Result | Status |
|---|---|---|---|
| exp04_fr90 | Threshold 0.9 for unseen countries (0.7 in-country) | LB +0.001 (≈ +0.007 on France) | ✅ |
| exp05_fr95 / fr80 | 0.95 / 0.8 instead of 0.9 | LOCO curve flat 0.9–0.95 | ⚪ not worth a slot |
| exp29t80 | France threshold 0.8 | **LB 0.987799 (−0.00009)** → French p in (0.8, 0.9] ≈65% true; 0.9 confirmed optimal | ❌ (LB-tested) |
| exp29t85 / t95 | 0.85 / 0.95 | expected ±0.0001–0.0002 | ⚪ |
| Two-threshold decode (lower threshold for S1s with no match) | helps in-country (+0.00014) | **hurts out of country**: lone candidates of empty S1 are mostly distractors | ❌ for France |
| Margin-aware decode | accept only if p − runner-up > 0.6 | +0.00002 LB | ❌ |
| S1-level completion / lone-candidate rescue | add the next candidate for S1s with 0–1 matches | −0.001 to −0.003 (LOCO) for empty S1s; ≈ +0.00003 for 1-match S1s | ❌ |
| Expected-F0.5 decoder | per-S1 optimal subset | +0.00005 | ❌ |

### 5.3 Word-role rules (fix the unknown French vocabulary directly)

| # | Idea | Result | Status |
|---|---|---|---|
| **exp11a** | Accept pairs where a French **noise word** (fils, associés — found without labels as "swapped in at the same address, over-represented in records") replaced one S1 word at the same house number | **LB +0.00084** (13,944 French rows) | ✅ |
| **exp11b** | + dual-use words groupe / développement / france | **LB +0.00144 total** (25,185 rows) | ✅ |
| exp11c / exp14rc / **exp14rcd** | Also accept **descriptor swaps** (Club ↔ École) and dual-use words *appended* at the same address | **LB 0.983376 (−0.0031)**: +33K French matches, mostly sibling organisations | ❌ (LB-tested; lesson: never bundle untested rules) |
| exp12 | Translate French words into training vocabulary (fils → services, groupe → partners, family words → holdings) and recompute test features | +24.1K / −5.7K accepts; mapping several family words onto one token made sister companies look like duplicates (~4K+ wrong accepts) | ❌ |
| exp12s / sr | Translate only pure noise words, each to a distinct training word | +7K / −2.3K vs exp11b, expected ≈ +0.0001 | ⚪ superseded by exp13 |
| **exp33** | Demote French **descriptor ↔ descriptor** swaps the model still accepted (43-word vocabulary found without labels; "compagnie" excluded because records add "Cie") | 2,134 pairs demoted; part of the exp35 bundle | ✅ |

### 5.4 Acronyms and candidates (exp32 / exp35)

| # | Idea | Result | Status |
|---|---|---|---|
| **exp32** | Accept records whose name is the S1's initials, same house number, unique acronym S1 | in-country 99.86% true; **the only pair type that stays near-certain out of country** (LOCO 93–100% in every p bin > 0.1); 2,515 French flips | ✅ |
| **exp35** | Acronym **candidate channel**: pruning dropped 20% of French acronym matches (acronyms have name similarity ≈ 0 and French streets are dense); add them back | 3,572 pairs added, France 3.255 → 3.269 matches per S1 | ✅ |
| Sibling-acronym check | Do descriptor siblings also get acronyms ("AM" next to "AF")? | yes: rule ≈96% precise in France (break-even ≈71%) | ✅ kept |

### 5.5 Structural "alignment": accept French pair types that are near-certain in-country (exp19, H5)

| # | Idea | Result | Status |
|---|---|---|---|
| exp19 | Flip rejected French pairs of types that are ≥96% true in US/India (brand name at the same address, identical name at a far number, address-less unique name, street-noise, typo swaps): 19,615 flips | **LOCO with labels: −0.0007 to −0.0042**. Out of country the *flipped* subset is only 46–62% true: the model's p is already calibrated within each type | ❌ withdrawn before submission |
| Per-type re-check (09-27) | Same, bin by bin | no type is above the ≈0.72 break-even below p 0.8 | ❌ |
| Brand names at a unique address | narrower version | ≈ +0.00002 | ❌ |
| French (0.8, 0.9] band "others" | would they be worth accepting? | ≈75% true vs ≈71% break-even: ≈ +0.00005 | ❌ |

### 5.6 Domain adaptation (make the model itself behave better on an unseen country)

| # | Idea | Result | Status |
|---|---|---|---|
| Importance weighting (exp09 era) | Train with weights = odds that a pair "looks like" the target country | LOCO +0.0024 average (≈ +0.0004 LB) | — superseded |
| Importance weighting (exp21, current features) | same, on today's features | **hurts both directions** (−0.0009 / −0.0008): role features already absorb it | ❌ |
| Per-country rank normalization | Convert features to within-country ranks | LOCO 0.872 (broken calibration) | ❌ |
| Quantile mapping (H1) | Map French density features (margins, cosines) onto the training distribution | LOCO −0.0006 to −0.005: embedding closeness carries real ambiguity, not a nuisance shift | ❌ |
| Stage 2 on unseen countries | Does stage 2 hurt France? | no: two-stage LOCO slightly *better* than stage 1 | ✅ kept |

### 5.7 Self-training on French test data (exp16)

| # | Idea | Result | Status |
|---|---|---|---|
| exp16 / 16r | Pseudo-labels from our own confident French predictions (818K positives incl. 16.5K LB-validated rule flips, 40K negatives); stage 2 retrained on train + French pseudo-labels | LOCO +0.0017 at a fixed 0.98 threshold. On France: −14.8K / +3.7K matches; the drops are patterns that are 92–99.9% true in training, the adds are the suspected-false appended words | ❌ (model re-learns its own mistakes: the hard pairs are exactly where pseudo-labels are least reliable) |
| Retry 09-27 12:40 | held-out-country gain ≈ +0.00025 LB, needs 6–8 GB RAM | not feasible alongside CE training | ❌ |

### 5.8 Neural models and LLMs (understand French text directly)

| # | Idea | Result | Status |
|---|---|---|---|
| exp06 | Multilingual cross-encoder (e5-small, reads raw "name ; address" text) on uncertain pairs, used for all countries | in-country +0.0019, but trained on US and applied to India: AUC 0.743 (0.90 in-country), F0.5 **−0.0009** | ❌ for France |
| exp06_ood | Same CE for France at a low, transfer-calibrated weight (0.22) | changes only 9K French pairs, ≈ ±0.0002 | ⚪ |
| CE on LB-established French truths | Does the CE understand French noise swaps? | **anti-correlated**: known-true fils/associés swaps CE mean −3.9 (4% positive), known-false descriptor swaps −0.77 (37% positive): it memorised US/India vocabulary | ❌ CE stays off for France |
| exp18/23/26 | More CEs (e5-small v2, bert-base, multilingual e5-base 278M) | help in-country (+0.0003 total); bigger model reads hard pairs no better (AUC 0.897 vs 0.895) | ✅ US/India only |
| **exp25** | Zero-shot multilingual LLM judge (Qwen2.5-1.5B-Instruct), no US/India bias | in-country hard pairs AUC **0.52** (random); on French groups says "yes" to 94% of true noise swaps and 86% of false descriptor swaps | ❌ |
| Bigger LLM (7B+) | same idea with more capacity | does not fit the 4 GB laptop GPU; possible on Kaggle, untested | ⚪ open |

Why language models struggle here: the data is **synthetic**. Whether "Club → Comité" at the same address is the same entity is a convention of the data generator, not a fact of French. The signal lives in the data's statistics (which words get swapped at the same address, which get appended elsewhere), which is what the role tables and count matching mine.

### 5.9 Hypotheses tested and rejected with labels (no submission needed)

| Hypothesis | Result |
|---|---|
| France loses a *volume* of matches (blocking) | no: record coverage 100%, acceptance share equal to US/India |
| French names collide more | no: 50% of French S1 names shared vs 40% US / 53% India |
| Shared addresses drive French uncertainty | no: 0.48 vs 0.43 uncertain pairs per S1 |
| Descriptor-swap records are orphan clusters of a dropped sibling (H3) | no: 34,334 of 34,367 have no support |
| Field-level contrast between best and runner-up S1 fixes wrong assignments (H4) | no: only 65 of 22,671 cases have a field signal |
| French no-candidate records are a blocking hole | no: they are sister companies and legal-form siblings |
| France is under-matching (fewer matches than it should) | no: fewer matches only because France has 4% fewer records per S1 |

### 5.10 Findings from today's pipeline audit (Steps 1–5)

| Finding | France relevance | Decision |
|---|---|---|
| Role feature `lrate` is shifted between train and test (a units bug + more sister-company records in test) | removing the shift would lower p1 for 1,321 sampled French same-address pairs that the final pipeline accepts; French CEs are split on them | no change (0.23% of model gain; not clearly beneficial) |
| France's looser candidate filter (p1 > 0.003) | admits 76K extra French pairs (0.29 per French S1); only 85 end up accepted (via the acronym rule) | kept (recall-safe) |
| Prep audit: French "Cie" (legal form) vs "Compagnie" (word), CB→club / FS→fils collide with ~150 acronyms | < +0.0001 | no change |

### 5.11 Transformation-type scan (09-27 ~21:00)

| Check | Result | Status |
|---|---|---|
| Each record's best pair typed by name transformation × house-number relation. Cross-country truth per type and score band vs French volume. | Probabilities track truth within every type out of country. Only pocket: identical name, record without a house number, p 0.7–0.9 (400 French pairs, ≈86% true) → ≈ +0.00001 LB | ❌ not worth a rule |
| French accepted-per-1K by type vs US/India | Differences are known generator traits (house numbers rarely change, acronyms). The two-word-swap excess is ordinary noise. Descriptor→descriptor multi-word swaps are already rejected (8 of 2,355 accepted). | ❌ nothing to fix |

## 6. LB submissions that tested France directly

| Submission | LB | Δ | What it established |
|---|---|---|---|
| exp04 → exp04_fr90 | ≈0.979 → ≈0.980 | +0.001 | unseen countries need threshold ≈0.9 |
| exp10_diag | 0.849712 | – | France ≈0.96, US/India on test = validation |
| exp11a | 0.985847 | +0.00084 | fils/associés are noise words |
| exp11b | 0.986449 | +0.00144 | groupe/développement/france swaps are noise too |
| exp14r | 0.986936 | +0.00049 | role features help France |
| exp14rcd | 0.983376 | −0.0031 | descriptor swaps / appended dual-use words are siblings (false) |
| exp29t80 | 0.987799 | −0.00009 | French p (0.8, 0.9] ≈65% true; keep 0.9 |
| **exp35** | **0.98891** | +0.00102 | acronym rule + candidates + descriptor demotion (+ US sibling rule): net positive |

## 7. What we learned

1. **What transfers:** language-agnostic structure (typo vs swapped word, house-number arithmetic), statistics computed from each country's own unlabeled data (noise scores, word roles, count matching), and one near-universal pair type (acronyms).
2. **What does not transfer:** anything fitted to US/India text (cross-encoders, vocabulary), blanket corrections by pair type, feature re-normalization, pseudo-labels.
3. **Out-of-country probabilities are informative and roughly calibrated.** France is not systematically too strict or too loose (threshold probes confirmed 0.9). The model is simply less discriminative there: it swaps right and wrong pairs in similar volume.
4. **Only targeted fixes of a specifically identified failure have worked** (French noise words, descriptor siblings, acronyms), each worth +0.0001 to +0.0014.
5. **Never bundle untested French rules** (exp14rcd cost −0.0031).

## 8. What is left, with honest expectations

| Option | Expected LB gain | Cost / risk |
|---|---|---|
| Another round of label-free count matching on French pair types | +0.0001–0.0003 per validated rule | 30–60 min each; has worked three times; each rule needs a submission to confirm |
| 7B+ multilingual LLM on Kaggle (T4 × 2), validated first on labelled US/India hard pairs, then applied to ~70–100K uncertain French pairs | 0 to +0.0005 (the 1.5B model was random) | 2–3 h; data is synthetic, so linguistic knowledge may not help |
| Cross-encoder fine-tuned on French pseudo-labels | uncertain sign | same confirmation-bias problem as exp16; hours of GPU |
| France to US/India level (theoretical ceiling) | +0.0016 | would need a French matcher as discriminative as the in-country one, i.e. French labels |

**Bottom line:** France has gone from ≈0.95 to ≈0.978 through normalization, word-role statistics learned from the French test data, and targeted rules. Every general-purpose transfer technique (domain adaptation, self-training, multilingual cross-encoders, a zero-shot LLM) was tested with labels and failed or was neutral. The remaining French gap (≈0.012, worth ≈0.0016 LB) is spread thinly over many uncertain pairs rather than concentrated in a pattern a rule can fix.
