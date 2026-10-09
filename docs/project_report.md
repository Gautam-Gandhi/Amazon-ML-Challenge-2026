# Business Entity Resolution at Scale
### Project report: Amazon ML Challenge 2026 · Team !COYS!

*Rithvik Achutuni · Gautam Gandhi · Deepak Pandey · Manas Inamdar*

---

## At a glance

| | |
|---|---|
| **Task** | For every business record in a reference source (Source 1), find all records in two other sources (Source 2 and Source 3) that describe the same business. |
| **Scale** | 1.73M reference businesses and about 10M candidate records in the test set; 2.2M reference businesses in training. |
| **Metric** | Macro F0.5 per reference business (precision counts twice as much as recall). |
| **Twist** | Training data covers the US and India only. The test set also contains **France** (15% of the businesses), with no labels at all. |
| **Best public leaderboard score** | **0.988929**, up from 0.978934 for our first submission |
| **Final standing** | **Top 50** overall, which earned Pre-Placement Interviews for the Applied Scientist Intern role |
| **Candidate set** | **3.80 candidates per reference business** (from 10M records), with 98.4% of true pairs kept |
| **Hardware** | One laptop: Intel i5-12450H (12 threads), 16 GB RAM, NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM |

**Final pipeline in one line:** text normalization → learned embedding blocker with GPU nearest-neighbour search → 124 pair features → two-stage gradient-boosted matcher with a learned candidate filter → four fine-tuned cross-encoders and a gradient-boosted stacker (US/India) → word-role and structural rules for the unseen country (France) → assignment decoding.

---

## 1. The challenge

### 1.1 What we had to build
The competition gave us three sources of business records. Each record has only four fields: `entity_id`, `business_name`, `business_address` and `country`.
- **Source 1 (S1)** is the reference list.
- **Source 2 and Source 3 (S2/S3)** are noisy copies of S1 businesses, mixed with records of businesses that are not in S1 at all.

For every S1 business we had to output the list of S2/S3 records that belong to it. That file, `matching_results.tsv`, is what the leaderboard scores.

### 1.2 How it is scored
The metric is **F0.5 computed per S1 business and then averaged** over all S1 businesses:
- F0.5 weights precision twice as much as recall, so merging two different businesses (a false match) hurts more than missing a link.
- Every S1 business counts equally. An S1 business with no true match scores 1.0 only if we predict nothing for it, and 0.0 if we predict anything.

So the metric punishes "greedy" matching, and it rewards knowing when to say "no match".

### 1.3 Rules and the late update
- No external data, lookups or APIs (business registries, geocoding, and so on).
- Pretrained models are allowed only if they are MIT or Apache-2.0 licensed and have at most 8B parameters.
- The final package must contain code that regenerates the output files from the raw data.
- **Organisers' update, during the competition:** the candidate set (`candidate_pairs.tsv`, the pairs the final model scores) also counts in the final ranking. A **smaller candidate set per S1 business ranks higher**, because blocking has to scale to billions of records in real life. This changed our priorities halfway through (Section 10).

---

## 2. Our setup and way of working

### 2.1 Hardware and constraints
Everything ran on a single Windows 11 laptop:
- **CPU:** Intel i5-12450H, 12 threads.
- **RAM:** 16 GB. Out-of-memory crashes were a real risk with tables of 30M+ rows, so we kept one heavy job at a time, cast everything to 32-bit floats, and processed data in chunks.
- **GPU:** NVIDIA RTX 3050 Laptop, **4 GB VRAM**.
  - Enough for our own embedding models, XGBoost, and fine-tuning small transformer models (about 100–300M parameters).
  - Not enough for a 7B-parameter language model, and not enough to run two GPU jobs at once: when we tried, both slowed down 5–10×.
- **Disk:** limited, and it filled up completely once during a feature build. Cleaning superseded caches became a routine.

We also had access to Kaggle notebooks (T4 ×2 / P100 GPUs). In the end every experiment ran on the laptop, because moving 30+ GB of intermediate data was slower than running locally.

### 2.2 Software stack

| Purpose | Library |
|---|---|
| Data processing | **polars** (fast, multi-threaded DataFrames written in Rust), numpy, pyarrow |
| Fuzzy string matching | **rapidfuzz** (C++ implementations of Levenshtein, Jaro–Winkler, token-sort/set ratios) |
| Text normalization | anyascii (transliteration of any script to Latin) plus hand-written rules |
| Embedding blocker and nearest-neighbour search | **PyTorch 2.9** (CUDA 12.8), with our own GPU IVF index |
| Matcher and stackers | **XGBoost 3.2** (GPU histogram method), scikit-learn (logistic regression) |
| Cross-encoders | Hugging Face **transformers** |
| Environment | Python 3.11 virtual environment, Git Bash and PowerShell |

**Pretrained models used** (all within the license and size rules, each pinned to an exact revision):

| Model | License | Size | Use |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 118M | cross-encoders 1 and 2 |
| bert-base-uncased | Apache-2.0 | 110M | cross-encoder 3 |
| intfloat/multilingual-e5-base | MIT | 278M | cross-encoder 4 |
| Qwen2.5-1.5B-Instruct | Apache-2.0 | 1.5B | zero-shot judge experiment (rejected, deleted) |

Everything else, including the embedding blocker and all the matchers, we trained from scratch on the competition data.

### 2.3 How we worked
We followed a few rules from the start, and they shaped the whole project:
- **Hypothesis first.** Each experiment started from a measured problem. It became a new versioned code file (`exp01`, `exp02`, … up to `exp39`) with a docstring explaining why it exists.
- **One experiment log** (`EXPERIMENTS.md`). Every result was recorded: the hypothesis, the change, the numbers, the conclusion, and the leaderboard score if any. Most of this report is drawn from it.
- **Honest validation.** We only trusted gains measured on a validation setup we had shown to match the leaderboard (Section 5).
- **Leaderboard discipline.** We had 5 submissions per day. Each submission tested at most one untested idea on top of an already-validated base, and we wrote down the expected score before submitting. We learned this rule the hard way (Section 9.5).
- **Reproducibility.** Every entry point fixes its random seeds and uses deterministic GPU kernels. We checked the final package by regenerating the submitted files byte for byte.

---

## 3. First look at the data

Before building anything, we studied the data and checked each finding against the training labels.

- **Size:**
  - Train: 2.2M S1 businesses (US 1.32M, India 0.88M) with their S2/S3 records.
  - Test: 1.73M S1 businesses (US 663K, India 810K, France 259K) and about 10M S2/S3 records.
- **Each S2/S3 record belongs to at most one S1 business.**
  - On average an S1 business has 3.46 true records, about 1.8 per source.
  - 5.6% of S1 businesses have no true record at all ("singletons").
  - This "one record, one business" rule turned out to be the single most useful signal in the whole project.
- **The data is synthetic.** The noise follows clear patterns:
  - **names:** typos, digit-for-letter swaps ("5tar", "8lue"), legal-form changes (LLC ↔ Inc, Pvt Ltd), acronyms, reordered words, random brand names at the same address, native-script names in India (Devanagari, Kannada);
  - **addresses:** street-type abbreviations, reordered components, house-number typos, missing parts, and in France a *department* written where S1 has the *region*;
  - **missing fields:** about 2.7% of S2/S3 records have no address at all.
- **Distractors:** the noise copies are mixed with look-alike records of *other* businesses:
  - sister companies ("Falcon Corp" vs "Falcon Group" at another house number);
  - legal-form siblings at a neighbouring number;
  - in France, sibling organisations ("X Club" vs "X École" at the same address).
- **The test set is denser than training.** Test has 5.5–5.8 S2/S3 records per S1 business against 4.7 in training. Since the number of true matches per business is about the same, roughly **40% of test records are distractors, against 26% in training**.
- **France is a different world:**
  - it has its own vocabulary;
  - its businesses are crowded: 15% of French S1 businesses share their exact address with another business, against 4–5% in the US and India;
  - and we have no labels for it.
- **No leaks:** IDs and file order are random.

---

## 4. Where we started: a scalable baseline

### 4.1 Cleaning the text
Our first normalization pass (`prep_v1`):
- Transliterate everything to Latin script (anyascii), lowercase, remove punctuation, write `&` as "and".
- **Names:**
  - strip junk such as "(ID: 123)" or "| www.x.com";
  - split aliases ("X doing business as Y" → Y is the real name);
  - detect domain-style and hashtag names;
  - move legal forms (LLC, Inc, SARL, Pvt Ltd, …) into a separate field, giving a "core" name without them.
- **Native-script names:** from the training labels we mined a word dictionary. 551,230 of 551,240 native-script names align word by word with their S1 name, which gave 1,347 word translations such as प्राइवेट → private, plus 16 state names.
- **Addresses:**
  - shorten common words to canonical forms (street → st, rue → r);
  - map US and Indian states to codes;
  - extract house numbers.

### 4.2 Blocking: finding candidates without comparing everything
Comparing 1.73M businesses with 10M records means 17 trillion pairs, which is impossible. We needed a **blocking** step that keeps a small candidate list per business while losing almost no true pairs.

**What did not work:**
- **Sparse TF-IDF nearest neighbours:** more than 7 minutes per 50K queries, far too slow.
- **Exact GPU brute force:** about 8 trillion scores per channel; hours of memory-bound computation.

**What we built: a learned two-tower embedding** (`exp01_block.py`):
- Each record becomes a bag of hashed **character 3-grams and words** (2^19 buckets, IDF-weighted). Character n-grams make the model robust to typos.
- Two small "towers" (EmbeddingBag layers, 128 dimensions), one for names and one for addresses, turn each bag into a vector.
- **Contrastive training on the training pairs:** true pairs are pulled together, and other businesses in the same batch are pushed apart. Batches are sorted by name, so each batch is full of hard look-alikes. Singletons and distractors serve as negatives.
- One model per half of the training businesses (cross-fitting). Each training business's candidates therefore come from a model that never saw its labels, just like test.

**Search:** we wrote a **GPU IVF index** in PyTorch:
- k-means clusters of about 1,000 vectors; each query searches its 32 nearest clusters.
- **Four search channels:** best 20 by name, best 20 by address, best 30 by name + address, and a **reverse channel** where each S2/S3 record looks up its own best 3 S1 businesses. The reverse channel guarantees that every record reaches some candidate list.

**Pruning:** keep a pair if it is among the S1 business's best 15 (by combined similarity), or if the S1 business is among the record's best 2.

**Result:**
- The raw union of the channels had 72.7 candidates per business and kept 98.99% of true pairs.
- After pruning: about 18.6 per business and **98.4% recall**.
- A perfect matcher on these candidates would score F0.5 ≈ 0.995, so the blocker was good enough to aim very high.

### 4.3 The first matcher
For each candidate pair we computed **65 features**:
- fuzzy name and address scores (rapidfuzz: ratio, token-sort, token-set, partial, Jaro–Winkler);
- house-number overlap and legal-form agreement;
- record flags (native script, domain name, alias);
- how common the name is;
- the blocker's similarities and ranks;
- **competition features**, i.e. how this pair compares with the business's other candidates and with the record's other candidate businesses.

The model was **XGBoost** (gradient-boosted decision trees):
- Trained on all true pairs plus a random 30% of the false ones, re-weighted so the probabilities stay calibrated.
- Cross-fitted in two folds by S1 business, so every training pair gets an honest **out-of-fold (OOF)** prediction.

**Decoding**, i.e. turning probabilities into matches:
- Because each record belongs to at most one business, every record goes only to its **best-scoring** business, and only if the probability is above a threshold.
- This "assignment decoding" beat plain thresholding (0.98490 vs 0.98441).
- An expected-F0.5 optimizer and separate thresholds per source did not beat it.

**Result:** OOF F0.5 **0.98490** → first submission, leaderboard **0.978934**.

**An early pitfall:** one submission was made before the test features had finished computing, so every US row was empty, and the official validator still passed. From then on we always checked per-country coverage before submitting.

---

## 5. The first surprise: validation gains that never reached the leaderboard

### 5.1 Stage-2 stacking (exp02)
**Idea:** the true records of one business resemble each other. So the second-stage model gets the first model's probabilities as features:
- rank, gap to the best, sum, and count of confident candidates within the business;
- rank, margin and runner-up within the record;
- similarity of each candidate to the business's 3 most confident candidates ("anchors").

**Result:** validation went up by 0.002 (0.98685), with a big gain on singletons (0.978 → 0.988), **but the leaderboard did not move.** Validation was clearly not telling us the truth.

### 5.2 Hypothesis 1: test density (exp03, the "dense world")
The test has about 40% distractors, against 26% in training. A model and threshold tuned in the cleaner training world would accept too much in test.

**Fix:** we built a **dense validation world**:
- randomly drop 20% of the training S1 businesses;
- their S2/S3 records stay, now as distractors, which gives about 41% distractors, close to test;
- all ranks, competition features and name counts are recomputed in that world, and both stages are retrained.

This made validation more honest (the old model drops from 0.9849 to 0.9832 there). But it only explained a part of the gap. Something bigger was missing.

### 5.3 Hypothesis 2: France (leave-one-country-out)
To imitate an unseen country with labels, we trained on the US only and tested on India, and the other way round. We called this **LOCO** (leave-one-country-out). It became our main tool for anything France-related.

| | trained in-country | trained on the other country |
|---|---|---|
| India | 0.98473 | **0.95393** |
| US | 0.98380 | **0.97066** |

**Finding:** the matcher **does not transfer to an unseen country**. It accepts too much there (singletons drop to 0.88–0.92) and needs a much higher threshold. France is exactly this case. A stricter threshold for unseen countries (0.9 instead of 0.7) gave about **+0.001 on the leaderboard (≈ +0.007 on France)**.

We also tried to make the model "more generic": monotone constraints, shallower trees, removing count, length or similarity features. **Every restricted version transferred worse.** Regularization was not the answer.

### 5.4 French addresses (exp04)
French S1 addresses end in the *region* (95%). French S2/S3 records often use the *department* instead (Nord, Gironde, …) or nothing. We mapped every French region and its departments to one token, as US state codes already were:
- median address similarity of French best pairs went from 0.82 to 0.90 (training: 0.91);
- competing businesses per record dropped from 11 to 5.7.

It was clearly correct normalization, and it became part of the final pipeline. On its own it did not move the leaderboard, because the threshold was the bigger problem at the time.

---

## 6. Learning from the errors: better features

### 6.1 What the errors looked like
A loss breakdown on the dense world showed:
- false negatives inside the candidates: 37% of the loss;
- true matches missing from the candidates: 35%;
- false positives: 28%.

The uncertain pairs were dominated by **"family" distractors**:
- the same address with one name word swapped ("Recherche Club" vs "Recherche Pharmacie");
- a house number off by a little (1704 vs 1708).

Whole-string fuzzy scores treat these like typos, which they are not.

### 6.2 Token alignment and house-number arithmetic (exp05)
New features that separate *a typo* from *a swapped word*, and *a dropped digit* from *a different house number*:
- **Word alignment:** each word looks for its best partner on the other side (Jaro–Winkler).
  - We measure the weakest word's score, how many words stay unmatched, how rare the unmatched words are (IDF), and an IDF-weighted coverage, in both directions, for names and addresses.
- **House numbers:** log difference, relative difference, "one is a prefix/suffix of the other" (a dropped digit), edit distance, and how many S1 numbers the record lacks.

**Result:**
- Dense validation **0.98764** (+0.0019).
- **LOCO improved more than in-country:** US→India 0.950 → 0.962, India→US 0.971 → 0.978.
- Leaderboard **0.982907** (+0.003). It helped France too, because these features do not depend on language.

### 6.3 Mining normalization gaps and noise words (exp07)
We compared the word differences of 400K confident pairs and listed every systematic gap:
- alias phrases that were not split ("formerly known as");
- address designators (unit, apt, #, "N°");
- French "et" for "&", French street types, bis/ter;
- Indian state-code variants and renamed cities;
- ordinals, city suffixes, and so on.

We fixed them all in a new normalization (`prep_v3`).

We also added a **noise-word score** for every word, computed from each split's own unlabeled records: how much more often the word appears in S2/S3 names than in S1 names. Words injected by the data generator ("services", "groupe", "associés") are heavily over-represented in S2/S3 names, while real name words are not. The model can now tell *what kind* of word is unmatched, not just that something is unmatched.

We then rebuilt **everything from scratch in one clean, deterministic run** (new folder `work_v3`). After that we fixed seeds, forced deterministic cuBLAS/cuDNN kernels, used eager attention in the transformers, and pinned the model revisions. Two reruns now gave bit-identical models.

**Result:** +0.0003 in-country. The in-country matcher was starting to saturate.

### 6.4 Cluster consensus (exp09)
**Idea:** a distractor business ("Recherche Pharmacie" next to S1 "Recherche Club") usually has its *own* group of 3–4 records, all carrying the different word and the same other house number. Random noise, in contrast, changes records independently. So for each candidate we asked:
- Is this record's odd word **shared by other candidates** of the same business?
- How many candidates contain the S1 word this record lacks?
- How many share the house number?

**Result:**
- In a quick test the signal was strong: the support of a record's extra word was 0.67 for non-matches vs 0.05 for matches.
- Stage-1 log-loss dropped 7%.
- But F0.5 only rose by +0.0002 in-country and +0.0002 in LOCO.

### 6.5 A first plateau
By now every feature idea moved validation by only about +0.0002. Meanwhile out-of-country performance stayed about **0.018 below in-country**. Our conclusion at that point: *better features no longer close the gap to an unseen country; that needs a different kind of technique.*

---

## 7. Reading the text with neural models: cross-encoders

Some decisions need language understanding that hand-made features cannot provide. For example, "Solidair Sportive" vs "Solidair Loisirs" are different clubs, while "Elevate Sportif SAS" vs "Elevate SAS Cie" is noise.

**Cross-encoder (exp06, then exp08 on the clean world):**
- A pretrained multilingual transformer (**multilingual-e5-small**, 118M parameters) reads both records together as text, "name ; address" vs "name ; address", and outputs a match score.
- It only re-scores the **uncertain band** (matcher probability between 0.01 and 0.99): about 725K training pairs and 1M test pairs.
- To fit the 4 GB GPU and run fast, we **froze the embedding matrix** (96M of the 118M parameters). That gives 1.8× the speed and 2.1 GB of VRAM, and unseen French and Hindi words keep their pretrained meaning.
- 2 epochs, AdamW, learning rate 3e-5, 2-fold cross-fitting like everything else.
- A small logistic regression **stacks** the matcher's probability and the cross-encoder score.

**Result:**
- Alone, the cross-encoder was weaker than the matcher on the hard band (AUC 0.90 vs 0.94), but it was **complementary**.
- Stacked, it gave the biggest single in-country gain of the project: **+0.0019** (0.98764 → 0.98954).

**The transfer check changed how we used it.** A cross-encoder trained on the US scored AUC 0.74 on India (0.90 in-country), and with full weight it *lowered* India's F0.5 (−0.0009). It had learned country-specific vocabulary. So we used **cross-encoders only for the US and India**, never for France. Later, when we had French pairs whose true status the leaderboard had confirmed, the cross-encoder turned out to be **anti-correlated** with the truth on them, which confirmed the decision.

**exp10** (consensus features + cross-encoder for US/India): dense **0.98965**, leaderboard **0.985008**.

---

## 8. Measuring France without labels: a diagnostic submission

We needed to know whether the leaderboard gap came from France or from US/India being harder on test than in validation. These two cases need opposite fixes.

**Diagnostic:** we submitted exp10 with **every French row empty**, keeping US/India identical. French businesses then only score on their singletons (about 6%). Solving the two leaderboard scores gave:
- **US+India on test ≈ 0.9896**, exactly the dense validation score (0.98965). The dense world was accurate.
- **France ≈ 0.96**. The whole remaining gap was France.

From then on we could translate any leaderboard score into a France score, and the dense world became fully trusted for US/India.

---

## 9. The France breakthrough: word roles

### 9.1 Why France loses matches
About 10% of accepted French pairs sat in the uncertain band, against 3.5% for US/India. In most of them the S2/S3 name added a word the S1 name lacked: groupe, france, associés, fils, cie, développement.

Studying the training labels showed that the data generator **uses words in two roles, with a separate vocabulary per country**:
- **Noise words** *replace* a descriptor and keep the address ("Obrien Lion LLC" → "Obrien LLC Services"). This is a true match.
- **Family words** are *appended* to the full name of a sister company, usually at another house number ("Falcon Corp" → "Falcon Group"). This is a different business; the true-match rate is 0.2%.

US/India noise words are center, services and partners. In France the noise words are fils, associés and others, **which the model had never seen**. So it rejected true French matches.

These roles can be measured **without labels**, in each country's own test data, from simple statistics over "near-duplicate" pairs (a record and its closest business whose names differ by one word):
- **swap share:** how often the word *replaced* another word;
- **same-number share:** how often the house number stayed the same;
- **over-representation** in S2/S3 names.

| Word | Swap share | Same house number | Model's mean p (swap, same number) |
|---|---|---|---|
| US center / services | 0.83 | 0.63 | 1.00 |
| France services | 0.77 | 0.76 | 0.98 |
| **France fils** | 0.76 | 0.85 | **0.27** |
| **France associés** | 0.76 | 0.80 | **0.50** |
| France international / holding / participations | ~0.01 | 0.02 | rejected (correct: family words) |

### 9.2 The word-role rule (exp11)
**Rule** (country-agnostic, no labels): accept a pair when
- the names differ by exactly one swapped word;
- the house number is the same;
- the swapped-in word has a noise role in its own country.

On the training labels this rule is 99.9% (US) and 98.5% (India) precise. We applied it to unseen countries only.

| Submission | Leaderboard | Change |
|---|---|---|
| exp11a: fils/associés only | 0.985847 | +0.00084 |
| exp11b: all five French noise words | **0.986449** | **+0.00144** |

France rose from about 0.960 to about 0.969. **This was the first method that genuinely helped the unseen country.**

### 9.3 A failed shortcut: translating French words (exp12)
If the model knows US noise words, why not translate French words into them? We mapped:
- fils/associés → services;
- groupe/développement/france → partners;
- family words → holdings.

Then we recomputed the test features.

It backfired: mapping several *different* family words onto one token made whole sister-company families (X Participations, X International, X Holding) look like identical records. About 4,000+ distractors were accepted. A narrower version (pure noise words only, each to a distinct word) was roughly neutral. We dropped the idea.

### 9.4 Giving the model the roles (exp13)
Instead of a rule, we gave the matcher **13 word-role features**: swap share, same-number share, frequency and over-representation of the record's extra words and the S1's missing words. All are computed per split and per country without labels. The model can now learn "a noise-role word swapped at the same address → match" from US/India and apply it to *any* vocabulary.

**exp14r** (role model + cross-encoder + rule): leaderboard **0.986936** (+0.0005). The role model helped France too.

### 9.5 The costly lesson: never bundle untested ideas (exp14rcd)
Encouraged, we added two more French ideas in **one** submission:
- accept French **descriptor swaps** at the same address ("Club" ↔ "École");
- accept dual-use words *appended* at the same address.

Together they added 33,000 French matches. The leaderboard **dropped to 0.983376 (−0.0031)**: both patterns were mostly **sibling organisations**, not noise.

Because the two changes were bundled, the score alone could not tell which part hurt. From then on, **every submission tested at most one untested change on top of a validated base.**

---

## 10. A new requirement: a smaller candidate set (exp15)

After the organisers announced that smaller candidate sets rank higher, we looked at our 18.6 candidates per business (32.2M test pairs) for about 3.4 true matches.

| Option (dense validation) | Per business | Recall | F0.5 |
|---|---|---|---|
| Current blocking | 19.0 | 0.9843 | 0.98842 |
| Tighter blocking cut-offs (top-3 … top-10) | 6.0–11.0 | 0.9756–0.9797 | lower |
| **Learned filter: keep pairs with stage-1 p > 0.01** | **3.65** | **0.9842** | **0.98842 (unchanged)** |

**The learned filter wins clearly.** The stage-1 matcher is cheap and already knows which pairs are hopeless; tighter embedding cut-offs lose real matches.

**Implementation:**
- Keep pairs with stage-1 probability above 0.01 (US/India) or 0.003 (unseen countries, to stay safe on recall).
- **Retrain stage 2 on the kept pairs only**, with its competition features computed over the kept pairs.

**Result:**
- Test candidates went from 32.2M to **6.84M (3.95 per business, −79%)** with unchanged recall.
- Stage 2 even improved slightly (0.98858), because hopeless pairs no longer blurred the competition features.
- `candidate_pairs.tsv` is exactly the set the final model scores.

---

## 11. Squeezing the US/India matcher

With validation trusted, we pushed the in-country score step by step:

| Step | Dense F0.5 |
|---|---|
| Compact candidates + stage 2 | 0.98858 |
| + cross-encoder 1, coverage filled for the new candidate band (exp17) | 0.98985 |
| + cross-encoder 2 (e5-small trained on the new band) + a "two-threshold" decoder (exp20) | 0.98998 |
| + cross-encoder 3: **bert-base-uncased**, for architecture diversity (exp24) | 0.99015 |
| **Gradient-boosted stacker** instead of logistic regression, with 12 structural features (exp28) | 0.99032 |
| + **"rival" features** (exp29g) | **0.99050** |
| + cross-encoder 4: multilingual-e5-base (exp30) | 0.99054 |

Notes on the main steps:
- **Two-threshold decoding:** S1 businesses with no accepted match may take their best candidate at a slightly lower threshold (0.6 instead of 0.7). It is worth +0.00014 in-country. It *hurts* out of country (an unseen country's lone candidates are mostly distractors), so it is used only for the US and India.
- **Diversity beat size:** adding bert-base helped as much as a second e5-small model. The larger e5-base (278M) did not read the hard pairs any better (AUC 0.897 vs 0.895).
- **GBDT stacker:** it learns *when* each cross-encoder is reliable, for example when house numbers agree, the address is missing, or the name is common.
- **Rival features:** cross-encoders score each pair on its own. For each record we added "this score minus the best score among the record's *other* candidate businesses", for the matcher and for every cross-encoder. That gave +0.00019, stable across random seeds.

**Leaderboard:** exp29gf (compact candidates + 3 cross-encoders + GBDT stacker with rivals) = **0.987891**.

**Things that did not help:**
- Tuning stage-2 hyperparameters (depth/learning rate): ±0.00006.
- Ensembling with older models: +0.00005.
- An expected-F0.5 decoder: +0.00005.
- Rival features on the business side, or a second stacking round: no gain.

---

## 12. Hitting the wall: why this approach could not reach 0.99

### 12.1 Where the remaining in-country error is
We measured exactly what an oracle fix of each error category would be worth:
- **Records with no address whose exact name belongs to several businesses** cost **half of all remaining loss**.
  - Example: a record "Meridian" with no address, when 572 S1 businesses are called "Meridian".
  - No method can place these: the information simply is not in the data.
- **Score ceilings with our candidates:**
  - a perfect matcher: **0.99524**;
  - a perfect matcher that still cannot resolve those ambiguous records: **0.99284**;
  - we were at 0.99050.
- Blocking was not the problem: candidate recall is 99.65% for records with an address and 71.6% for records without one, and nearly all of those misses are the ambiguous case.

So the remaining in-country headroom was about 0.002, all of it in genuinely hard pairs.

### 12.2 Everything we tried to transfer better to France
France was at about 0.973 against 0.9905 for US/India. Reaching 0.99 would have needed France at about 0.986. We tried every transfer technique we knew, **always testing on LOCO (labels) before touching France**:

| Technique | What it does | Result |
|---|---|---|
| Importance weighting | Up-weight training pairs that "look like" the target country | Early version +0.0024 out-of-country (≈ +0.0004 LB). With the final features it **hurt** in both directions. |
| Rank normalization | Convert features to within-country ranks | Broke calibration (LOCO 0.87) |
| Quantile mapping | Map the target country's similarity/margin distributions onto training | Hurt: closeness in the embedding carries real ambiguity, not a "shift" to remove |
| Type alignment | Accept French pair *types* that are ≥96% true in-country (brand names at the same address, identical names, typos, …) | LOCO **−0.0007 to −0.0042**. Within each type, the pairs the model rejects are the genuinely doubtful ones, so out-of-country probabilities are already calibrated *within* types. Withdrawn before submitting. |
| Self-training (exp16) | Retrain stage 2 on French pseudo-labels (our own confident predictions + rule flips) | LOCO +0.0008 / +0.0025, but on France it dropped patterns that are 92–99.9% true in training. Not adopted. |
| France threshold probes | 0.8 instead of 0.9 (exp29t80) | Leaderboard −0.00009: French p in (0.8, 0.9] is only ~65% true. 0.9 is the optimum. |
| Margin-aware decoding, lone-candidate rescue | Accept only clear winners / add best candidates to empty businesses | +0.00002 / −0.001 to −0.003 |
| Cross-encoders on France | Use text models for French pairs | Anti-correlated with French pairs of known status |
| **Zero-shot LLM judge** (exp25) | Ask Qwen2.5-1.5B-Instruct "same business?", with no US/India bias | AUC **0.52** (random) on labelled hard pairs; said "yes" to almost anything similar. A larger LLM does not fit 4 GB of VRAM. |

**Why language models struggled:** the data is synthetic. Whether "Club → Comité" at the same address is the same business is a *convention of the data generator*, not a fact of the French language. The useful signal lives in the data's statistics, which is exactly what the role features capture.

We also tested and rejected a dozen more structural ideas with labels:
- blocking volume and larger candidate pools;
- co-location of French businesses;
- margin normalization;
- "orphan clusters" of distractor records;
- field-level contrast between a record's best and second-best business;
- an exact-name key for address-less records;
- French-specific blocking;
- per-source match quotas (none exist in the data);
- a typo-specific embedding.

**Conclusion at this point:** the pipeline was close to its ceiling with this model family. France's gap was not a volume problem: France accepts the same share of records as US/India (0.588 vs 0.590). It was a *discrimination* problem, with wrong and right pairs swapped in similar numbers. That needs French labels, which do not exist.

---

## 13. A new angle: label-free count matching

When feature engineering and transfer learning stopped paying off, we changed the question. Instead of improving the model, we asked: **where do the model's decisions on test disagree with what the training data says should happen?**

### 13.1 The method
The data generator creates the *true* noisy copies the same way in training and in test. So, for every fine pair type (name relation × house-number relation × legal-form relation × size of the number difference), we compared two numbers:
- **test:** accepted pairs per 1,000 businesses;
- **dense validation:** *correctly* accepted pairs per 1,000 businesses.

If test accepts noticeably more of a type than validation says is true, the excess is probably test-only distractors. If it accepts fewer, the model is missing matches. No labels are needed on the test side.

### 13.2 What it found
- **US legal-form siblings** (exp31):
  - Every US type agreed within ±1–3 per 1,000, except one family: an **identical name, the S1 without a legal form, the record with one, and a different house number** ("Maid Tavern" at 21402 vs "Maid Tavern LLC" at 21411).
  - The test generator creates 5–12× more of these siblings than our dense world.
  - Fix: demote those pairs (p × 0.5) below a confidence cut-off. That costs −0.00015 on validation, and a simulation estimated about +0.0008 on the leaderboard.
- **French acronyms** (exp32):
  - French test data has 17× more acronym records than the US ("CC" for "Caducee Comite").
  - An acronym that matches the initials of exactly one business at the same house number is the **only pair type that stays near-certain out of country** (LOCO: 93–100% true in every probability bin).
  - We accept those (2,515 pairs).
- **An acronym leak in blocking** (exp35):
  - Pruning ranks by name + address similarity, and an acronym has almost no name similarity.
  - On crowded French streets, 20% of exact acronym matches were pruned away.
  - We added them back through a separate candidate channel (3,572 pairs).
- **French descriptor siblings** (exp33):
  - A descriptor-for-descriptor swap at the same house number ("Pessac Club" → "Pessac Ecole") is a sibling organisation. The leaderboard had already shown these are false (Section 9.5), yet the model still accepted 2,134 of them.
  - We built the descriptor vocabulary without labels: frequent S1 words that are *not* over-represented in S2/S3 names. Then we demoted those pairs.

### 13.3 Result
- **exp35** (exp29gf + all four count-matching fixes): leaderboard **0.98891** (+0.00102), our first gain of this size since the word-role rule.
- **exp36** (the same rules on the 4-cross-encoder stack): **0.988929, our best score.**

---

## 14. Final checks and a last bet

### 14.1 A stage-by-stage audit
Before finalizing, we re-checked every stage for hidden mistakes, with data:
- **Normalization:** all remaining gaps found were worth less than +0.0001.
- **Blocking:** recall 98.42%, with the losses explained by address-less ambiguity.
- **Features:** we compared train and test distributions on pairs that mean the same thing in both, exact copies. We found two real inconsistencies and measured both as harmless:
  - one role statistic was shifted between train and test;
  - test predictions used 50 more trees than the validation predictions did. With labels: identical recall, F0.5 and log-loss.
- **Why gradient boosting and not a "simple" model?** On the same features, folds and weighting:

| Model | Stage-1 F0.5 | True pairs lost by the candidate filter |
|---|---|---|
| XGBoost (production) | **0.98752** | **961** |
| XGBoost on a 10% sample | 0.98543 | 2,832 |
| Logistic regression, 32 bins per feature | 0.97869 | 15,463 |
| Logistic regression, plain | 0.97072 | 34,535 |

  Per-feature curves and special codes ("not found", "missing") explain half of the gap. **Feature interactions** explain the rest: for example, an identical name is strong evidence for a unique name and weak evidence for "Meridian". A weighted sum cannot express "A matters only when B". (AUC was above 0.9994 for every model, which shows how misleading AUC is here: the contest is decided in the hard 1% of pairs.)

### 14.2 A last scan of French patterns
We typed every French best pair by its transformation and house-number relation, and compared each type against LOCO labels:
- transformations: identical, reordered, acronym, word added or dropped, abbreviation, typo, one- or two-word swap;
- house-number relations: same, different, missing.

Out of country, the model's probabilities tracked the truth *within every type*. The largest remaining pocket was 400 pairs, worth about +0.00001. **No French rule was left to find.**

### 14.3 The final high-risk submission
With only the best score counting, we spent the last experimental submission on the highest-upside idea: **self-training for France** (exp38).
- The pseudo-labels were improved with the leaderboard-confirmed French facts: acronyms as positives, descriptor siblings as negatives.
- France was decoded at the strict 0.98 threshold.

It scored **0.988669 (−0.00026)**. Working back from the score, the ~13,600 French pairs it dropped were about 75–85% true. **France was not over-matching:** a looser decoder (the 0.8 probe) and a stricter one (exp38) both lost points. That confirmed 0.9 as the right French threshold, and that the remaining French gap needs better discrimination, not different cut-offs.

### 14.4 A smaller candidate file for free
Pairs with stage-2 probability ≤ 0.01 can never be accepted by any later stage:
- cross-encoders only re-score 0.01–0.99;
- the French rules need p > 0.01;
- every threshold is ≥ 0.6.

A final gate removes them from `candidate_pairs.tsv`: **3.95 → 3.80 candidates per business**, with zero accepted matches lost and a byte-identical `matching_results.tsv`.

---

## 15. The final pipeline

```
 Raw TSVs (S1, S2, S3; train + test)
        │
 1. Normalization (prep_v3)
        │    transliteration, legal forms, aliases, native-script dictionary, address canonical forms,
        │    French region mapping, house-number extraction
        ▼
 2. Blocking (exp01_block)
        │    two-tower hashed char-3-gram + word embeddings (contrastive, cross-fitted)
        │    GPU IVF search: name@20, address@20, joint@30, reverse@3 → prune → ~18.6 per business
        ▼
 3. Pair features: 124 per pair, in 4 row-aligned layers
        │    base 65 (fuzzy scores, numbers, legal forms, counts, blocking ranks and margins)
        │    tokens 37 (word alignment, noise-word scores, house-number arithmetic)
        │    consensus 9 (does an odd word or number recur among the business's other candidates)
        │    roles 13 (per-country word roles learned from unlabeled data)
        ▼
 4. Stage-1 XGBoost (2-fold cross-fit by business, dense training world)
        ▼
 5. Learned filter p1 > 0.01 / 0.003 → 3.95 per business → stage-2 XGBoost (142 features)
        │
        ├── US / India ──► 6. Four cross-encoders on the uncertain band
        │                     (e5-small ×2, bert-base, e5-base; frozen embeddings, cross-fitted)
        │                  7. GBDT stacker: p + CE scores + 12 structural + rival features
        │
        └── France ──────► 8. Word-role rule for noise words (learned per country, no labels)
        ▼
 9. Assignment decoding (each record → its best business; thresholds 0.7 / 0.6 for US-India, 0.9 for France)
        │    + count-matching rules: US legal-form siblings, French acronyms, French descriptor siblings,
        │      French acronym candidate channel
        │    + final candidate gate (3.80 per business)
        ▼
 matching_results.tsv + candidate_pairs.tsv
```

**Reproducibility:**
- One script (`run_pipeline.py`) runs every step (about 40) from the raw TSVs. It takes about 11 hours on the laptop; the long parts are the embedding models, the training-side search, and the cross-encoders.
- Every stage caches its outputs, so a run can resume after an interruption.
- All entry points fix seeds and use deterministic GPU kernels.
- We verified the package by regenerating the decoding stages from cached intermediate results and comparing SHA-256 checksums with the submitted files: identical.

---

## 16. Results

### 16.1 Leaderboard progression

| Submission | What changed | Public LB |
|---|---|---|
| exp01 | Embedding blocker + 65-feature XGBoost | 0.978934 |
| exp04_fr90 | Stricter threshold (0.9) for the unseen country | ≈0.980 |
| exp05 | Token-alignment and house-number features | 0.982907 |
| exp10 | Consensus features + cross-encoder (US/India) | 0.985008 |
| exp10_diag | *Diagnostic:* France rows emptied | 0.849712 |
| exp11a / exp11b | French noise-word rule | 0.985847 / 0.986449 |
| exp14rcd | *Two untested French rules bundled* | 0.983376 |
| exp14r | Word-role features in the model | 0.986936 |
| exp29gf | Compact candidates + 3 cross-encoders + GBDT stacker with rivals | 0.987891 |
| exp29t80 | *Probe:* France threshold 0.8 | 0.987799 |
| exp35 | + count-matching rules (US siblings, French acronyms, French descriptor siblings) | 0.98891 |
| **exp36** | **+ 4th cross-encoder** | **0.988929** |
| exp38 | *Probe:* self-trained France model | 0.988669 |

### 16.2 Validation progression (US/India, dense world)
0.98576 → 0.98764 → 0.98787 → 0.98808 → 0.98965 → 0.98975 → 0.98985 → 0.98998 → 0.99015 → 0.99032 → **0.99050** → 0.99054

### 16.3 Final picture
- **US/India on test ≈ 0.9905**, the same as validation.
- **France ≈ 0.978.** France improved from about 0.95 at the start through normalization, word roles and count-matching rules.
- **Candidate set: 3.80 per business** (from 18.6), with 98.4% of true pairs kept.
- **Final standing: top 50** overall.

---

## 17. What we learned

1. **Build a validation world you can trust, then prove it.** Our first validation over-estimated the leaderboard by 0.006. The dense world plus one diagnostic submission (France emptied) showed that validation matched the leaderboard for US/India exactly. After that, every in-country decision could be made offline.
2. **Know where the error is before fixing it.** The diagnostic submission showed the entire gap was France, so we stopped polishing the in-country model at the right time.
3. **Use held-out countries to test transfer.** LOCO rejected several attractive ideas (type alignment, quantile mapping, importance weighting) *before* they cost a submission.
4. **Describe roles, not words.** The idea that moved France was to describe *what role a word plays* (does it replace a word at the same address, or get appended at another one?) using statistics from each country's own unlabeled data. Translating words or using bigger language models did not help.
5. **Synthetic data rewards studying the generator.** The best late gains came from count matching: comparing how often each pattern is accepted on test against how often it is true in validation.
6. **Interactions matter in tabular matching.** Gradient-boosted trees beat logistic regression by 0.017 F0.5 on the same features, mostly because "A matters only when B" is everywhere in entity resolution.
7. **Diversity beats size for stacking.** A second small cross-encoder and a different architecture (BERT) helped. A model 2.4× larger did not.
8. **Never bundle untested ideas.** One bundled submission cost −0.0031 and taught us nothing about which part was wrong.
9. **Compute limits shape good engineering.** A 4 GB GPU pushed us to frozen embeddings, our own IVF index, chunked processing and a learned candidate filter. The filter ended up being one of the most useful parts of the system.
10. **Some error is irreducible.** Half of the remaining in-country loss is records with no address and a name shared by many businesses. Knowing that stopped us from chasing the impossible.

---

## 18. What we would try with more time

- **A stage 2 trained on cross-country predictions:** train stage 1 on one country, score the other, and teach stage 2 how to correct the weaker, over-confident probabilities of an unseen country. This was our top open idea and was never built.
- **French-style training data:** rewrite US/India training pairs with French conventions (French noise words, acronyms, sibling organisations), so the model learns them *with labels*.
- **Fully fine-tuned cross-encoders:** train all layers instead of freezing the embeddings, for longer, on more GPU memory.
- **"Record vs its rivals" models:** a model that sees one record together with all its candidate businesses and picks one, instead of judging each pair alone.
- **Grouping records into their own businesses:** link S2 and S3 records to each other, so that distractor businesses can be recognised as complete groups.

---

## Appendix A: Glossary

| Term | Meaning in this report |
|---|---|
| S1, S2/S3 | Source 1 (reference businesses) and Sources 2 and 3 (noisy records to be matched) |
| Singleton | An S1 business with no true record; it scores 1 only if we predict nothing |
| F0.5 | A score that combines precision and recall, with precision counting twice as much |
| Blocking | Finding a small list of candidate pairs without comparing everything with everything |
| IVF index | Nearest-neighbour search that clusters vectors and only searches the closest clusters |
| Cross-fitting / out-of-fold (OOF) | Training on one half of the businesses and predicting the other half, so every prediction is "unseen" |
| Dense world | Our validation setup with 20% of training businesses removed, so distractor density matches test |
| LOCO | Leave-one-country-out: train on the US, test on India, and vice versa; our proxy for France |
| Cross-encoder | A transformer that reads both records together and outputs a match score |
| Stacking | A second model that combines the scores of several models |
| Rival features | A pair's score minus the best score among the record's other candidate businesses |
| Word role | Whether a word *replaces* another word at the same address (noise) or is *appended* at another address (sister company) |
| Count matching | Comparing accepted pairs per pattern on test with true pairs per pattern in validation, to find mistakes without labels |

## Appendix B: Experiment index

| Exp | Idea | Outcome |
|---|---|---|
| exp00 | Normalization v1, native-script dictionary | base of all work |
| exp01 | Embedding blocker + GPU IVF + 65-feature XGBoost | LB 0.978934 |
| exp02 | Stage-2 stacking | +0.002 validation, LB flat |
| exp03 | Dense validation world | more honest validation |
| exp04 | French region/department mapping; unseen threshold 0.9 | kept; LB ≈0.980 |
| exp05 | Token alignment + house-number features | LB 0.982907 |
| exp06 / exp08 | Multilingual cross-encoder (US/India only) | +0.0019 validation |
| exp07 | Normalization v3 + noise-word features, clean deterministic rebuild | +0.0003 |
| exp09 | Cluster-consensus features | +0.0002 |
| exp10 | Consensus + cross-encoder stack | LB 0.985008 |
| exp11 | French word-role rule | LB 0.986449 |
| exp12 | Translating French words | rejected |
| exp13 / exp14r | Word-role features in the model | LB 0.986936 |
| exp14rcd | Two bundled French rules | LB 0.983376 (lesson) |
| exp15 | Learned candidate filter, stage 2 on the compact set | 3.95 per business, F unchanged |
| exp16 | Self-training for France | not adopted |
| exp17 | Cross-encoder coverage fill | +0.00003 |
| exp18 / exp20 | Second cross-encoder + two-threshold decoding | 0.98998 |
| exp19 | French type alignment | LOCO negative, withdrawn |
| exp21 | Importance weighting | rejected |
| exp22 | Stage-2 hyperparameters | flat |
| exp23 / exp24 | bert-base cross-encoder | 0.99015 |
| exp25 | Zero-shot LLM judge (Qwen2.5-1.5B) | AUC 0.52, rejected |
| exp26 / exp30 | multilingual-e5-base cross-encoder | 0.99054 |
| exp28 | GBDT stacker | 0.99032 |
| exp29g | Rival features | 0.99050; LB 0.987891 |
| exp31 | US legal-form sibling demotion (count matching) | in exp35 |
| exp32 | French acronym rule | in exp35 |
| exp33 | French descriptor-sibling demotion | in exp35 |
| exp35 | French acronym candidate channel | LB 0.98891 |
| exp36 | Rules on the 4-cross-encoder stack | **LB 0.988929 (best)** |
| exp37 | Final candidate gate | 3.80 per business, matches unchanged |
| exp38 | Self-trained France, final probe | LB 0.988669 |
