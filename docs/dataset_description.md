# Dataset Description

This document is the reference profile of the Business Entity Resolution dataset for
the Amazon ML Challenge 2026. Everything below was measured directly from the files
under `dataset/` (not assumed from the problem statement) so it can be trusted when
designing blocking keys, features, and validation splits. Regenerate the numbers with
the snippets in each section if the dataset changes.

## 1. Files and scale

| File | Rows (incl. header) | Role |
| --- | --- | --- |
| `train/train_source1.tsv` | 2,206,822 | Reference entities (S1), train |
| `train/train_source2.tsv` | 5,034,617 | S2 records, train |
| `train/train_source3.tsv` | 5,285,604 | S3 records, train |
| `train/train_ground_truth.tsv` | 2,206,822 | One row per S1 entity |
| `test/test_source1.tsv` | 1,732,545 | Reference entities (S1), test |
| `test/test_source2.tsv` | 4,887,274 | S2 records, test |
| `test/test_source3.tsv` | 5,082,317 | S3 records, test |

Total ≈ 26.4M rows across all files. This rules out any O(S1 × S2) or O(S1 × S3)
pairwise comparison strategy — at 2.2M × 5M that's ~1.1×10¹³ pairs. Every blocking
design decision in this project should be justified against this scale, not against
what's convenient on a laptop-sized sample.

**Schema** (identical across all three source files):

| Column | Description |
| --- | --- |
| `entity_id` | Unique ID; prefix (`S1-`/`S2-`/`S3-`) is the only source indicator — there is no separate source column |
| `business_name` | Free-text business name |
| `business_address` | Free-text address, no fixed component order or schema |
| `country` | Open-set string label (`US`, `India` in train; `France` added in test) |

`train_ground_truth.tsv`: `source1_entity_id`, `matched_entity_ids` (comma-separated
`S2-`/`S3-` IDs, empty for singletons).

Read all files with `sep="\t"` explicitly — commas appear inside addresses and inside
`matched_entity_ids`, so a naive `read_csv` silently collapses everything into one
column.

## 2. Ground truth: match structure

Measured on `train/train_ground_truth.tsv` (2,206,821 data rows):

| Matches per S1 entity | Count | % |
| --- | --- | --- |
| 0 (singleton) | 123,247 | 5.6% |
| 1 | 119,157 | 5.4% |
| 2 | 375,212 | 17.0% |
| 3 | 530,841 | 24.1% |
| 4 | 484,115 | 21.9% |
| 5 | 321,957 | 14.6% |
| 6 | 164,868 | 7.5% |
| 7 | 63,968 | 2.9% |
| 8 | 18,680 | 0.8% |
| 9 | 4,205 | 0.2% |
| 10 | 534 | <0.1% |
| 11 | 37 | <0.1% |

- Mean matches per non-singleton entity ≈ 3.5; max observed is 11.
- Of 7,638,365 total ground-truth pairs, 3,693,619 (48.4%) point to S2 and 3,944,746
  (51.6%) to S3 — matches are roughly evenly split between the two non-reference
  sources.
- **Only 5.6% of entities are singletons.** Recall is the dominant lever here: for
  94.4% of entities there is real signal to find, and every missed match directly
  costs F₀.₅ on that entity (a missed match caps recall below 1.0 for that row, and
  there is no way to "make up" F₀.₅ elsewhere since it's computed and averaged
  per-entity). Conversely, since singletons are graded 1.0 only on a correctly
  predicted *empty* list, blocking must not manufacture spurious candidates for the
  5.6% of entities that truly have none — precision-heavy final matching depends on
  blocking not flooding singleton entities with noise, but recall depends on not
  starving the other 94.4%.

Regenerate:
```python
import csv
from collections import Counter
counts = Counter()
with open('train/train_ground_truth.tsv') as f:
    r = csv.reader(f, delimiter='\t'); next(r)
    for row in r:
        m = row[1]
        counts[0 if m == '' else len(m.split(','))] += 1
```

## 3. Country field

Distribution (train):

| Country | S1 | S2 | S3 |
| --- | --- | --- | --- |
| US | 1,323,633 | 3,016,817 | 3,170,056 |
| India | 883,188 | 2,017,799 | 2,115,547 |

Distribution (test) — France is new and appears at meaningful volume in **all three**
sources, not just S1:

| Country | S1 | S2 | S3 |
| --- | --- | --- | --- |
| US | 663,106 | 1,871,330 | 1,945,701 |
| India | 809,986 | 2,312,565 | 2,405,000 |
| France | 259,452 | 703,378 | 731,615 |

**Verified fact:** across all 7,638,365 ground-truth pairs in train, **zero** have a
country mismatch between the S1 entity and its matched S2/S3 record. `country` is a
100%-reliable partition key in the training data — grouping candidate generation by
exact `country` string before applying any other blocking logic costs no measured
recall and cuts the search space by roughly half again per partition. Since country is
an open set (France has no training examples), treat this as "group by whatever
string is present," not a hardcoded `{US, India}` branch — the pipeline must not
special-case or filter on specific country values.

Regenerate (join on `entity_id` → `country`, then check every ground-truth pair):
```python
s1 = {row[0]: row[3] for row in read('train_source1.tsv')}
s2 = {row[0]: row[3] for row in read('train_source2.tsv')}
s3 = {row[0]: row[3] for row in read('train_source3.tsv')}
mismatches = sum(
    1 for s1id, matches in ground_truth if matches
    for m in matches.split(',')
    if (s2.get(m) or s3.get(m)) != s1[s1id]
)  # == 0
```

## 4. Business name noise

### 4.1 Script diversity (India only)

Non-ASCII business names by (file, country):

| File | India non-ASCII | India total | US non-ASCII | US total |
| --- | --- | --- | --- | --- |
| S2 | 562,437 (27.9%) | 2,017,799 | 202,171 (6.7%) | 3,016,817 |
| S3 | 391,172 (18.5%) | 2,115,547 | 215,565 (6.8%) | 3,170,056 |

These are two *different* phenomena that need different handling — don't lump them
under one "non-ASCII" flag:

**(a) Genuine multi-script transliteration (India-labeled records only).** Real
Devanagari, Tamil, and Gujarati script business names with no Latin counterpart in the
same field, e.g.:
```
राम मार्केटिंग प्राइवेट लिमिटेड        (Hindi/Devanagari)
குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்        (Tamil)
શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ   (Gujarati)
```
S1 (the reference source) contains **zero** non-ASCII names — it is always Latin
script. So for these ~18-28% of India S2/S3 records, any name-similarity channel that
assumes shared script (token overlap, char n-gram TF-IDF, Levenshtein) gets **zero**
signal against S1, no matter how good the string metric is.

**Important corollary, verified directly:** when the name is in native script, the
**address on the same row is still almost always in Latin script** — street/plot
numbers, building names, and city names stay Latin; only occasionally the state name
is also transliterated (e.g., `महाराष्ट्र` instead of `Maharashtra`, `ગુજરાત` instead
of `Gujarat`). Example (same row):
```
name: ग्लोबल इन्वेस्टमेंट प्रा. लि.
addr: PLOT NO. ##74, SECOND FLOOR ZONE-2, M.P. NAGAR, BHOPAL, मध्य प्रदेश
```
This means **address-based blocking channels are load-bearing, not optional, for the
~20% of India records with non-Latin names** — they are structurally the only
signal-carrying field for those rows unless a transliteration step is added.

**(b) Synthetic diacritic noise (both US and India records).** Ordinary Latin text
with vowels/consonants swapped for accented look-alikes — a noise-injection pattern,
not real orthography:
```
Animal Welfare Nétwork
Cater Ínfrastructure of Henrico County Ínc
DUNKELBERGER ÁND WEEDMAN FISHERIES (LLC)
```
This is cheap to neutralize with Unicode NFKD normalization + combining-mark strip
(`unicodedata.normalize('NFKD', s).encode('ascii','ignore')`), which will not touch
case (a) — Devanagari/Tamil/Gujarati have no Latin decomposition, so NFKD strip is
safe to apply universally as a no-op on those.

### 4.2 Length and legal-suffix tokens (S1, train)

- Name length: min 3, mean 24.0, max 105 characters.
- Legal-suffix / entity-type token frequency in S1 names (out of 2,206,821 rows —
  note many names contain more than one, e.g. "Pvt Ltd"):

| Token | Count | Token | Count |
| --- | --- | --- | --- |
| limited | 522,340 | associates | 42,934 |
| private | 432,292 | group | 40,495 |
| llc | 355,736 | llp | 39,726 |
| inc | 238,309 | partners | 35,049 |
| ltd | 148,599 | corp | 34,788 |
| pvt | 121,462 | co | 27,613 |
| & | 111,583 | pc | 27,151 |
| and | 57,296 | corporation | 14,366 |
| | | company | 13,868 |
| | | holdings | 11,076 |

These are exactly the tokens that will form oversized, low-value blocks if used
directly as a token-blocking key (a "limited" or "llc" block would span hundreds of
thousands of records). They should be stripped into a normalized/canonicalized
suffix feature (or dropped from the blocking key entirely) rather than indexed as
ordinary name tokens — see `docs/` blocking notes for the meta-blocking / block-purge
treatment.

## 5. Address noise

- Address length (S1, train): min 11, mean 52.1, max 256 characters.
- Mean comma-separated components: 3.16 per address — consistent with the README's
  note on component reordering (addresses are not fixed-schema; "city, state, street"
  vs. "street, city, state" both occur).
- Missing addresses (`business_address` empty string): **0% in S1**, but **3.4% in
  S2** (168,967 / 5,034,616) and **3.3% in S3** (175,916 / 5,285,603). Any
  address-only blocking channel needs a name-based fallback for these rows — they
  can't be dropped, since the README confirms even sparse records can still have
  matches.
- ZIP/PIN reliability, checked with a plain regex (`\d{5}(-\d{4})?` for US,
  `\d{6}` for India), consistent across S1/S2/S3:

| Country | Detected ZIP/PIN rate |
| --- | --- |
| US | ~10.1–10.8% |
| India | 0.0% (no isolated 6-digit PIN found by this pattern) |

  Postal codes are **not** a usable primary blocking key — the README's "missing
  components (no PIN code)" warning is confirmed empirically. Treat any ZIP/PIN match
  as a strong bonus signal when present, never as a required key.
- Landmark-based references (e.g. "Near SBI ATM", mentioned in the README) mean some
  India addresses carry no structured street/PIN information at all — for these,
  city/state tokens are the most stable available address signal.

## 6. What this implies for blocking (summary)

1. Partition by exact `country` string first — verified zero-cost, roughly halves
   search space per partition, must stay open-set (no hardcoded country list).
2. Apply Unicode NFKD + diacritic strip universally in preprocessing — neutralizes
   the synthetic accent-noise pattern in ~7% of Latin names for free, and is a no-op
   on genuine non-Latin script.
3. Do not rely on name-only blocking for India records — ~18–28% of India S2/S3 names
   have no Latin-script overlap with S1 at all. Address-token blocking is the
   necessary fallback/primary channel for this slice, since addresses stay Latin even
   when names don't.
4. Strip or down-weight the legal-suffix tokens in section 4.2 before token-blocking
   on names — they're frequent enough to create unusably large blocks.
5. Don't build a blocking key around ZIP/PIN — coverage is too sparse (~10% US, ~0%
   detectable India) to be anything but an opportunistic bonus signal.
6. Budget for empty addresses (~3.4% of S2/S3) and ensure the name-based channels
   alone can still produce candidates for those rows.
7. Recall priority is justified by the ground-truth shape: 94.4% of S1 entities have
   at least one true match, and F₀.₅ is macro-averaged per entity — a single missed
   match on a non-singleton entity is unrecoverable for that row's score, so blocking
   recall should be optimized aggressively (target ~99%+, measured directly against
   `train_ground_truth.tsv`, not assumed) before precision is tuned in the final
   matching stage.

## 7. Regenerating this document

All numbers above came from ad hoc scripts run against `dataset/train/*.tsv` and
`dataset/test/*.tsv` with Python's stdlib `csv` module (no pandas dependency assumed
in this repo yet). If the dataset is updated, re-run the snippets embedded in each
section — none of them take more than a couple of minutes over the full files.
