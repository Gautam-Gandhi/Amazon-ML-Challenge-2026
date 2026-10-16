# Analysis scripts

One-off analyses run during the competition (audits, label checks, count matching, leaderboard arithmetic). Results are recorded in `docs/experiment_log.md`. Each script reads the caches of a pipeline run (`ER_WORK_DIR`, default `<repo>/work`) and prints its findings.

| Script | What it checks |
|---|---|
| `fr_scan.py` | French transformation-type scan (label-free on France, labels out-of-country). Each record's best pair (assign) is typed by name transformation x house-number relation. LOCO (exp13 stage 1 trained on one country, scoring |
| `lr_vs_xgb.py` | Why XGBoost and not a conventional logistic regression for p1? Same 124 features, same dense world, same S1 folds and weighting (all positives + 30% negatives x 1/0.3), trained on a 10% sample of the stage-1 training row |
| `lrate_check.py` | exp13 role feature 'lrate' = log10((n+1)/N1). On train, n comes from the dense world (80% of S1 kept) but N1 counts ALL train S1 -> train lrate is ~log10(0.8) = -0.097 lower than the test convention. How much does that m |
| `lrate_examples.py` | Which pairs does the family-lrate inflation push over p=0.5 on test? Same sample/seed as lrate_family.py. Prints examples and whether each pair is accepted in the final exp35 output. |
| `lrate_family.py` | Direction of the lrate effects on test (stage-1 fold-0 model, 1M-row sample per part): A : normalisation fix (train divided by kept S1) == test lrate - log10(1/0.8) A+F: additionally remove the residual +0.25 of family-t |
| `lrate_words.py` | Per-word comparison of the exp13 role tables, train (dense world) vs test, same country: is e_lrate shifted, and is the shift explained by the normalisation (per S1) rather than by the words' behaviour? Also the density- |
| `one_pair.py` | Print the 124 stage-1 features of one real test pair, layer by layer, with the normalized inputs. |
| `s1_ceiling.py` | Ceilings with stage 1 fixed (dense world, labels): perfect matcher on the candidates that survive p1 > thr, vs on all candidates; the current final pipeline (exp29g OOF, assign @0.7); and where its remaining loss is (FP  |
| `s1_trees.py` | Step 4 audit: (1) do test predictions use more trees than the OOF predictions (default inplace_predict vs iteration_range up to best_iteration)? (2) how much do the two fold models disagree on test (averaging effect)? Sa |
| `s1_trees_train.py` | Step 4 audit, with labels (dense world): stage-1 OOF with ALL trees (what test currently gets) vs the stored OOF at the best iteration. Filter recall at 0.003 / 0.01, pairs kept, stage-1 assign F0.5, calibration of the s |
| `s2_audit.py` | Step 5 audit (exp15 stage 2 on the filtered candidates). (1) tree count: OOF with all trees (= what test gets) vs stored best-iteration OOF, with labels (2) test: stored test_pred (all trees) vs best iteration (3) the 18 |
| `acro.py` | Acronym records: R name_core is one short token equal to the initials of a candidate S1's name words. Per country (test): count, p distribution, number relation, whether the acronym-matching S1 is the R's best S1. Train  |
| `acro_join_train.py` | Join-based acronym channel on labeled data (dense world, kept S1): record name = initials of an S1 name, same first house number, street words Jaccard >= 0.5, exactly one such S1 for the record. Precision + funnel (raw/p |
| `acro_loco.py` | Acronym pairs in LOCO predictions (out-of-country model): true rate by p bin and by number relation. |
| `addr_key.py` | How much can an exact-address key recover? (dense world, labels) |
| `block_fr.py` | France blocking check with near-certain pairs found by construction (no labels needed): (a) acronym record: name = initials of an S1's name, same first house number, same street words (b) same name_core + same first hous |
| `block_funnel.py` | In-country blocking funnel with labels (dense world, kept S1): raw search -> pruning -> filter. And: an extra keep rule 'rk_addr <= K' (S1's top-K by address similarity): true pairs recovered vs extra pairs. |
| `brand_addr.py` | Brand-name records at the same house number: true rate by whether the S1 address is unique among S1s (same country), in-country OOF, LOCO both directions, and French counts. |
| `census.py` | Noise-type census: per 1K S1, how many accepted pairs of each (name relation x house-number relation) type, for train truth (dense world, kept S1), US / India / France test accepted (exp29gf decode). |
| `census2.py` | Accepted census: dense OOF (accepted + truth + FP) vs test accepted, per country, by (name type x number x sfx). |
| `completion_loco.py` | LOCO: S1-level completion. At unseen threshold 0.9, for S1s with k accepted (assigned) pairs, true rate of their assigned candidates with p in lower bands. Positive if a band's true rate exceeds the break-even (~0.7 for  |
| `desc_steal.py` | French records whose assigned (best) S1 has a different descriptor than the record, while another candidate S1 has the record's descriptor. Counts, p of best vs alternative, samples. |
| `domain_r.py` | All R records whose raw name is a domain/tag: per country, how many, accepted?, and for their best candidate: number relation and p. Also: do non-accepted French domain records have a same-name S1 with a different number |
| `empty_s1.py` | French S1s with an empty prediction: what are their candidates? (+ same stats for US/India) |
| `idleak.py` | – |
| `idnum.py` | Identical name_core, different first house number: split by number distance and legal form. Train (dense OOF, labels) vs test (by country). |
| `india_neg.py` | Negative excess groups: test vs OOF counts in p bins including the rejected range, and how many competing S1 the R has. |
| `ivf_vs_exact.py` | Does the approximate (IVF) search lose true matches vs an exact search? Sample of US train S1 (fold 0), searched with the fold1 model (as in production), forward channels name@20 / addr@20 / joint@30 only. |
| `look_band.py` | – |
| `look_fr.py` | – |
| `loss_anat.py` | Loss anatomy of the current best dense OOF (exp29g oof_combined, assign + thr 0.7 / t_empty 0.6). |
| `miss_addr.py` | Examples of with-address true pairs missing from the compact candidate set (dense world). |
| `noaddr_copy.py` | Do address-less records share their exact noisy name with another record of the SAME entity (derivation), more than with records of a same-name sibling S1? (train GT, full world) |
| `noaddr_counts.py` | Address-less R whose top-2 candidate S1s share name_core: does the ratio of their accepted with-address record counts predict the owner beyond the model's p? (dense OOF, labels) |
| `noaddr_sig.py` | Is there residual signal to pick the right S1 for address-less R whose name_core is shared by >=2 kept S1s? |
| `noaddr_uniq.py` | Address-less R whose true S1 has a unique name (among kept S1), missing from candidates: what are they? |
| `nocand.py` | What are the test records with no candidate (after the compact filter)? Look up same-name / same-address S1s. |
| `pdist.py` | – |
| `prep_audit.py` | Audit of prep_v3 normalization: measure suspected issues on real data. |
| `prep_impact.py` | Does each normalization gap actually cost accuracy? Dense OOF (labels) for US/India, test p for France. |
| `scan.py` | Systematic label-shift scan: for each fine group x p-bin, test accepted per 1K S1 minus OOF (TP+FP) per 1K S1. Positive excess concentrated where OOF has few TPs => test-only distractors the model accepts. |
| `scan2.py` | – |
| `sfx_scan.py` | Legal-form tokens on records: per 1K S1 in train vs test by country; and French SNC records vs their best S1. |
| `sib.py` | identical name_core, different first house number, legal form on one side only: OOF vs test, split further. |
| `sib2.py` | Full table for the sibling group + France + samples of accepted US test pairs. |
| `sib3.py` | p distributions in the sibling group: OOF TP/FP vs test accepted (US), for group-specific thresholds. |
| `sib_numset.py` | exp31 demoted pairs: split by whether S1 and R number sets intersect (parse artefacts) - OOF TP rate and test counts. |
| `sim31.py` | Monte Carlo estimate of exp31's LB effect: P(true) per demoted pair from count matching (OOF TP / test count per (distance bucket, p bin)); other accepted pairs assumed true; F0.5 per affected S1 before/after. |
| `swapwords.py` | US swap1 samenum same-sfx pairs: which words are swapped, by p band, OOF (with labels) vs test. |
| `typer.py` | Fine pair typer (name-derivation noise types) for assigned pairs (R's best S1). Used to find unseen-country fixes that are safe OUT OF COUNTRY (LOCO labels) and to size them on French test. |
| `typer_run.py` | Size French uncertain pairs by fine type; true rate of the same types out of country (LOCO) by p bin. |
| `us_err.py` | Sample US/India S1s with loss (dense OOF, exp29g) and print all candidates with p and label. |
| `ceiling.py` | Score ceilings on the dense training world (labels), current candidate set (exp29g OOF pairs): A: perfect matcher on the candidates (accept exactly the true candidate pairs) B: perfect everywhere EXCEPT address-less reco |
| `feat_drift.py` | Step 3 audit: train (dense world) vs test distribution of every stage-1 feature on a population that means the same thing in both worlds: exact copies (nc_ratio == 100 and ad_ratio == 100: identical name core and address |
| `margin_check.py` | Train (dense world) vs test: candidate S1s per record (r_ns1) and the runner-up margin (r_margin2), the top stage-1 feature. Compared on (a) all records' best pair and (b) confident matches (train: true pairs; test: p>0. |
| `recall_train.py` | Blocking recall on train (labels), per stage, per country, by address presence; anatomy of the misses; and what an exact-name channel for address-less records would add (recall vs candidate cost). |
| `rkrev_check.py` | Is the stale reverse-channel rank (rk_rev) in the dense training world a problem? 1) how often it differs from a correctly recomputed value (dense world, pruned train candidates) 2) how much the stage-1 / stage-2 models  |
| `sibling_acro.py` | Are there sibling acronyms in France? At each S1 address (same first number + street), count acronym records whose letters equal the S1's initials (A) vs equal them except the LAST letter (B: e.g. 'VA' next to 'Volley Co |
