"""End-to-end reproduction of the final submission (LB 0.98891): raw TSVs -> normalization -> blocking -> pair features
-> stage-1 matcher -> candidate filter (= candidate_pairs.tsv) -> stage-2 matcher -> cross-encoders + stacking
(training countries) -> word-role rule (unseen countries) -> decoding -> post-decoding rules -> matching_results.tsv.

Usage (from this src/ folder):
    python run_pipeline.py --data <folder with train/ and test/> --work <scratch folder> --out <output folder>
                           [--from_step <name>] [--to_step <name>] [--list]

Every step caches its results under <work>/data/cache and <work>/runs, so --from_step resumes after an interruption.
All entry points call er_common.set_determinism(0); a rerun on the same hardware/software stack reproduces the files.

Stages (README.md explains each one):
  1  normalization                                   prep_v3.py
  2  blocking: learned two-tower char-n-gram embedding, GPU IVF search, pruning to ~19 candidates per S1
                                                     exp01_block.py, exp01_match.py (test), exp03_dense.py (train)
  3  pair features, 4 row-aligned layers (124 columns)
       base      exp01_match.py / exp03_dense.py (fuzzy name/address scores, numbers, legal forms, counts, blocking)
       tokens    exp07_tokfeat2.py (word alignment, typo vs swapped word, noise-word score, house numbers)
       consensus exp09_consensus.py (does a record's odd word / number recur among the S1's other candidates)
       roles     exp13_rolefeat.py (per-country word roles learned from each split's own data, no labels)
  4  stage-1 matcher: XGBoost, 2-fold cross-fit by S1 on the dense training world (20% of train S1 dropped)
                                                     exp13_rolefeat.py train1 / feats2
  5  candidate filter p1 > 0.01 (US/India) / 0.003 (other countries) -> ~3.95 pairs per S1 = candidate_pairs.tsv;
     stage-2 XGBoost on the kept pairs (124 features + p1 + 17 p1-based competition / consistency features)
                                                     exp15_compact.py
  6  cross-encoders on the uncertain band (training countries): e5-small v1 (trained on the exp07 matcher's band,
     filled for exp15's band), e5-small v2, bert-base                exp07_tokfeat2.py (matcher), exp06_crossenc.py,
                                                                     exp17_cefill.py
  7  stacking (training countries): logistic stack (exp10) and a GBDT stacker with rival features (exp28)
  8  unseen countries: word-role rule for noise words (exp11)
  9  decoding (assignment, thresholds, two-threshold rescue; exp19) and post-decoding rules:
       exp31 US legal-form siblings, exp32 French acronyms, exp33 French descriptor swaps,
       exp35 French acronym candidates (candidate channel after the matcher)
     and the final candidate gate (exp37): pairs with stage-2 p <= 0.01, which no later stage can accept, leave
     candidate_pairs.tsv (3.95 -> 3.80 per S1; matching_results.tsv unchanged)
"""
import os
import sys
import shutil
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
T = ["--test_feat", "feat_v1/test"]

# final configuration = the submitted exp36 (LB 0.988929, see EXPERIMENTS.md): 4 cross-encoders (the 4th, multilingual-e5-base,
# +0.00004 on validation, +0.00002 on the LB over the 3-CE exp35), GBDT stacker with record-side rival features
USE_CE4 = True
T_EMPTY = "0.6"
GB_COMP = "r"

STEPS = [
    # ---- 1. normalization
    ("prep", ["prep_v3.py"]),
    # ---- 2. blocking
    ("emb_feats", ["exp01_block.py", "feats"]),
    ("emb_fold0", ["exp01_block.py", "train", "--tag", "fold0", "--epochs", "3", "--eval_q", "0"]),
    ("emb_fold1", ["exp01_block.py", "train", "--tag", "fold1", "--epochs", "3", "--eval_q", "0"]),
    ("search_train", ["exp01_block.py", "search", "--split", "train"]),
    ("search_test", ["exp01_block.py", "search", "--split", "test", "--test_tag", "fold0"]),
    # ---- 2-3. pruning + base features (test world; dense training world)
    ("feats_test", ["exp01_match.py", "feats", "--split", "test", "--max_cand", "15", "--keep_rrank", "2"]),
    ("feats_train_dense", ["exp03_dense.py", "feats", "--run", "exp03", "--keep_frac", "0.8", "--world_seed", "0"]),
    # ---- 3. token / consensus / role layers
    ("feats_tokens", ["exp07_tokfeat2.py", "feats", "--run", "exp07"] + T),
    ("feats_consensus", ["exp09_consensus.py", "feats", "--run", "exp09"] + T),
    ("feats_roles", ["exp13_rolefeat.py", "feats", "--run", "exp13"] + T),
    # ---- 4. stage-1 matcher -> p1
    ("stage1_train", ["exp13_rolefeat.py", "train1", "--run", "exp13"] + T),
    ("stage1_p1", ["exp13_rolefeat.py", "feats2", "--run", "exp13"] + T),
    # ---- 5. candidate filter (= candidate_pairs.tsv) + stage-2 matcher
    ("stage2_feats", ["exp15_compact.py", "feats2", "--run", "exp15"]),
    ("stage2_train", ["exp15_compact.py", "train2", "--run", "exp15"]),
    ("stage2_predict", ["exp15_compact.py", "predict", "--run", "exp15", "--unseen_thr", "0.9"]),
    # ---- 6. cross-encoders (training countries). CE v1 is trained on the band of the exp07 matcher (token layer only)
    ("exp07_train1", ["exp07_tokfeat2.py", "train1", "--run", "exp07"] + T),
    ("exp07_feats2", ["exp07_tokfeat2.py", "feats2", "--run", "exp07"] + T),
    ("exp07_train2", ["exp07_tokfeat2.py", "train2", "--run", "exp07"] + T),
    ("exp07_predict", ["exp07_tokfeat2.py", "predict", "--run", "exp07", "--unseen_thr", "0.9"] + T),
    ("ce1_pairs", ["exp06_crossenc.py", "pairs", "--run", "exp08", "--base", "exp07"]),
    ("ce1_train", ["exp06_crossenc.py", "train", "--run", "exp08", "--base", "exp07", "--epochs", "2"]),
    ("ce1_fill", ["exp17_cefill.py", "fill", "--run", "exp17", "--base", "exp15", "--ce_run", "exp08"]),
    ("ce2_pairs", ["exp06_crossenc.py", "pairs", "--run", "exp18", "--base", "exp15"]),
    ("ce2_train", ["exp06_crossenc.py", "train", "--run", "exp18", "--base", "exp15", "--epochs", "3"]),
]
BERT = ["--model", "bert-base-uncased", "--model_rev", "86b5e0934494bd15c9632b12f734a8a67f723594"]
STEPS += [
    ("ce3_pairs", ["exp06_crossenc.py", "pairs", "--run", "exp23", "--base", "exp15"] + BERT),
    ("ce3_train", ["exp06_crossenc.py", "train", "--run", "exp23", "--base", "exp15", "--epochs", "2", "--bs", "32"] + BERT),
]
E5B = ["--model", "intfloat/multilingual-e5-base", "--model_rev", "d128750597153bb5987e10b1c3493a34e5a4502a"]
if USE_CE4:
    STEPS += [
        ("ce4_pairs", ["exp06_crossenc.py", "pairs", "--run", "exp26", "--base", "exp15", "--test_countries", "US,India"] + E5B),
        ("ce4_train", ["exp06_crossenc.py", "train", "--run", "exp26", "--base", "exp15", "--epochs", "2", "--bs", "16"] + E5B),
    ]
EXTRA = ",".join(["exp18", "exp23"] + (["exp26"] if USE_CE4 else []))
CE = ["--base", "exp15", "--ce_run", "exp17", "--ce_run2", EXTRA]
CE_RUNS = "exp17," + EXTRA
GB = ["--base", "exp15", "--ce_runs", CE_RUNS, "--fallback", "final_c", "--comp", GB_COMP]
STEPS += [
    # ---- 7. stacking (training countries)
    ("stack_eval", ["exp10_combine.py", "eval", "--run", "final_c"] + CE),
    ("stack_predict", ["exp10_combine.py", "predict", "--run", "final_c", "--unseen_thr", "0.9"] + CE + ["--t_empty", T_EMPTY]),
    ("gbstack_eval", ["exp28_gbstack.py", "eval", "--run", "final_g"] + GB),
    ("gbstack_predict", ["exp28_gbstack.py", "predict", "--run", "final_g"] + GB),
    # ---- 8. unseen countries: word-role rule
    ("role_rule", ["exp11_rolerule.py", "predict", "--run", "final_cr", "--base", "final_g"]),
    # ---- 9. decoding, then the post-decoding rules (each re-decodes with the same thresholds)
    ("final_decode", ["exp19_align.py", "predict", "--run", "final", "--base", "final_cr", "--metrics_run", "final_g",
                      "--t_empty", T_EMPTY, "--types", "none"]),
    ("rule_us_siblings", ["exp31_sibprior.py", "predict", "--run", "final31", "--base", "final", "--metrics_run", "final_g"]),
    ("rule_fr_acronyms", ["exp32_acro.py", "--run", "final32", "--base", "final31", "--metrics_run", "final_g"]),
    ("rule_fr_descswap", ["exp33_descswap.py", "--run", "final33", "--base", "final32", "--metrics_run", "final_g"]),
    ("rule_fr_acro_cands", ["exp35_acrocand.py", "--run", "final35", "--base", "final33", "--metrics_run", "final_g"]),
    # ---- final candidate set: drop pairs that no stage after stage 2 can accept (stage-2 p <= 0.01); matches unchanged
    ("candidate_gate", ["exp37_candgate.py", "--run", "final37", "--base", "final35", "--stage2", "exp15"]),
]
FINAL = "final37"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, help="dataset folder containing train/ and test/")
    ap.add_argument("--work", required=False, help="working folder for caches and runs (~50 GB free recommended)")
    ap.add_argument("--out", default="", help="where to copy matching_results.tsv and candidate_pairs.tsv")
    ap.add_argument("--validator", default="", help="optional path to a validate_submission.py (default: utils/)")
    ap.add_argument("--from_step", default=None, help="start from this step (earlier caches must exist)")
    ap.add_argument("--to_step", default=None, help="stop after this step")
    ap.add_argument("--list", action="store_true", help="print the step names and exit")
    args = ap.parse_args()
    names = [n for n, _ in STEPS]
    if args.list:
        for n, cmd in STEPS:
            print(f"{n:20s} {' '.join(cmd)}")
        return
    if not args.data or not args.work:
        ap.error("--data and --work are required")
    env = dict(os.environ)
    env["ER_DATA_DIR"] = os.path.abspath(args.data)
    env["ER_WORK_DIR"] = os.path.abspath(args.work)
    env["PYTHONIOENCODING"] = "utf-8"
    env["ER_VALIDATOR"] = os.path.abspath(args.validator) if args.validator else os.path.join(HERE, "utils", "validate_submission.py")
    os.makedirs(args.work, exist_ok=True)
    start = names.index(args.from_step) if args.from_step else 0
    stop = names.index(args.to_step) + 1 if args.to_step else len(STEPS)
    for name, cmd in STEPS[start:stop]:
        print(f"==> step {name}: {' '.join(cmd)}", flush=True)
        subprocess.run([sys.executable] + cmd, cwd=HERE, env=env, check=True)
    if args.out and stop == len(STEPS):
        src = os.path.join(args.work, "runs", FINAL, "output")
        os.makedirs(args.out, exist_ok=True)
        for f in ["matching_results.tsv", "candidate_pairs.tsv"]:
            shutil.copy2(os.path.join(src, f), os.path.join(args.out, f))
        print(f"done: outputs copied to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
