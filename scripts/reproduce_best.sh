#!/bin/bash
# Reproduce the CURRENT BEST submission: runs/exp06_exp05/output/matching_results.tsv
# (exp06 cross-encoder re-scoring for seen countries on top of exp05; France keeps exp05 p with thr 0.9)
#
#   scripts/reproduce_best.sh predict   -> regenerate the TSV from the SAVED models/caches (byte-identical, ~10 min)
#   scripts/reproduce_best.sh full      -> retrain everything from the raw TSVs (~5-6 h on RTX 3050 / 16 GB RAM)
#
# Exact commands/arguments below are the ones that produced the submission (see EXPERIMENTS.md for each step).
# Since 09-26 every entry point calls er_common.set_determinism(0) (seeds, deterministic torch/cuBLAS, eager
# attention for the cross-encoder, pinned HF model revision): the `full` path is bit-reproducible run-to-run on the
# same hardware/software stack (tools/determinism_check.py). The artifacts currently in runs/ were trained before that
# fix, so the FINAL submission is produced by one clean `full` run (fresh ER_WORK_DIR) and packaged with this code.
# The `predict` path reproduces the current TSV exactly from runs/MANIFEST_best.sha256 artifacts.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
PY="$ROOT/.venv/Scripts/python.exe"
MODE=${1:-predict}

if [ "$MODE" = "full" ]; then
  # exp00: preprocessing (+ native-script maps mined from train GT)
  $PY prep_v1.py
  # exp01: hashed n-gram features, fold embedding models, IVF candidate search (train OOF + test with fold0 model)
  $PY exp01_block.py feats
  $PY exp01_block.py train --tag fold0 --epochs 3
  $PY exp01_block.py train --tag fold1 --epochs 3 --eval_q 0
  $PY exp01_block.py search --split train
  $PY exp01_block.py search --split test --test_tag fold0
  # exp03: dense (S1-dropout 20%, seed 0) train world features
  $PY exp03_dense.py feats --run exp03 --keep_frac 0.8 --world_seed 0
  # exp04: French department->region canonicalization, test side (prep_v2 -> emb_v2 -> cand_v2 -> feat_v4)
  $PY exp04_frnorm.py prep
  $PY exp04_frnorm.py emb
  $PY exp04_frnorm.py search
  $PY exp04_frnorm.py feats
  # exp05: token-alignment + house-number features, stage 1 + stage 2
  $PY exp05_tokfeat.py feats  --run exp05
  $PY exp05_tokfeat.py train1 --run exp05
  $PY exp05_tokfeat.py feats2 --run exp05
  $PY exp05_tokfeat.py train2 --run exp05
  $PY exp05_tokfeat.py predict --run exp05 --unseen_thr 0.9
  # exp06: cross-encoder on the uncertain band (cross-fit), stacker, LOCO models (for the OOD analysis)
  $PY exp06_crossenc.py pairs --run exp06
  $PY exp06_crossenc.py train --run exp06 --epochs 2
  $PY exp06_crossenc.py stack --run exp06
  $PY exp06_crossenc.py predict2 --run exp06 --unseen_mode exp05
elif [ "$MODE" = "predict" ]; then
  # needs caches: prep_v2/test, feat_v4/test, feat_v5/test, feat_v5s2/test, runs/exp06/test_band.parquet
  # and the saved models: runs/exp05/stage{1,2}_fold{0,1}.json, runs/exp06/ce_fold{0,1}/, runs/exp06/stacker.pkl
  $PY exp05_tokfeat.py predict --run exp05 --unseen_thr 0.9
  $PY exp06_crossenc.py score_test --run exp06
  $PY exp06_crossenc.py predict2 --run exp06 --unseen_mode exp05
else
  echo "usage: $0 [predict|full]"; exit 1
fi
sha256sum runs/exp06_exp05/output/matching_results.tsv
echo REPRODUCE_DONE
