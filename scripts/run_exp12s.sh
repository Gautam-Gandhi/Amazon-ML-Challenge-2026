#!/bin/bash
# exp12s: role translation (pure noise words only, --mode noise) of unseen-country name words (France) -> recompute the TEST side in work_v5 with the
# unchanged exp09 models, then combine (France from work_v5, US/India from exp10). World built by:
#   ER_WORK_DIR=work_v3 python exp12_roletrans.py prep        (translated test prep + hard-linked train side)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
PY="$ROOT/.venv/Scripts/python.exe"
T="--test_feat feat_v1/test"
export ER_WORK_DIR="$ROOT/work_v5"
# the blocker feature stage also hashes the train split; only test is searched -> placeholders skip train
for f in train_s1_name train_s1_addr train_r_name train_r_addr; do touch "$ER_WORK_DIR/data/cache/emb_v1/$f.npz"; done
$PY exp01_block.py feats
$PY exp01_block.py search --split test --test_tag fold0
$PY exp01_match.py feats --split test --max_cand 15 --keep_rrank 2
$PY exp07_tokfeat2.py feats --run exp07 $T
$PY exp09_consensus.py feats --run exp09 $T
$PY exp09_consensus.py feats2 --run exp09 $T
$PY exp09_consensus.py predict --run exp09 $T --unseen_thr 0.9
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
$PY exp12_roletrans.py combine --run exp12s --out_world "$(pwd)/work_v5"
$PY exp12_roletrans.py combine --run exp12sr --rule --out_world "$(pwd)/work_v5"
echo EXP12_DONE
