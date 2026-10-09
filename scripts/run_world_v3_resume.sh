#!/bin/bash
# resume of scripts/run_world_v3.sh from the train search (the first attempt died with a RAM OOM while another job ran;
# prep_v3, embedding features and fold models were complete). Same commands/arguments as run_world_v3.sh.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
T="--test_feat feat_v1/test"
$PY exp01_block.py search --split train
$PY exp01_block.py search --split test --test_tag fold0
$PY exp01_match.py feats --split test --max_cand 15 --keep_rrank 2
$PY exp03_dense.py feats --run exp03 --keep_frac 0.8 --world_seed 0
$PY exp07_tokfeat2.py feats  --run exp07 $T
$PY exp07_tokfeat2.py train1 --run exp07 $T
$PY exp07_tokfeat2.py feats2 --run exp07 $T
$PY exp07_tokfeat2.py train2 --run exp07 $T
$PY exp07_tokfeat2.py predict --run exp07 $T --unseen_thr 0.9
echo WORLD_V3_DONE
