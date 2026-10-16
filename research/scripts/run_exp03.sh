#!/bin/bash
# exp03: dense-world (S1 dropout) training/validation + stage-2 stacking; uses cached cand_v1 + exp01 code
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
PY="$REPO/.venv/Scripts/python.exe"
A="--run exp03 --keep_frac 0.8 --world_seed 0"
$PY exp03_dense.py feats  $A
$PY exp03_dense.py train1 $A
$PY exp03_dense.py feats2 $A
$PY exp03_dense.py train2 $A
$PY exp03_dense.py predict $A
echo EXP03_DONE
