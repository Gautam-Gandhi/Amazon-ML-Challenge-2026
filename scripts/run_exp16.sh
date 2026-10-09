#!/bin/bash
# exp16: self-trained stage 2 for unseen countries on the exp15 compact candidates (pseudo-labels from exp15cr),
# unseen threshold fixed a priori to 0.98 (LOCO), exp11 rule re-applied with p_set 0.99.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
$PY exp16_selftrain.py fit_predict --run exp16 --base exp15cr --base_pre_rule exp15c
$PY exp11_rolerule.py predict --run exp16r --base exp16 --p_set 0.99 --unseen_thr 0.98
echo EXP16_DONE
