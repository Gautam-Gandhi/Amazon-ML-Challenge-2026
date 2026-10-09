#!/bin/bash
# exp15: compact candidate set (ANN -> stage-1 filter p1 > 0.01 seen / 0.003 unseen) + stage 2 retrained on it,
# then the exp14r recipe on top: exp08 CE stack (seen countries) + exp11 rule (unseen countries).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
$PY exp15_compact.py feats2 --run exp15
$PY exp15_compact.py train2 --run exp15
$PY exp15_compact.py predict --run exp15 --unseen_thr 0.9
$PY exp10_combine.py eval --run exp15c --base exp15 --ce_run exp08
$PY exp10_combine.py predict --run exp15c --base exp15 --ce_run exp08 --unseen_thr 0.9
$PY exp11_rolerule.py predict --run exp15cr --base exp15c
echo EXP15_DONE
