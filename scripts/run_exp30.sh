#!/bin/bash
# exp30: GBDT stacker with rival features (exp29g recipe) over FOUR cross-encoders (+ exp26 multilingual-e5-base),
# logistic 4-CE stack exp27c as fallback. Adopt only if dense F0.5 > exp29g (0.99050).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
G="--ce_runs exp17,exp18,exp23,exp26 --fallback exp27c --comp r"
$PY exp28_gbstack.py eval --run exp30g $G
$PY exp28_gbstack.py predict --run exp30g $G
$PY exp11_rolerule.py predict --run exp30gr --base exp30g
$PY exp19_align.py predict --run exp30gf --base exp30gr --metrics_run exp30g --t_empty 0.6 --types none
echo EXP30_DONE
