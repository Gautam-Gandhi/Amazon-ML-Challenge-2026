#!/bin/bash
# exp24: exp20 recipe with THREE cross-encoders stacked (exp17 = e5-small v1 filled, exp18 = e5-small v2,
# exp23 = bert-base-uncased), two-threshold decoding, exp11 rule, final decode. Waits for exp23 training.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
until grep -qE "EXP23_TRAINED|Traceback" $ROOT/logs/run_exp23.log; do sleep 30; done
grep -q EXP23_TRAINED $ROOT/logs/run_exp23.log
CE="--base exp15 --ce_run exp17 --ce_run2 exp18,exp23"
$PY exp10_combine.py eval --run exp24c $CE
$PY exp10_combine.py predict --run exp24c $CE --unseen_thr 0.9 --t_empty 0.6
$PY exp11_rolerule.py predict --run exp24cr --base exp24c
$PY exp19_align.py predict --run exp24f --base exp24cr --metrics_run exp24c --t_empty 0.6 --types none
echo EXP24_DONE
