#!/bin/bash
# exp20: exp15 matcher + two cross-encoders (exp17 = exp08 coverage-filled, exp18 = CE v2 on exp15's band) stacked,
# two-threshold decoding for training countries (t_empty 0.6), exp11 rule + exp19 alignment for unseen countries.
# Waits for the CE v2 training (scripts/run_exp18.sh).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
until grep -qE "EXP18_TRAINED|Traceback" $ROOT/logs/run_exp18.log; do sleep 30; done
grep -q EXP18_TRAINED $ROOT/logs/run_exp18.log
CE="--base exp15 --ce_run exp17 --ce_run2 exp18"
$PY exp10_combine.py eval --run exp20c $CE
$PY exp10_combine.py predict --run exp20c $CE --unseen_thr 0.9 --t_empty 0.6
$PY exp11_rolerule.py predict --run exp20cr --base exp20c
$PY exp19_align.py eval --run exp20 --base_oof exp20c
$PY exp19_align.py predict --run exp20 --base exp20cr --metrics_run exp20c
echo EXP20_DONE
