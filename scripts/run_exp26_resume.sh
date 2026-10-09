#!/bin/bash
# exp26: fourth cross-encoder, intfloat/multilingual-e5-base (MIT, 278M; local copy of revision
# d128750597153bb5987e10b1c3493a34e5a4502a), frozen embeddings, on exp15's band. The test band is restricted to
# training countries (CEs are only used there) to save ~20% of the slow scoring. Then a 4-CE stack (exp27).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
M="--model models/multilingual-e5-base"
$PY exp06_crossenc.py train --run exp26 --base exp15 --epochs 2 --bs 16 $M
echo EXP26_TRAINED
CE="--base exp15 --ce_run exp17 --ce_run2 exp18,exp23,exp26"
$PY exp10_combine.py eval --run exp27c $CE
$PY exp10_combine.py predict --run exp27c $CE --unseen_thr 0.9 --t_empty 0.6
$PY exp11_rolerule.py predict --run exp27cr --base exp27c
$PY exp19_align.py predict --run exp27f --base exp27cr --metrics_run exp27c --t_empty 0.6 --types none
echo EXP27_DONE
