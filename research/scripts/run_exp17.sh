#!/bin/bash
# exp17: fill cross-encoder coverage for exp15's uncertain band (saved exp08 fold models, no training), refit the
# stacker (exp10_combine) and apply the exp11 rule (France). Compare dense F0.5 to exp15c (0.98982).
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$REPO/work}"
PY="$REPO/.venv/Scripts/python.exe"
$PY exp17_cefill.py fill --run exp17 --base exp15 --ce_run exp08
$PY exp10_combine.py eval --run exp17c --base exp15 --ce_run exp17
$PY exp10_combine.py predict --run exp17c --base exp15 --ce_run exp17 --unseen_thr 0.9
$PY exp11_rolerule.py predict --run exp17cr --base exp17c
echo EXP17_DONE
