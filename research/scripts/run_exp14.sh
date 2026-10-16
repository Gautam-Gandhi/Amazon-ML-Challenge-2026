#!/bin/bash
# exp14 = exp13 (word-role features) + exp08 cross-encoder stacker for seen countries (exp10 recipe), then the exp11
# word-role rule for France (r) and the descriptor-swap rule (rc; only if exp11c wins on the LB).
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$REPO/work}"
PY="$REPO/.venv/Scripts/python.exe"
$PY exp10_combine.py eval --run exp14 --base exp13 --ce_run exp08
$PY exp10_combine.py predict --run exp14 --base exp13 --ce_run exp08 --unseen_thr 0.9
$PY exp11_rolerule.py predict --run exp14r --base exp14
$PY exp11_rolerule.py predict --run exp14rc --base exp14r --metrics_run exp14 --pattern desc
echo EXP14_DONE
