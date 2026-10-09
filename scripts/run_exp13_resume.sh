#!/bin/bash
# resume of run_exp13.sh at train2 (first attempt: out of RAM while a queued LOCO job started on the traceback)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
T="--test_feat feat_v1/test"
$PY exp13_rolefeat.py train2 --run exp13 $T
$PY exp13_rolefeat.py predict --run exp13 $T --unseen_thr 0.9
echo EXP13_DONE
