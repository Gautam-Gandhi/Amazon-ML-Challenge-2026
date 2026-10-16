#!/bin/bash
# exp22: stage-2 XGBoost hyperparameters on the filtered set (never tuned: depth 8 / eta 0.1 since exp01)
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$REPO/work}"
PY="$REPO/.venv/Scripts/python.exe"
$PY exp15_compact.py train2 --run exp22a --depth 10 --eta 0.05 --rounds 4000
$PY exp15_compact.py train2 --run exp22b --depth 6 --eta 0.05 --rounds 4000
echo EXP22_DONE
