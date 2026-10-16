#!/bin/bash
# exp08: exp06-style multilingual cross-encoder on the uncertain band of exp07 (v3 world); CE used for seen
# countries only (exp06 transfer check: CE does not transfer to unseen countries); France keeps exp07 p.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$REPO/work}"
PY="$REPO/.venv/Scripts/python.exe"
A="--run exp08 --base exp07"
$PY exp06_crossenc.py pairs $A
$PY exp06_crossenc.py train $A --epochs 2
$PY exp06_crossenc.py stack $A
$PY exp06_crossenc.py predict2 $A --unseen_mode exp05
echo EXP08_DONE
