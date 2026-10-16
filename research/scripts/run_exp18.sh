#!/bin/bash
# exp18: cross-encoder v2 (same frozen-embedding multilingual-e5-small recipe as exp08) trained on the band
# (0.01, 0.99) of the CURRENT stage-2 model exp15 (harder pairs than exp07's band), 3 epochs; OOF train scores +
# fold-average test scores. Stacked together with exp17 (exp08 CE, coverage-filled) by exp18_stack.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$REPO/work}"
PY="$REPO/.venv/Scripts/python.exe"
$PY exp06_crossenc.py pairs --run exp18 --base exp15
$PY exp06_crossenc.py train --run exp18 --base exp15 --epochs 3
echo EXP18_TRAINED
