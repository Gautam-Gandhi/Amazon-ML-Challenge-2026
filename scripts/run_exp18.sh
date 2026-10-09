#!/bin/bash
# exp18: cross-encoder v2 (same frozen-embedding multilingual-e5-small recipe as exp08) trained on the band
# (0.01, 0.99) of the CURRENT stage-2 model exp15 (harder pairs than exp07's band), 3 epochs; OOF train scores +
# fold-average test scores. Stacked together with exp17 (exp08 CE, coverage-filled) by exp18_stack.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
$PY exp06_crossenc.py pairs --run exp18 --base exp15
$PY exp06_crossenc.py train --run exp18 --base exp15 --epochs 3
echo EXP18_TRAINED
