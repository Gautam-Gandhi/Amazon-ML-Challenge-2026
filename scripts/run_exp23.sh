#!/bin/bash
# exp23: third cross-encoder, bert-base-uncased (Apache-2.0, 110M, locally cached), frozen embeddings, trained on
# exp15's band (as exp18) for architecture diversity in the stack (training countries only).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
M="--model bert-base-uncased --model_rev 86b5e0934494bd15c9632b12f734a8a67f723594"
$PY exp06_crossenc.py pairs --run exp23 --base exp15 $M
$PY exp06_crossenc.py train --run exp23 --base exp15 --epochs 2 --bs 32 $M
echo EXP23_TRAINED
