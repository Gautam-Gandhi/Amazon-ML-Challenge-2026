#!/bin/bash
# exp04: French region/department canonicalization, test side only (models unchanged)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
PY="$ROOT/.venv/Scripts/python.exe"
$PY exp04_frnorm.py prep
$PY exp04_frnorm.py emb
$PY exp04_frnorm.py search
$PY exp04_frnorm.py feats
echo EXP04_TESTSIDE_DONE
