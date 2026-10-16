#!/bin/bash
# exp04: French region/department canonicalization, test side only (models unchanged)
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
PY="$REPO/.venv/Scripts/python.exe"
$PY exp04_frnorm.py prep
$PY exp04_frnorm.py emb
$PY exp04_frnorm.py search
$PY exp04_frnorm.py feats
echo EXP04_TESTSIDE_DONE
