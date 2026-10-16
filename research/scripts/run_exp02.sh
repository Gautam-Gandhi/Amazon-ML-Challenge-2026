#!/bin/bash
# exp02: stage-2 stacking on exp01 (train features already built)
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
PY="$REPO/.venv/Scripts/python.exe"
$PY exp02_stack.py train --base exp01 --run exp02 --device cuda --eta 0.1 --rounds 1500
$PY exp02_stack.py feats --split test --base exp01 --run exp02
$PY exp02_stack.py predict --base exp01 --run exp02
echo EXP02_DONE
