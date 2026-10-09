#!/bin/bash
# exp02: stage-2 stacking on exp01 (train features already built)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
PY="$ROOT/.venv/Scripts/python.exe"
$PY exp02_stack.py train --base exp01 --run exp02 --device cuda --eta 0.1 --rounds 1500
$PY exp02_stack.py feats --split test --base exp01 --run exp02
$PY exp02_stack.py predict --base exp01 --run exp02
echo EXP02_DONE
