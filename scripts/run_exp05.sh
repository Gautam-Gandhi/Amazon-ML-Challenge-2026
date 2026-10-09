#!/bin/bash
# exp05: token-alignment + house-number features on dense world (exp03) + French-normalized test (exp04)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
PY="$ROOT/.venv/Scripts/python.exe"
$PY exp05_tokfeat.py feats  --run exp05
$PY exp05_tokfeat.py train1 --run exp05
$PY exp05_tokfeat.py feats2 --run exp05
$PY exp05_tokfeat.py train2 --run exp05
$PY exp05_tokfeat.py predict --run exp05 --unseen_thr 0.9
echo EXP05_DONE
