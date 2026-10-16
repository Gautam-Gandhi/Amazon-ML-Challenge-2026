#!/bin/bash
# exp05: token-alignment + house-number features on dense world (exp03) + French-normalized test (exp04)
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
PY="$REPO/.venv/Scripts/python.exe"
$PY exp05_tokfeat.py feats  --run exp05
$PY exp05_tokfeat.py train1 --run exp05
$PY exp05_tokfeat.py feats2 --run exp05
$PY exp05_tokfeat.py train2 --run exp05
$PY exp05_tokfeat.py predict --run exp05 --unseen_thr 0.9
echo EXP05_DONE
