#!/bin/bash
# second resume of the v3 world (disk ran full while writing stage-2 test features); same commands as run_world_v3.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
T="--test_feat feat_v1/test"
$PY exp07_tokfeat2.py feats2 --run exp07 $T
$PY exp07_tokfeat2.py train2 --run exp07 $T
$PY exp07_tokfeat2.py predict --run exp07 $T --unseen_thr 0.9
echo WORLD_V3_DONE
