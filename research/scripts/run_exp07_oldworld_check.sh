#!/bin/bash
# exp07 feature layer validated in the OLD (prep_v1) world: isolates the value of noise-word + abbreviation features
# (compare stage-1 dense OOF with exp05 stage 1 = 0.98662, same world, same candidates)
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
PY="$REPO/.venv/Scripts/python.exe"
$PY exp07_tokfeat2.py feats  --run exp07_old
$PY exp07_tokfeat2.py train1 --run exp07_old
echo EXP07_OLD_DONE
