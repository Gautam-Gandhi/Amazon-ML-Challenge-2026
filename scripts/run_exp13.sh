#!/bin/bash
# exp13: word-role feature layer (feat_v13). LOCO first (France proxy; baseline with feat_v7+feat_v9: US->IN 0.96298,
# IN->US 0.97852), then the in-country stages and test prediction. Features: exp13_rolefeat.py feats ($ROOT/logs/exp13_feats.log)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
T="--test_feat feat_v1/test"
$PY -u tools/loco.py --feat_dir work_v3/data/cache/feat_v3/k80s0 \
  --extra_dir work_v3/data/cache/feat_v7/train,work_v3/data/cache/feat_v9/train,work_v3/data/cache/feat_v13/train \
  --configs base --save_pred work_v3/runs/loco_pred13
$PY exp13_rolefeat.py train1 --run exp13 $T
$PY exp13_rolefeat.py feats2 --run exp13 $T
$PY exp13_rolefeat.py train2 --run exp13 $T
$PY exp13_rolefeat.py predict --run exp13 $T --unseen_thr 0.9
echo EXP13_DONE
