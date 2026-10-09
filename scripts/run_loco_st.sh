#!/bin/bash
# LOCO self-training test (France proxy): base vs st (1 round of pseudo-labels on the target country), same reduced
# samples (40% source S1 train, 25% target S1 eval) to fit 16 GB RAM with the 4 feature layers. Waits for exp13.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
: # waited for exp13 (finished 19:21)
.venv/Scripts/python.exe -u tools/loco.py --feat_dir work_v3/data/cache/feat_v3/k80s0 \
  --extra_dir work_v3/data/cache/feat_v7/train,work_v3/data/cache/feat_v9/train,work_v3/data/cache/feat_v13/train \
  --configs "base;st" --tr_frac 0.4 --ev_frac 0.25
echo LOCO_ST_DONE
