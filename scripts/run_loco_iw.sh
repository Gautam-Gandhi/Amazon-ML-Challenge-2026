#!/bin/bash
# single-stage LOCO in the v3 world (feat_v3 + exp07 tokens + exp09 consensus): base vs covariate-shift importance
# weighting (iw10 / iw5 / iw20). Waits for the exp09 chain (GPU/RAM) to finish.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
until grep -qE "EXP09_DONE|Traceback" $ROOT/logs/run_exp09.log; do sleep 30; done
.venv/Scripts/python.exe -u tools/loco.py --feat_dir work_v3/data/cache/feat_v3/k80s0 \
  --extra_dir work_v3/data/cache/feat_v7/train,work_v3/data/cache/feat_v9/train --configs "base;iw10;iw5;iw20"
echo LOCO_IW_DONE
