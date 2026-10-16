#!/bin/bash
# single-stage LOCO in the v3 world: per-country rank normalization (and rank + importance weights); after loco_iw
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$REPO/research"                        # research code lives in research/src
cd "$ROOT/src"
mkdir -p "$REPO/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$REPO/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$REPO/work}"
until grep -qE "LOCO_IW_DONE|Traceback" $REPO/logs/loco_iw.log; do sleep 30; done
.venv/Scripts/python.exe -u tools/loco.py --feat_dir work_v3/data/cache/feat_v3/k80s0 \
  --extra_dir work_v3/data/cache/feat_v7/train,work_v3/data/cache/feat_v9/train --configs "rank"
echo LOCO_RANK_DONE
