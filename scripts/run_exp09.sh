#!/bin/bash
# exp09 (consensus features, v3 world): waits for exp08 (GPU) to finish, then stage 1/2 + predict, two-stage LOCO
# with the 3 feature layers, and exp10 (exp09 p + exp08 cross-encoder for seen countries). Features were computed by
#   exp09_consensus.py feats --run exp09 --test_feat feat_v1/test   ($ROOT/logs/exp09_feats.log)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repository root; modules live in src/
cd "$ROOT/src"
mkdir -p "$ROOT/logs"
export ER_DATA_DIR="${ER_DATA_DIR:-$ROOT/dataset}"
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work}"
PY="$ROOT/.venv/Scripts/python.exe"
T="--test_feat feat_v1/test"
until grep -qE "EXP08_DONE|Traceback" $ROOT/logs/run_exp08.log; do sleep 30; done
$PY exp09_consensus.py train1 --run exp09 $T
$PY exp09_consensus.py feats2 --run exp09 $T
$PY exp09_consensus.py train2 --run exp09 $T
$PY exp09_consensus.py predict --run exp09 $T --unseen_thr 0.9
$PY exp10_combine.py eval --run exp10 --base exp09 --ce_run exp08
$PY exp10_combine.py predict --run exp10 --base exp09 --ce_run exp08 --unseen_thr 0.9
$PY -u tools/loco2.py --feat1 work_v3/data/cache/feat_v3/k80s0 --feat2 work_v3/data/cache/feat_v7/train,work_v3/data/cache/feat_v9/train --rounds 600
echo EXP09_DONE
