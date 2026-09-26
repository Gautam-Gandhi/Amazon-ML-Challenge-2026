#!/bin/bash
# second resume of the v3 world (disk ran full while writing stage-2 test features); same commands as run_world_v3.sh
set -e
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="$(pwd)/work_v3"
PY=.venv/Scripts/python.exe
T="--test_feat feat_v1/test"
$PY exp07_tokfeat2.py feats2 --run exp07 $T
$PY exp07_tokfeat2.py train2 --run exp07 $T
$PY exp07_tokfeat2.py predict --run exp07 $T --unseen_thr 0.9
echo WORLD_V3_DONE
