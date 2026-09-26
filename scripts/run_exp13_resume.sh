#!/bin/bash
# resume of run_exp13.sh at train2 (first attempt: out of RAM while a queued LOCO job started on the traceback)
set -e
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="$(pwd)/work_v3"
PY=.venv/Scripts/python.exe
T="--test_feat feat_v1/test"
$PY exp13_rolefeat.py train2 --run exp13 $T
$PY exp13_rolefeat.py predict --run exp13 $T --unseen_thr 0.9
echo EXP13_DONE
