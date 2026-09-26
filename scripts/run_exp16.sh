#!/bin/bash
# exp16: self-trained stage 2 for unseen countries on the exp15 compact candidates (pseudo-labels from exp15cr),
# unseen threshold fixed a priori to 0.98 (LOCO), exp11 rule re-applied with p_set 0.99.
set -e
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="$(pwd)/work_v3"
PY=.venv/Scripts/python.exe
$PY exp16_selftrain.py fit_predict --run exp16 --base exp15cr --base_pre_rule exp15c
$PY exp11_rolerule.py predict --run exp16r --base exp16 --p_set 0.99 --unseen_thr 0.98
echo EXP16_DONE
