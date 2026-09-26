#!/bin/bash
# resume of run_exp12.sh from the test search (the first attempt ran out of RAM while analyses ran alongside)
set -e
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
T="--test_feat feat_v1/test"
export ER_WORK_DIR="$(pwd)/work_v4"
$PY exp01_block.py search --split test --test_tag fold0
$PY exp01_match.py feats --split test --max_cand 15 --keep_rrank 2
$PY exp07_tokfeat2.py feats --run exp07 $T
$PY exp09_consensus.py feats --run exp09 $T
$PY exp09_consensus.py feats2 --run exp09 $T
$PY exp09_consensus.py predict --run exp09 $T --unseen_thr 0.9
export ER_WORK_DIR="$(pwd)/work_v3"
$PY exp12_roletrans.py combine --run exp12
$PY exp12_roletrans.py combine --run exp12r --rule
echo EXP12_DONE
