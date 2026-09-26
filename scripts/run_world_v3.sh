#!/bin/bash
# v3 world: full from-scratch run (raw TSVs -> exp07 predictions) in a fresh work dir, determinism on.
# prep_v3 normalization + blocker + candidates + dense-world features + exp07 token features v2 + stages 1/2.
set -e
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="$(pwd)/work_v3"
PY=.venv/Scripts/python.exe
T="--test_feat feat_v1/test"
$PY prep_v3.py
$PY exp01_block.py feats
$PY exp01_block.py train --tag fold0 --epochs 3 --eval_q 0
$PY exp01_block.py train --tag fold1 --epochs 3 --eval_q 0
$PY exp01_block.py search --split train
$PY exp01_block.py search --split test --test_tag fold0
$PY exp01_match.py feats --split test --max_cand 15 --keep_rrank 2
$PY exp03_dense.py feats --run exp03 --keep_frac 0.8 --world_seed 0
$PY exp07_tokfeat2.py feats  --run exp07 $T
$PY exp07_tokfeat2.py train1 --run exp07 $T
$PY exp07_tokfeat2.py feats2 --run exp07 $T
$PY exp07_tokfeat2.py train2 --run exp07 $T
$PY exp07_tokfeat2.py predict --run exp07 $T --unseen_thr 0.9
echo WORLD_V3_DONE
