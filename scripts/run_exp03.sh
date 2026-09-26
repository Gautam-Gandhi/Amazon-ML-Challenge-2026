#!/bin/bash
# exp03: dense-world (S1 dropout) training/validation + stage-2 stacking; uses cached cand_v1 + exp01 code
set -e
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
A="--run exp03 --keep_frac 0.8 --world_seed 0"
$PY exp03_dense.py feats  $A
$PY exp03_dense.py train1 $A
$PY exp03_dense.py feats2 $A
$PY exp03_dense.py train2 $A
$PY exp03_dense.py predict $A
echo EXP03_DONE
