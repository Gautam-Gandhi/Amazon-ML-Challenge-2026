#!/bin/bash
# exp07 feature layer validated in the OLD (prep_v1) world: isolates the value of noise-word + abbreviation features
# (compare stage-1 dense OOF with exp05 stage 1 = 0.98662, same world, same candidates)
set -e
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
$PY exp07_tokfeat2.py feats  --run exp07_old
$PY exp07_tokfeat2.py train1 --run exp07_old
echo EXP07_OLD_DONE
