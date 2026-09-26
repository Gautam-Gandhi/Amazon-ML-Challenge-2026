#!/bin/bash
# exp08: exp06-style multilingual cross-encoder on the uncertain band of exp07 (v3 world); CE used for seen
# countries only (exp06 transfer check: CE does not transfer to unseen countries); France keeps exp07 p.
set -e
export PYTHONIOENCODING=utf-8
export ER_WORK_DIR="$(pwd)/work_v3"
PY=.venv/Scripts/python.exe
A="--run exp08 --base exp07"
$PY exp06_crossenc.py pairs $A
$PY exp06_crossenc.py train $A --epochs 2
$PY exp06_crossenc.py stack $A
$PY exp06_crossenc.py predict2 $A --unseen_mode exp05
echo EXP08_DONE
