#!/bin/bash
# exp06: multilingual cross-encoder (multilingual-e5-small, frozen embeddings) on exp05's uncertain band
set -e
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
$PY exp06_crossenc.py pairs --run exp06
$PY exp06_crossenc.py train --run exp06 --epochs 2
$PY exp06_crossenc.py stack --run exp06
$PY exp06_crossenc.py predict --run exp06 --unseen_thr 0.9
$PY exp06_crossenc.py train --run exp06 --epochs 2 --loco
$PY exp06_crossenc.py loco_eval --run exp06
$PY exp06_crossenc.py predict2 --run exp06 --unseen_mode exp05   # country-aware submission (added: run manually on 09-26)
$PY exp06_crossenc.py predict2 --run exp06 --unseen_mode ood
echo EXP06_DONE
