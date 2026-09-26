#!/bin/bash
# exp04: French region/department canonicalization, test side only (models unchanged)
set -e
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
$PY exp04_frnorm.py prep
$PY exp04_frnorm.py emb
$PY exp04_frnorm.py search
$PY exp04_frnorm.py feats
echo EXP04_TESTSIDE_DONE
