#!/bin/bash
PYTHON_BIN="/home/jules/.pyenv/versions/3.12.13/bin/python3"

echo "Comparing Transformers and AMN4W"
$PYTHON_BIN scripts/train.py --model transformer --seed 42
$PYTHON_BIN scripts/evaluate.py --model transformer --seed 42
