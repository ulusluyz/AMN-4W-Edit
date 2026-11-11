#!/bin/bash
PYTHON_BIN="/home/jules/.pyenv/versions/3.12.13/bin/python3"

# Just testing AMN4W with seed 42 to verify we get non-zero performance quickly.
$PYTHON_BIN scripts/train.py --model amn4w --seed 42
$PYTHON_BIN scripts/evaluate.py --model amn4w --seed 42
