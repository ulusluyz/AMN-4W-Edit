#!/bin/bash
PYTHON_BIN="/home/jules/.pyenv/versions/3.12.13/bin/python3"
for seed in 42 43 44; do
    echo "Running Advanced Training Seed $seed"
    $PYTHON_BIN scripts/train_advanced.py --model amn4w --seed $seed
done
./scripts/evaluate_adv.sh
