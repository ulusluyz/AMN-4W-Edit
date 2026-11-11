#!/bin/bash
PYTHON_BIN="/home/jules/.pyenv/versions/3.12.13/bin/python3"
for seed in 42 43 44; do
    echo "Evaluating advanced AMN4W Seed $seed"
    # Overwrite weights so standard evaluate.py uses them
    cp weights/amn4w_adv_seed${seed}.pt weights/amn4w_seed${seed}.pt
    $PYTHON_BIN scripts/evaluate.py --model amn4w --seed $seed --max_len 64 --embed_dim 64 --memory_size 64
done
