"""Reproducible audit for the current Turkish character-LM data.

The maximum teacher-forced training token accuracy is below 100% because
identical causal prefixes have conflicting next-character targets.

Run: python experiments/accuracy_ceiling_audit.py
Never use held-out test labels to select hyperparameters or early stopping.
"""
from collections import defaultdict, Counter
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from benchmark_turkish_1000 import dataset


def compute_ceiling(x, y):
    groups = defaultdict(Counter)
    for tokens, labels in zip(x.tolist(), y.tolist()):
        for i, target in enumerate(labels):
            if target != 0:
                groups[tuple(tokens[:i + 1])][target] += 1
    n = sum(sum(c.values()) for c in groups.values())
    max_correct = sum(max(c.values()) for c in groups.values())
    nll_floor = sum(-v * math.log(v / sum(c.values()))
                    for c in groups.values() for v in c.values()) / n
    conflicting = []
    for prefix, values in groups.items():
        if len(values) > 1:
            conflicting.append({
                "prefix_tokens": list(prefix),
                "next_character_token_counts": dict(values),
                "unavoidable_errors": sum(values.values()) - max(values.values()),
            })
    return {
        "tokens": n,
        "max_correct": max_correct,
        "max_accuracy": max_correct / n,
        "unavoidable_errors": n - max_correct,
        "minimum_possible_cross_entropy_nats": nll_floor,
        "conflicting_prefixes": conflicting,
    }


def audit_repository_dataset():
    _, train, heldout, eval_rows = dataset(96)
    train_rows = json.loads((ROOT / "data/train.json").read_text(encoding="utf-8"))
    def serialized(row):
        ctx = row["context"]
        if isinstance(ctx, list):
            ctx = " ".join(ctx)
        return f'{ctx} Soru: {row["question"]} Cevap: {row["answer"]}'
    train_set = set(map(serialized, train_rows))
    duplicates = [i for i, row in enumerate(eval_rows) if serialized(row) in train_set]
    return {
        "training_examples": train[0].shape[0],
        "evaluation_examples": heldout[0].shape[0],
        "duplicated_evaluation_indices": duplicates,
        "truly_distinct_evaluation_indices": [
            i for i in range(len(eval_rows)) if i not in duplicates
        ],
        "training_next_character_limit": compute_ceiling(*train),
        "note": "Training and evaluation share tokenizer vocabulary in existing benchmark; "
                "this is not an independent general-language evaluation.",
    }


if __name__ == "__main__":
    print(json.dumps(audit_repository_dataset(), ensure_ascii=False, indent=2))
