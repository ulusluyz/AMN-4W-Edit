"""Regression of the 2026-11-11 Turkish corpus impossibility bound.
The audit must be updated if the dataset is intentionally revised.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from accuracy_ceiling_audit import audit_repository_dataset, compute_ceiling
from benchmark_turkish_1000 import dataset


def test_causal_upper_bound_is_not_one():
    _, train, _, _ = dataset(96)
    value = compute_ceiling(*train)
    assert value["tokens"] == 438
    assert value["max_correct"] == 431
    assert value["unavoidable_errors"] == 7
    assert value["max_accuracy"] == 431 / 438
    assert abs(value["minimum_possible_cross_entropy_nats"] - 0.024544650263398016) < 1e-10


def test_evaluation_contains_training_duplicates():
    result = audit_repository_dataset()
    assert result["training_examples"] == 6
    assert result["evaluation_examples"] == 4
    assert result["duplicated_evaluation_indices"] == [0, 1]
    assert result["truly_distinct_evaluation_indices"] == [2, 3]
