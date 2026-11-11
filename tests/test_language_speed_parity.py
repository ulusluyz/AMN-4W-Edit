"""Regression: the optimized Turkish character LM must preserve AMN math and gradients."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import torch
from benchmark_turkish_1000 import dataset, check_parity


def test_language_corpus_not_truncated():
    _, train, held, rows = dataset(96)
    assert train[0].shape == (6, 96)
    assert held[0].shape == (4, 96)
    assert len(rows) == 4


def test_original_vs_precomputed_amns_forward_backward():
    torch.set_num_threads(1)
    tok, train, _, _ = dataset(96)
    for seed in (0, 1, 2):
        result = check_parity(tok, train, seed, 96, 32, 32)
        assert result["max_logits_abs"] < 1e-4
        assert result["max_state_abs"] < 1e-4
        assert result["max_grad_abs"] < 1e-4
