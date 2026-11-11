"""Guardrails for the *independent* 2026-11-11 CPU study.

These tests do NOT validate AMN-4W v0.4 trained language behavior.
Run: python -m pytest -q tests/test_independent_memory_20261111.py
"""
import importlib.util
from pathlib import Path

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "independent_memory_20261111.py"
SPEC = importlib.util.spec_from_file_location("independent_memory_20261111", SOURCE)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_numeric_dual_erase_and_write():
    r = study.numeric(seed=42, trials=32, d=32, N=5)
    assert r["erase_ok_fraction"] == 1.0
    assert r["duplicate_erase_ok_fraction"] == 1.0
    assert r["target_remaining_max"] < 1e-5
    assert r["neighbor_change_max"] < 1e-5
    assert r["duplicate_neighbor_change_max"] < 1e-5
    for row in r["writes"].values():
        assert row["write_ok_fraction"] == 1.0
        assert row["dual_neighbor_cos"] > 0.9999
        assert row["dual_max_neighbor_change"] < 1e-4


def test_low_rank_new_key_is_rejected_not_merged():
    torch.manual_seed(71)
    b, d = 2, 16
    K = study.unit(torch.randn(b, 3, d))
    S = torch.randn(b, d, d)
    k = K[:, 1].clone()
    v = study.unit(torch.randn(b, d))
    result, rho, ok = study.dual_write(S, K, k, v, torch.ones(b, 1))
    assert not ok.any()
    assert torch.equal(result, S)
    assert (rho < 1e-6).all()


def test_dual_write_autograd_finite_for_separated_keys():
    torch.manual_seed(9)
    b, d = 2, 16
    K = study.unit(torch.randn(b, 3, d)).requires_grad_()
    k = study.unit(torch.randn(b, d)).requires_grad_()
    value = study.unit(torch.randn(b, d)).requires_grad_()
    S = torch.zeros(b, d, d)
    updated, rho, ok = study.dual_write(S, K, k, value, torch.ones(b, 1))
    assert ok.all()
    loss = updated.square().sum()
    loss.backward()
    for x in (K, k, value):
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("seed", [0, 1])
def test_tiny_training_runs_without_nonfinite_metrics(seed):
    # Smoke test only; deliberately no accuracy claim at 5 steps.
    result = study.train(seed=seed, steps=5, V=20, M=2)
    for key in ("final", "r2_C", "leak"):
        assert torch.isfinite(torch.tensor(result[key]))
