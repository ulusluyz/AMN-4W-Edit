"""Numerical parity tests for the isolated speedup.
Run from repository root: python -m pytest -q tests/test_speed_safe_memory.py
"""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from amn4w_local_test import AMN4W
from safe_memory_speedup import memory_forward_precomputed


def compare(mask):
    torch.manual_seed(123)
    model = AMN4W(50, 64, 2)
    ctx = torch.randint(2, 50, (mask.size(0), mask.size(1)))
    ref = model(ctx, mask)
    ref_loss = ref.square().sum()
    ref_loss.backward()
    ref_grad = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    candidate = memory_forward_precomputed(model, ctx, mask)
    test_loss = candidate.square().sum()
    if test_loss.requires_grad:
        test_loss.backward()
    assert torch.allclose(ref, candidate, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ref_loss, test_loss, atol=1e-6, rtol=1e-5)
    for n, p in model.named_parameters():
        if n in ref_grad:
            actual = p.grad if p.grad is not None else torch.zeros_like(ref_grad[n])
            assert torch.allclose(ref_grad[n], actual, atol=2e-5, rtol=1e-5), n


def test_shared_mask():
    compare(torch.tensor([[0., 1., 1., 0., 1., 1.]]).expand(8, -1).clone())


def test_per_example_masks_do_not_skip_partial_columns():
    mask = torch.tensor([[0., 1., 1., 0., 0., 1.],
                         [0., 1., 0., 0., 1., 1.],
                         [0., 1., 1., 0., 1., 0.]])
    compare(mask)


def test_all_disabled():
    compare(torch.zeros(4, 6))
