"""Experimental semantics-preserving optimization for the synthetic AMN-4W proxy.

This module intentionally does not modify the production model. Run tests:
    python -m pytest -q tests/test_speed_safe_memory.py

Performance applies to masks that have entire time columns disabled for the
whole batch. Never skip an index with any sample enabled.
"""
import torch
from torch.nn import functional as F


def read(state, key):
    return torch.bmm(state, key.unsqueeze(-1)).squeeze(-1)


def memory_forward_precomputed(module, tokens, write_mask, *, skip_all_zero=True):
    """Same recurrence as experiments/amn4w_local_test.py::AMN4W.forward.

    All projection layers come from the provided, unchanged model. Both forward
    and backward remain differentiable. Does not use dual-update operations.
    """
    batch, length = tokens.shape
    embedded = module.embedding(tokens)
    state = embedded.new_zeros(batch, module.d, module.d)
    if length < 2:
        return state
    previous = embedded[:, :-1]
    current = embedded[:, 1:]
    indices = list(range(length - 1))
    if skip_all_zero:
        # Valid for arbitrary per-example masks: skip a timestep only when ALL
        # examples in this batch have a zero mask at that timestep.
        indices = [i for i in indices if bool(torch.all(write_mask[:, i + 1] == 0))]
        active = [i for i in range(length - 1) if i not in indices]
    else:
        active = indices
    if not active:
        return state
    keys = F.normalize(module.fc_k(previous[:, active]), dim=-1)
    values = F.normalize(module.fc_v(current[:, active]), dim=-1)
    etas = module.fc_eta(current[:, active]).sigmoid() * write_mask[:, [i + 1 for i in active], None]
    for index in range(len(active)):
        key = keys[:, index]
        value = values[:, index]
        eta = etas[:, index]
        state = state + torch.bmm((eta * (value - read(state, key))).unsqueeze(-1), key.unsqueeze(1))
    return state
