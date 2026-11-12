"""Persist the finite recurrent inference state between processes.

Persistent state is not evidence of unlimited memory, independent reasoning
or learned online weight updates. Session snapshots may encode private context.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import torch

from .state import BlockState, NMCState

FIELDS = ("hidden", "fast", "keys", "values", "usage", "read")


def model_sha256(model):
    """Stable hash of tensor names, shape, dtype and raw bytes, without NumPy."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(bytes(tensor.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


def save_session(path, model, state, *, tokens_seen):
    if tokens_seen < 0 or state is None or len(state.blocks) != len(model.blocks):
        raise ValueError("Invalid session state")
    payload = {
        "format": "AMN-NMC-RUNTIME-STATE-v1",
        "model_sha256": model_sha256(model),
        "tokens_seen": int(tokens_seen),
        "blocks": [
            {key: getattr(block, key).detach().cpu().clone() for key in FIELDS}
            for block in state.blocks
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".amn_session_", suffix=".tmp",
                                     dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload["model_sha256"]


def load_session(path, model):
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    if data.get("format") != "AMN-NMC-RUNTIME-STATE-v1":
        raise ValueError("Unknown session-state format")
    if data.get("model_sha256") != model_sha256(model):
        raise ValueError("Session checkpoint does not match neural weights")
    expected = model.initial_state(1)
    if len(data.get("blocks", [])) != len(expected.blocks):
        raise ValueError("Wrong number of session blocks")
    blocks = []
    for saved, template in zip(data["blocks"], expected.blocks):
        tensors = []
        for key in FIELDS:
            value = saved.get(key)
            reference = getattr(template, key)
            if (not isinstance(value, torch.Tensor)
                    or value.shape != reference.shape
                    or value.dtype != reference.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError("Invalid session tensor: " + key)
            tensors.append(value)
        blocks.append(BlockState(*tensors))
    count = int(data["tokens_seen"])
    if count < 0:
        raise ValueError("Negative token count")
    return NMCState(tuple(blocks)), count
