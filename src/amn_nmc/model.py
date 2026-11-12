"""Autoregressive byte-level Neural Memory Core: logits[B,T,256] and state."""
import torch
from torch import nn
from .block import NMCBlock
from .config import NMCConfig
from .state import NMCState


class AMNNMC(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or NMCConfig()
        c = self.config
        self.embedding = nn.Embedding(256, c.d_model)
        self.blocks = nn.ModuleList([
            NMCBlock(c.d_model, c.memory_dim, c.slots, c.expansion,
                     c.use_fast, c.use_episodic)
            for _ in range(c.layers)
        ])
        self.head = nn.Linear(c.d_model, 256)

    def initial_state(self, batch, *, device=None, dtype=None):
        anchor = self.embedding.weight
        device = anchor.device if device is None else device
        dtype = anchor.dtype if dtype is None else dtype
        return NMCState(tuple(block.initial_state(batch, device=device, dtype=dtype)
                              for block in self.blocks))

    def forward(self, tokens, state=None):
        if tokens.ndim != 2 or tokens.size(1) == 0 or tokens.dtype != torch.long:
            raise ValueError("expected long tokens shaped [batch, nonzero_sequence]")
        if tokens.min() < 0 or tokens.max() >= 256:
            raise ValueError("input contains non-byte token")
        if state is None:
            state = self.initial_state(tokens.shape[0], device=tokens.device)
        if len(state.blocks) != len(self.blocks):
            raise ValueError("state must have one entry per block")
        x = self.embedding(tokens)
        new_blocks = []
        for block, old in zip(self.blocks, state.blocks):
            if old.hidden.shape[0] != tokens.shape[0]:
                raise ValueError("state batch size mismatch")
            x, new = block(x, old)
            new_blocks.append(new)
        return self.head(x), NMCState(tuple(new_blocks))
