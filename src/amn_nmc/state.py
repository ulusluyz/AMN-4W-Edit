"""Differentiable per-sequence runtime states (not learned fact tables)."""
from dataclasses import dataclass
import torch


@dataclass
class BlockState:
    hidden: torch.Tensor       # [batch, d_model]
    fast: torch.Tensor         # [batch, memory_dim, memory_dim]
    keys: torch.Tensor         # [batch, slots, memory_dim]
    values: torch.Tensor       # [batch, slots, memory_dim]
    usage: torch.Tensor        # [batch, slots]
    read: torch.Tensor         # [batch, d_model]

    def detach(self):
        return BlockState(*(value.detach() for value in (
            self.hidden, self.fast, self.keys, self.values, self.usage, self.read)))


@dataclass
class NMCState:
    blocks: tuple[BlockState, ...]

    def detach(self):
        return NMCState(tuple(block.detach() for block in self.blocks))
