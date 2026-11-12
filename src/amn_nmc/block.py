"""Causal, differentiable recurrent state + delta fast weights + soft episodic memory.

Reads occur before writes at each input byte. No operation classes, relation schemas,
parser, answer table, content fingerprints or hard address allocation are used.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .state import BlockState


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class NMCBlock(nn.Module):
    def __init__(self, d_model, memory_dim, slots, expansion=2,
                 use_fast=True, use_episodic=True):
        super().__init__()
        self.d, self.m, self.slots = d_model, memory_dim, slots
        self.use_fast, self.use_episodic = use_fast, use_episodic
        self.gate = nn.Linear(3 * d_model, d_model)
        self.candidate = nn.Linear(3 * d_model, d_model)
        self.hidden_norm = RMSNorm(d_model)
        self.key = nn.Linear(d_model, memory_dim)
        self.query = nn.Linear(d_model, memory_dim)
        self.value = nn.Linear(d_model, memory_dim)
        self.fast_write = nn.Linear(d_model, 1)
        self.fast_forget = nn.Linear(d_model, 1)
        self.long_write = nn.Linear(d_model, 1)
        self.long_allocate = nn.Linear(d_model + 1, 1)
        self.slot_anchors = nn.Parameter(torch.randn(slots, memory_dim) * 0.1)
        self.slot_bias = nn.Parameter(torch.linspace(0.5, -0.5, slots))
        self.read_merge = nn.Linear(2 * memory_dim, d_model)
        self.fusion_norm = RMSNorm(d_model)
        self.mix_a = nn.Linear(d_model, expansion * d_model)
        self.mix_b = nn.Linear(d_model, expansion * d_model)
        self.mix_out = nn.Linear(expansion * d_model, d_model)
        self.output_norm = RMSNorm(d_model)
        with torch.no_grad():
            self.fast_write.bias.fill_(-2.0)
            self.fast_forget.bias.fill_(-4.0)
            self.long_write.bias.fill_(-2.0)

    def initial_state(self, batch, *, device, dtype):
        d, m, s = self.d, self.m, self.slots
        zeros = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)
        # Trainable distinct slot seeds prevent a symmetric all-zero memory bank.
        anchors = F.normalize(self.slot_anchors.to(device=device, dtype=dtype), dim=-1)
        keys = anchors.unsqueeze(0).expand(batch, -1, -1).clone()
        return BlockState(zeros(batch, d), zeros(batch, m, m), keys,
                          zeros(batch, s, m), zeros(batch, s), zeros(batch, d))

    def step(self, x, state):
        context = torch.cat((x, state.hidden, state.read), dim=-1)
        gate = torch.sigmoid(self.gate(context))
        candidate = torch.tanh(self.candidate(context))
        h = self.hidden_norm(gate * state.hidden + (1.0 - gate) * candidate)
        k = F.normalize(F.elu(self.key(h)) + 1.0, dim=-1, eps=1e-6)
        q = F.normalize(F.elu(self.query(h)) + 1.0, dim=-1, eps=1e-6)
        v = torch.tanh(self.value(h))

        if self.use_fast:
            fast_read = torch.bmm(state.fast, q.unsqueeze(-1)).squeeze(-1)
            prediction = torch.bmm(state.fast, k.unsqueeze(-1)).squeeze(-1)
            eta = torch.sigmoid(self.fast_write(h)).unsqueeze(-1)
            # Forget range [0.99, 1.0] limits indiscriminate decay in this prototype.
            retention = (1.0 - 0.01 * torch.sigmoid(self.fast_forget(h))).unsqueeze(-1)
            correction = (v - prediction).unsqueeze(-1) * k.unsqueeze(-2)
            new_fast = retention * state.fast + eta * correction
        else:
            fast_read = torch.zeros_like(v)
            new_fast = state.fast

        if self.use_episodic:
            similarities = torch.bmm(state.keys, q.unsqueeze(-1)).squeeze(-1)
            address = torch.softmax(similarities / 0.35 + 0.25 * torch.log1p(state.usage), dim=-1)
            long_read = torch.bmm(address.unsqueeze(1), state.values).squeeze(1)
            novelty = (1.0 - address.square().sum(-1, keepdim=True)).clamp(0, 1)
            allocation_gate = torch.sigmoid(self.long_allocate(torch.cat((h, novelty), -1)))
            allocation = torch.softmax(-4.0 * state.usage + self.slot_bias, dim=-1)
            allocation_weight = (1.0 - allocation_gate) * address + allocation_gate * allocation
            write_strength = torch.sigmoid(self.long_write(h)) * allocation_weight
            new_keys = F.normalize(
                (1.0 - write_strength.unsqueeze(-1)) * state.keys
                + write_strength.unsqueeze(-1) * k.unsqueeze(1), dim=-1, eps=1e-6)
            new_values = ((1.0 - write_strength.unsqueeze(-1)) * state.values
                          + write_strength.unsqueeze(-1) * v.unsqueeze(1))
            new_usage = 0.97 * state.usage + 0.03 * (address + allocation_weight)
        else:
            long_read = torch.zeros_like(v)
            new_keys, new_values, new_usage = state.keys, state.values, state.usage

        read = self.read_merge(torch.cat((fast_read, long_read), -1))
        combined = self.fusion_norm(h + read)
        mixed = self.mix_out(F.silu(self.mix_a(combined)) * torch.sigmoid(self.mix_b(combined)))
        output = self.output_norm(combined + mixed)
        return output, BlockState(h, new_fast, new_keys, new_values, new_usage, read)

    def forward(self, x, state):
        outputs = []
        for t in range(x.size(1)):
            y, state = self.step(x[:, t], state)
            outputs.append(y)
        return torch.stack(outputs, dim=1), state
