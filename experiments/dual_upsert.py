"""Dual-address memory with last-write-wins updates, no parameter changes.

Use EXACT original DualAddressAMN weights (no retraining):
    upsert = DualAddressUpsertAMN(48)
    upsert.load_state_dict(saved_dual['state_dict'], strict=True)

Inputs must be existing structured (entity, relation, value) triples.
This model does not parse free-form Turkish and cannot generalize vocabulary.
"""
import torch
from torch.nn import functional as F

try:
    from relational_qa import DualAddressAMN
except ImportError:
    from experiments.relational_qa import DualAddressAMN


def last_write_mask(triples):
    """True only at the LAST occurrence of each exact (entity, relation) pair.

    Comparison is on discrete token IDs, not approximate learned keys. Stable
    for arbitrary batch-specific sequences and repeated writes.
    """
    if triples.ndim != 3 or triples.size(-1) != 3:
        raise ValueError("Expected [batch, facts, 3] integer triples")
    pairs = triples[..., :2]
    n = pairs.size(1)
    same_pair = (pairs[:, :, None, :] == pairs[:, None, :, :]).all(dim=-1)
    slots = torch.arange(n, device=triples.device)
    later_slot = slots[None, :] > slots[:, None]
    return ~(same_pair & later_slot).any(dim=-1)


class DualAddressUpsertAMN(DualAddressAMN):
    """Protected dual memory + explicit upsert; inherits all learned parameters.

    For each (person, relation), only its final write contributes to the
    Gram solve. Masked slots are decoupled by identity diagonals. On contexts
    without duplicate keys this follows the UNMODIFIED dual formula exactly:
        D = solve(K K^T + 1e-5 I, K)
        S = V^T D

    The discarded old values contribute zero; active independent-key reads
    keep the dual property. This fixes duplicate-key rank deficiency, not
    near-duplicate key issues, nor delete/no-key and free-text parsing.
    """
    def forward(self, triples, query):
        keys = self.address(triples[:, :, :2])
        values = F.normalize(self.value(self.embed(triples[:, :, 2])), dim=-1)
        keep = last_write_mask(triples)
        mask = keep.unsqueeze(-1)
        km = keys * mask
        vm = values * mask
        n = keys.size(1)
        gram = torch.bmm(km, km.transpose(1, 2))
        eye = torch.eye(n, device=keys.device, dtype=keys.dtype)
        # Inactive slots must stay full-rank without coupling to live slots.
        system = gram + 1e-5 * eye + torch.diag_embed((~keep).to(keys.dtype))
        dual = torch.linalg.solve(system, km)
        state = torch.bmm(vm.transpose(1, 2), dual)
        q = self.address(query)
        read = torch.bmm(state, q.unsqueeze(-1)).squeeze(-1)
        return self.out(read)
