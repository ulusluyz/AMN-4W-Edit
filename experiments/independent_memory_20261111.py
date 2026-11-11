"""Independent CPU-only experimental memory probe (NOT the AMN-4W v0.4 runtime).
Reproduction: python experiments/independent_memory_20261111.py --numeric
              python experiments/independent_memory_20261111.py --train --steps 350
The experiment uses a standalone, synthetic A->B->C model and does not change any
production module. The --train run is not the full 3000-step sweep.
"""
import argparse
import json
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)


def unit(x):
    return F.normalize(x, dim=-1)


def read(S, k):
    return torch.bmm(S, k.unsqueeze(-1)).squeeze(-1)


def write(S, k, v, eta):
    return S + torch.bmm((eta * (v - read(S, k))).unsqueeze(-1), k.unsqueeze(1))


def span_residual(others, k):
    # Rank threshold reflects the incoming dtype, not the float64 SVD workspace.
    tol = max(others.shape[-2:]) * torch.finfo(others.dtype).eps
    U, s, _ = torch.linalg.svd(others.double(), full_matrices=False)
    valid = s > tol * s[:, :1]
    U = U * valid.unsqueeze(1)
    kd = k.double()
    return kd - torch.einsum("bdn,bn->bd", U, torch.einsum("bdn,bd->bn", U, kd))


def dual_write(S, K, k, v, eta, min_rho=1e-6):
    r = span_residual(K.transpose(1, 2), k)
    rho = r.norm(dim=-1)
    ok = rho >= min_rho
    dual = r / rho.clamp(min=min_rho).square().unsqueeze(-1)
    S64, k64 = S.double(), k.double()
    delta = eta.double() * (v.double() - read(S64, k64))
    nextS = S64 + torch.bmm(delta.unsqueeze(-1), dual.unsqueeze(1))
    return torch.where(ok[:, None, None], nextS, S64).to(S.dtype), rho, ok


def dual_erase(S, K, idx, min_rho=1e-6):
    b, N, d = K.shape
    rows = torch.arange(b, device=K.device)
    keep = torch.ones((b, N), dtype=torch.bool, device=K.device)
    keep[rows, idx] = False
    k = K[rows, idx]
    others = K[keep].reshape(b, N - 1, d).transpose(1, 2)
    r = span_residual(others, k)
    rho = r.norm(dim=-1)
    ok = rho >= min_rho
    dual = r / rho.clamp(min=min_rho).square().unsqueeze(-1)
    S64 = S.double()
    update = torch.bmm(read(S64, k.double()).unsqueeze(-1), dual.unsqueeze(1))
    return torch.where(ok[:, None, None], S64 - update, S64).to(S.dtype), rho, ok


def numeric(seed=42, trials=200, d=64, N=8):
    torch.manual_seed(seed)
    K = unit(torch.randn(trials, N, d))
    V = unit(torch.randn(trials, N, d))
    S = torch.zeros(trials, d, d)
    one = torch.ones(trials, 1)
    for i in range(N):
        S = write(S, K[:, i], V[:, i], one)
    idx = torch.zeros(trials, dtype=torch.long)
    deleted, _, ok = dual_erase(S, K, idx)
    result = {
        "seed": seed,
        "target_remaining_max": read(deleted, K[:, 0]).norm(dim=-1).max().item(),
        "neighbor_change_max": torch.einsum(
            "bij,bnj->bni", deleted - S, K[:, 1:]
        ).abs().max().item(),
        "erase_ok_fraction": ok.float().mean().item(),
        "writes": {},
    }
    Kdup = K.clone()
    Kdup[:, 2] = Kdup[:, 1]
    Sdup = torch.zeros_like(S)
    for i in range(N):
        Sdup = write(Sdup, Kdup[:, i], V[:, i], one)
    Sdup_del, _, dup_ok = dual_erase(Sdup, Kdup, idx)
    result["duplicate_neighbor_change_max"] = torch.einsum(
        "bij,bnj->bni", Sdup_del - Sdup, Kdup[:, 1:]
    ).abs().max().item()
    result["duplicate_erase_ok_fraction"] = dup_ok.float().mean().item()
    for c in (0.3, 0.9, 0.99, 0.9999):
        u = unit(torch.randn(trials, d))
        u = unit(u - (u * K[:, 1]).sum(dim=-1, keepdim=True) * K[:, 1])
        k = unit(c * K[:, 1] + math.sqrt(1 - c*c) * u)
        v = unit(torch.randn(trials, d))
        standard = write(S, k, v, one)
        dual, rho, okw = dual_write(S, K, k, v, one)
        before = read(S, K[:, 1])
        result["writes"][str(c)] = {
            "standard_neighbor_cos": F.cosine_similarity(before, read(standard, K[:, 1])).mean().item(),
            "dual_neighbor_cos": F.cosine_similarity(before, read(dual, K[:, 1])).mean().item(),
            "dual_max_neighbor_change": torch.einsum("bij,bnj->bni", dual-S, K).abs().max().item(),
            "dual_update_fro": (dual-S).flatten(1).norm(dim=1).mean().item(),
            "mean_rho": rho.mean().item(),
            "write_ok_fraction": okw.float().mean().item(),
        }
    return result


class TinyAMN(nn.Module):
    """Deliberately independent synthetic 2-hop proxy, not the repository runtime."""
    def __init__(self, V=50, d=64, chains=2):
        super().__init__()
        self.d = d
        self.embedding = nn.Embedding(V, d, padding_idx=0)
        self.pos = nn.Embedding(3*chains+2, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.eta = nn.Linear(d, 1)
        self.q = nn.Linear(d, d)
        self.q2 = nn.Linear(d, d)
        self.out1 = nn.Linear(2*d, d)
        self.out2 = nn.Linear(d, V)

    def memory(self, ctx, mask):
        tok = self.embedding(ctx)
        prev = F.pad(tok, (0, 0, 1, 0))[:, :-1]
        S = tok.new_zeros(ctx.size(0), self.d, self.d)
        for t in range(1, ctx.size(1)):
            eta = self.eta(tok[:, t]).sigmoid() * mask[:, t:t+1]
            S = write(S, unit(self.k(prev[:, t])), unit(self.v(tok[:, t])), eta)
        return S

    def query(self, S, a, index):
        tok = self.embedding(a)
        q1 = unit(self.q(tok))
        r1 = read(S, q1)
        q2 = unit(self.q2(r1))
        r2 = read(S, q2)
        pos = self.pos(torch.full_like(a, index))
        logits = self.out2(F.relu(self.out1(torch.cat([tok+pos, r2], -1))))
        return logits, (q1, r1, q2, r2)


def batch(bs, V, M):
    contexts, A, B, C = [], [], [], []
    for _ in range(bs):
        sampled = random.sample(range(2, V), 3*M)
        chains = [sampled[3*i:3*i+3] for i in range(M)]
        chosen = random.randrange(M)
        contexts.append([t for triple in chains for t in triple])
        A.append(chains[chosen][0])
        B.append(chains[chosen][1])
        C.append(chains[chosen][2])
    return (
        torch.tensor(contexts),
        torch.tensor([[0, 1, 1]*M]*bs, dtype=torch.float32),
        torch.tensor(A), torch.tensor(B), torch.tensor(C)
    )


def train(seed=0, leak_weight=0, decor_weight=0, steps=350, V=50, M=2):
    torch.manual_seed(seed)
    random.seed(seed)
    model = TinyAMN(V, 64, M)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for _ in range(steps):
        ctx, mask, a, b, c = batch(64, V, M)
        S = model.memory(ctx, mask)
        logits, (q1, r1, q2, r2) = model.query(S, a, 3*M)
        with torch.no_grad():
            ka, kb = unit(model.k(model.embedding(a))), unit(model.k(model.embedding(b)))
            vb, vc = unit(model.v(model.embedding(b))), unit(model.v(model.embedding(c)))
        loss = F.cross_entropy(logits, c)
        loss += sum(1-F.cosine_similarity(x, y, dim=-1).mean() for x, y in
                    ((q1, ka), (r1, vb), (q2, kb), (r2, vc)))
        if leak_weight:
            cand = ctx[:, 2::3]
            kc = unit(model.k(model.embedding(cand)))
            leak = torch.einsum("bij,bmj->bmi", S, kc)
            loss += leak_weight * leak.square().sum(-1).mean()
        if decor_weight:
            keys = ctx.reshape(64, M, 3)[:, :, :2].reshape(64, -1)
            K = unit(model.k(model.embedding(keys)))
            G = K @ K.transpose(1, 2)
            off = G - torch.diag_embed(torch.diagonal(G, dim1=1, dim2=2))
            loss += decor_weight * off.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        random.seed(10000+seed)
        torch.manual_seed(10000+seed)
        ctx, mask, a, b, c = batch(500, V, M)
        S = model.memory(ctx, mask)
        logits, (_, _, _, r2) = model.query(S, a, 3*M)
        cand = ctx[:, 2::3]
        kc = unit(model.k(model.embedding(cand)))
        kb = unit(model.k(model.embedding(b)))
        leak = ((S @ kc.transpose(1, 2)).norm(dim=1).mean(-1)
                / (read(S, kb).norm(dim=-1)+1e-8)).mean().item()
        ids = torch.arange(2, V)
        values = unit(model.v(model.embedding(ids)))
        r2_acc = (ids[(r2 @ values.T).argmax(-1)] == c).float().mean().item()
        return {"seed": seed, "final": (logits.argmax(-1)==c).float().mean().item(),
                "r2_C": r2_acc, "leak": leak}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    print("torch", torch.__version__)
    if args.numeric:
        print("NUMERIC", json.dumps(numeric(), ensure_ascii=False))
    if args.train:
        for name, leak, decor in (("baseline", 0, 0), ("leak0.5", 0.5, 0),
                                  ("decorr5", 0, 5), ("combined", 0.5, 5)):
            for seed in range(args.seeds):
                print("TRAIN", name, json.dumps(train(seed, leak, decor, args.steps)), flush=True)


if __name__ == "__main__":
    main()
