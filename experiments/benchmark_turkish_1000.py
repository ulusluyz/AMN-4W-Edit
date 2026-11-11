"""Run a like-for-like 1000-step timing and held-out teacher-forced text probe.

Source models and data are loaded from THIS repository:
  src/amn4w.py, src/transformer_baseline.py, src/tokenizer.py,
  data/train.json and data/test_new_knowledge.json.

The corpus has just SIX training QA records and FOUR held-out QA records;
two held-out records duplicate training records. This is not validation of
general Turkish language ability. The optimized variant changes neither the
learned parameters nor the AMN memory equation.

Run from repository root:
  python experiments/benchmark_turkish_1000.py --steps 1000 --seeds 0 1 2
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.amn4w import AMN4W
from src.transformer_baseline import DecoderOnlyTransformer
from src.tokenizer import CharTokenizer


class AMNPrecomputed(AMN4W):
    """Precompute token-local K,V,eta,gamma,Q; preserve recurrent state updates."""
    def forward(self, x, memory_state=None):
        B, L = x.shape
        if memory_state is None:
            memory_state = self.embedding.weight.new_zeros(B, self.memory_size, self.memory_size)
        pos = torch.arange(L, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, L)
        emb = self.embedding(x) + self.pos_embedding(pos)
        keys = self.fc_k(emb)
        vals = self.fc_v(emb)
        queries = self.fc_q(emb)
        # Exactly the same normalization as the original model.
        keys = keys / (torch.norm(keys, p=2, dim=-1, keepdim=True) + 1e-8)
        vals = vals / (torch.norm(vals, p=2, dim=-1, keepdim=True) + 1e-8)
        queries = queries / (torch.norm(queries, p=2, dim=-1, keepdim=True) + 1e-8)
        etas = torch.sigmoid(self.fc_eta(emb))
        gammas = torch.sigmoid(self.fc_gamma(emb))
        readouts = []
        for t in range(L):
            key = keys[:, t]
            value = vals[:, t]
            eta = etas[:, t]
            gamma = gammas[:, t]
            old_read = torch.bmm(memory_state, key.unsqueeze(-1)).squeeze(-1)
            mult = (1.0 - gamma) + eta * gamma
            diff = eta * value - mult * old_read
            memory_state = memory_state + torch.bmm(diff.unsqueeze(-1), key.unsqueeze(1))
            q = queries[:, t]
            readouts.append(torch.bmm(memory_state, q.unsqueeze(-1)).squeeze(-1))
        hidden = F.relu(self.fc_out1(torch.cat([emb, torch.stack(readouts, dim=1)], dim=-1)))
        return self.fc_out2(hidden), memory_state


def dataset(max_len):
    train = json.loads((ROOT / "data/train.json").read_text(encoding="utf-8"))
    test = json.loads((ROOT / "data/test_new_knowledge.json").read_text(encoding="utf-8"))
    def text(item):
        context = item["context"] if isinstance(item["context"], str) else " ".join(item["context"])
        return f'{context} Soru: {item["question"]} Cevap: {item["answer"]}'
    rows = train + test
    assert len(train) == 6 and len(test) == 4, "Corpus changed: do not compare old numbers"
    longest = max(len(text(row)) + 2 for row in rows)
    assert max_len + 1 >= longest, ("Truncation would remove answers", longest)
    tok = CharTokenizer()
    # Existing repository training method includes evaluation characters in vocab.
    tok.fit([text(row) for row in rows])
    series = [torch.tensor(tok.encode(text(row), max_len=max_len+1), dtype=torch.long) for row in rows]
    train_xy = torch.stack(series[:len(train)])
    eval_xy = torch.stack(series[len(train):])
    return tok, (train_xy[:, :-1], train_xy[:, 1:]), (eval_xy[:, :-1], eval_xy[:, 1:]), test


def logits(model, x, kind):
    y = model(x)
    return y[0] if kind != "transformer" else y


def cross_entropy(pred, target):
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1), ignore_index=0)


def check_parity(tok, samples, seed, max_len, dim, memory):
    torch.manual_seed(seed)
    baseline = AMN4W(tok.vocab_size, dim, memory, max_len)
    fast = AMNPrecomputed(tok.vocab_size, dim, memory, max_len)
    fast.load_state_dict(baseline.state_dict())
    x, y = samples
    x, y = x[:3], y[:3]
    before, S1 = baseline(x)
    after, S2 = fast(x)
    L1 = cross_entropy(before, y)
    L2 = cross_entropy(after, y)
    L1.backward()
    L2.backward()
    grad_delta = max(float((p.grad-q.grad).abs().max().detach())
                     for p, q in zip(baseline.parameters(), fast.parameters()))
    check = {
        "seed":seed,
        "max_logits_abs":float((before-after).abs().max().detach()),
        "max_state_abs":float((S1-S2).abs().max().detach()),
        "max_grad_abs":grad_delta,
        "loss_abs":float((L1-L2).abs().detach()),
    }
    assert check["max_logits_abs"] < 1e-4 and check["max_state_abs"] < 1e-4
    assert check["max_grad_abs"] < 1e-4
    return check


@torch.no_grad()
def evaluate(model, kind, test_xy, test_rows):
    model.eval()
    x, y = test_xy
    l = logits(model, x, kind)
    mask = y.ne(0)
    correct = l.argmax(dim=-1).eq(y)
    answer_chars, answer_total, exact = 0, 0, 0
    for i, row in enumerate(test_rows):
        context = row["context"] if isinstance(row["context"], str) else " ".join(row["context"])
        prefix = f'{context} Soru: {row["question"]} Cevap: '
        start = len(prefix)
        width = len(row["answer"])
        same = l[i, start:start+width].argmax(-1).eq(y[i, start:start+width])
        answer_chars += int(same.sum())
        answer_total += width
        exact += int(same.all())
    return {
        "loss":float(cross_entropy(l, y)),
        "teacher_forced_token_accuracy":int((correct & mask).sum())/int(mask.sum()),
        "teacher_forced_answer_char_accuracy":answer_chars/answer_total,
        "teacher_forced_answer_exact":exact/len(test_rows),
    }


def train(kind, seed, args, tok, train_xy, test_xy, test_rows):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if kind == "original":
        model = AMN4W(tok.vocab_size, args.dim, args.memory, args.max_len)
    elif kind == "optimized":
        model = AMNPrecomputed(tok.vocab_size, args.dim, args.memory, args.max_len)
    else:
        model = DecoderOnlyTransformer(tok.vocab_size, args.dim, 2, 2, args.max_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x, y = train_xy
    gen = torch.Generator().manual_seed(100000+seed)
    sampled = torch.randint(x.size(0), (args.steps, args.batch), generator=gen)
    t0 = time.perf_counter()
    for i in range(args.steps):
        model.train()
        xb, yb = x[sampled[i]], y[sampled[i]]
        optimizer.zero_grad(set_to_none=True)
        pred = logits(model, xb, kind if kind=="transformer" else "amn")
        loss = cross_entropy(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    duration = time.perf_counter() - t0
    return {
        "arch":kind,"seed":seed,"steps":args.steps,"training_s":duration,
        "params":sum(p.numel() for p in model.parameters()),
        "train_loss":float(loss.detach()),
        "evaluation":evaluate(model, kind if kind=="transformer" else "amn", test_xy, test_rows),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--memory", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", default="results/language-speed-1000.jsonl")
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    tok, training, heldout, rows = dataset(args.max_len)
    print("torch",torch.__version__,"vocab",tok.vocab_size,"training examples",training[0].shape[0],flush=True)
    for seed in args.seeds:
        print("PARITY",json.dumps(check_parity(tok,training,seed,args.max_len,args.dim,args.memory)),flush=True)
    path=ROOT/args.output
    path.parent.mkdir(parents=True,exist_ok=True)
    existing={(v["arch"],v["seed"]) for v in [json.loads(s) for s in path.read_text().splitlines() if s.strip()]} if path.exists() else set()
    for seed in args.seeds:
        for kind in ("original","optimized","transformer"):
            if (kind,seed) in existing:continue
            output=train(kind,seed,args,tok,training,heldout,rows)
            with path.open("a") as fp:fp.write(json.dumps(output,ensure_ascii=False)+"\n")
            print("RUN",json.dumps(output,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
