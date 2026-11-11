"""Training-ceiling early stopping for the existing tiny Turkish corpus.

Only training prefixes control stopping; held-out answers are reported AFTER
training and never used for model selection. This is NOT general language
performance. Run: python experiments/train_until_ceiling.py
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from benchmark_turkish_1000 import (
    AMNPrecomputed, AMN4W, DecoderOnlyTransformer, dataset, cross_entropy
)
from accuracy_ceiling_audit import compute_ceiling


def forward(model, kind, x):
    out = model(x)
    return out if kind == "transformer" else out[0]


@torch.no_grad()
def evaluate_answers(model, kind, held, test_rows, training_texts):
    model.eval()
    x, y = held
    output = forward(model, kind, x)
    result = []
    for i, row in enumerate(test_rows):
        context = row["context"]
        if isinstance(context, list):
            context = " ".join(context)
        prefix = f'{context} Soru: {row["question"]} Cevap: '
        full_text = prefix + row["answer"]
        a, b = len(prefix), len(row["answer"])
        exact = bool(torch.equal(output[i, a:a+b].argmax(-1), y[i, a:a+b]))
        result.append({"row":i, "duplicate_of_training":full_text in training_texts,
                       "answer_exact_teacher_forced":exact})
    novel = [r for r in result if not r["duplicate_of_training"]]
    return {"novel_n":len(novel),
            "novel_answer_exact_teacher_forced":sum(r["answer_exact_teacher_forced"] for r in novel)/len(novel),
            "details":result}


def run(kind, seed, lr, max_steps, interval, stable_checks):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok, training, heldout, test_rows = dataset(96)
    train_rows = json.loads((ROOT / "data/train.json").read_text(encoding="utf-8"))
    def full(row):
        c = row["context"]
        if isinstance(c, list): c = " ".join(c)
        return f'{c} Soru: {row["question"]} Cevap: {row["answer"]}'
    train_texts = set(map(full, train_rows))
    model = (AMNPrecomputed(tok.vocab_size,32,32,96) if kind == "optimized" else
             AMN4W(tok.vocab_size,32,32,96) if kind == "original" else
             DecoderOnlyTransformer(tok.vocab_size,32,2,2,96))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = training
    bound = compute_ceiling(x, y)
    gen = torch.Generator().manual_seed(100000+seed)
    indices = torch.randint(x.size(0), (max_steps,6), generator=gen)
    start = time.perf_counter()
    streak, first, checkpoints = 0, None, []
    for step in range(1, max_steps+1):
        model.train()
        xb, yb = x[indices[step-1]], y[indices[step-1]]
        opt.zero_grad(set_to_none=True)
        pred = forward(model, kind, xb)
        loss = cross_entropy(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % interval != 0 and step != max_steps: continue
        model.eval()
        with torch.no_grad():
            train_logits = forward(model, kind, x)
            correct = int(((train_logits.argmax(-1)==y)&y.ne(0)).sum())
        if correct >= bound["max_correct"]:
            streak += 1
            if first is None: first = step
        else:
            streak = 0
        checkpoints.append({"step":step, "train_correct":correct,
                            "seconds":time.perf_counter()-start})
        if streak >= stable_checks: break
    elapsed = time.perf_counter()-start
    return {"arch":kind, "seed":seed, "lr":lr, "steps_used":step,
            "train_upper_bound":bound["max_accuracy"],
            "train_upper_bound_first_step":first,
            "train_ceiling_stable":streak >= stable_checks,
            "training_s":elapsed, "checkpoints":checkpoints,
            "heldout":evaluate_answers(model,kind,heldout,test_rows,train_texts)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["optimized","transformer"],
                   choices=["original","optimized","transformer"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--interval", type=int, default=25)
    p.add_argument("--stable-checks", type=int, default=2)
    p.add_argument("--output", default="results/comparison/early_stopping_250.jsonl")
    args=p.parse_args(); torch.set_num_threads(1)
    target=ROOT/args.output; target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as out:
        for seed in args.seeds:
            for kind in args.models:
                record=run(kind,seed,args.lr,args.max_steps,args.interval,args.stable_checks)
                out.write(json.dumps(record,ensure_ascii=False)+"\n"); out.flush()
                print(kind,seed,record["steps_used"],round(record["training_s"],3),
                      record["heldout"]["novel_answer_exact_teacher_forced"],flush=True)


if __name__ == "__main__":
    main()
