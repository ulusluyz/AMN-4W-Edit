"""Sequential A->B learning/forgetting probe on the user's TRAIN split only.

The protocol deliberately creates an independent model for each seed. It does
not fine-tune the official checkpoint and it never reads held-out test pages.
It is a narrow controlled continual-learning diagnostic, not proof of AGI.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from .config import NMCConfig
from .corpus import load_split
from .model import AMNNMC
from .train import chunks, evaluate, _atomic_json
from .train_pdf import validate_source


def phase_train(model, optimizer, doc, *, updates=24, chunk_size=64):
    state=None
    seen=0
    model.train()
    iterations=list(chunks(doc,chunk_size))
    for step in range(updates):
        x,y=iterations[step % len(iterations)]
        if step and step % len(iterations)==0:
            state=None
        logits,state=model(x,state)
        loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        state=state.detach()
        seen+=y.numel()
    return seen


def measure_seed(source, *, seed=17, updates=24, config=None):
    docs=load_split(source,'train')
    if len(docs)<25:
        raise ValueError('Too few training pages for a separate A/B diagnostic')
    a,b=docs[2],docs[25]
    cfg=config or NMCConfig(d_model=24,memory_dim=12,slots=4,layers=1)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model=AMNNMC(cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
    # Both domains are taken from the user-provided PDF TRAIN partition.
    count_a=phase_train(model,optimizer,a,updates=updates)
    after_a_a=evaluate(model,[a],64,max_bytes=256)['bpb']
    after_a_b=evaluate(model,[b],64,max_bytes=256)['bpb']
    count_b=phase_train(model,optimizer,b,updates=updates)
    after_b_a=evaluate(model,[a],64,max_bytes=256)['bpb']
    after_b_b=evaluate(model,[b],64,max_bytes=256)['bpb']
    return {'seed':seed,'updates_per_phase':updates,
            'phase_a_training_bytes':count_a,'phase_b_training_bytes':count_b,
            'a_bpb_after_A':after_a_a,'a_bpb_after_B':after_b_a,
            'b_bpb_before_B':after_a_b,'b_bpb_after_B':after_b_b,
            'a_forgetting_bpb_delta':after_b_a-after_a_a,
            'b_learning_bpb_improvement':after_a_b-after_b_b,
            'a_retained_or_improved':after_b_a<=after_a_a,
            'b_adapted':after_b_b<after_a_b}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--updates-per-phase',type=int,default=24)
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    a=p.parse_args()
    manifest=validate_source(a.data)
    results=[measure_seed(a.data,seed=seed,updates=a.updates_per_phase)
             for seed in a.seeds]
    report={'protocol':'A then B, no replay, same PDF train pages only',
            'source_sha256':manifest['source_sha256'],
            'independent_model_seeds':a.seeds,'results':results,
            'all_seeds_adapted':all(v['b_adapted'] for v in results),
            'all_seeds_retained':all(v['a_retained_or_improved'] for v in results),
            'claim':'A within-PDF plasticity/forgetting probe; not out-of-domain continuous learning proof'}
    _atomic_json(report,a.output)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
