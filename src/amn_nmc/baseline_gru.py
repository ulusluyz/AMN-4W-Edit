"""Parameter-matched causal GRU reference, never part of the AMN architecture."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .corpus import load_split
from .train import evaluate, _make_order, atomic_save, _atomic_json
from .train_pdf import validate_source


class GRUByteLM(nn.Module):
    """Plain byte-level recurrent language model for controlled comparison."""
    def __init__(self, width=68, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(256, width)
        self.gru = nn.GRU(width, width, num_layers=layers, batch_first=True)
        self.head = nn.Linear(width, 256)
        self.width, self.layers = width, layers

    def forward(self, tokens, state=None):
        x = self.embedding(tokens)
        y, state = self.gru(x, state)
        return self.head(y), state


def train_reference(data, output, config, seed=17, width=68, layers=2):
    """Same documents, order, byte budget, optimizer and CE as NMC. No task labels."""
    manifest = validate_source(data)
    settings = json.loads(Path(config).read_text(encoding='utf-8'))['training']
    torch.set_num_threads(int(settings.get('threads', 2)))
    torch.manual_seed(seed)
    model = GRUByteLM(width, layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.get('lr', 7e-4),
                                  betas=(0.9, 0.95), weight_decay=settings.get('weight_decay', 0.01))
    docs = load_split(data, 'train')
    valid = load_split(data, 'validation')
    chunk_size = int(settings.get('chunk_size', 64))
    nsteps = int(settings.get('max_steps', 900))
    eval_every = int(settings.get('eval_every', 225))
    budget = settings.get('eval_max_bytes', 4096)
    folder = Path(output);folder.mkdir(parents=True, exist_ok=False)
    history = []
    steps, seen_bytes = 0, 0
    visited = set()
    initial = evaluate(model, valid, chunk_size, max_bytes=budget)
    best = initial
    atomic_save({'model':model.state_dict(),'config':{'width':width,'layers':layers},
                 'seed':seed,'step':0,'validation':initial},folder/'best.pt')
    for epoch in range(int(settings.get('epochs', 5))):
        for doc_idx in _make_order(seed,epoch,len(docs)):
            doc = docs[doc_idx]
            state=None
            bs = torch.tensor(list(doc),dtype=torch.long)
            for start in range(0,len(doc)-1,chunk_size):
                end=min(start+chunk_size,len(doc)-1)
                x=bs[start:end].unsqueeze(0)
                y=bs[start+1:end+1].unsqueeze(0)
                model.train()
                logits,state=model(x,state)
                loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
                if not torch.isfinite(loss):raise FloatingPointError('GRU nonfinite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                if not torch.isfinite(norm):raise FloatingPointError('GRU nonfinite gradient')
                optimizer.step()
                state=state.detach()
                steps+=1
                visited.add(doc_idx)
                seen_bytes+=int(y.numel())
                hist={'step':steps,'training_bytes':seen_bytes,'train_bpb':float(loss.detach()/math.log(2))}
                if eval_every and steps%eval_every==0:
                    dev=evaluate(model,valid,chunk_size,max_bytes=budget)
                    hist['validation_bpb']=dev['bpb']
                    if dev['bpb']<best['bpb']:
                        best=dev
                        atomic_save({'model':model.state_dict(),'config':{'width':width,'layers':layers},
                                     'seed':seed,'step':steps,'validation':best},folder/'best.pt')
                history.append(hist)
                if steps>=nsteps:break
            if steps>=nsteps:break
        if steps>=nsteps:break
    last=evaluate(model,valid,chunk_size,max_bytes=budget)
    if last['bpb']<best['bpb']:
        best=last
        atomic_save({'model':model.state_dict(),'config':{'width':width,'layers':layers},
                     'seed':seed,'step':steps,'validation':best},folder/'best.pt')
    full=evaluate(model,valid,chunk_size)
    stats={'model':'plain_causal_GRU_baseline_not_AMN','seed':seed,
           'params':sum(p.numel() for p in model.parameters()),'steps':steps,
           'training_bytes':seen_bytes,'visited_training_pages':len(visited),
           'initial_validation':initial,'final_validation':full,
           'best_validation_budgeted':best,
           'source_sha256':manifest['source_sha256'],
           'limitation':manifest['split_type']}
    atomic_save({'model':model.state_dict(),'config':{'width':width,'layers':layers},
                 'seed':seed,'step':steps,'validation':full},folder/'last.pt')
    _atomic_json(stats,folder/'summary.json')
    with (folder/'training.jsonl').open('w',encoding='utf-8') as output_log:
        for row in history:output_log.write(json.dumps(row)+'\n')
    return stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--config',default='configs/nmc/cpu_pilot_96k.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--width',type=int,default=68)
    args=parser.parse_args()
    print(json.dumps(train_reference(args.data,args.output,args.config,args.seed,args.width),indent=2))


if __name__=='__main__':main()
