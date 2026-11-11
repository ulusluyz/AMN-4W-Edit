"""AMN text-to-text prototype without regex entity extraction or Transformer attention.

Pure byte-level Turkish prompt -> autoregressive answer generation.  UTF-8
bytes cover any user input without an OOV token. The model is a recurrent GRU
controller plus causal normalized associative memory: S_t = S_{t-1}+k_t v_t^T.
This is an isolated experimental model, not a replacement for AMN-4W weights.

Example:
    python experiments/amn_open_turkish.py ask --checkpoint results/open_turkish/best.pt \
        --prompt 'Türkiye\'nin başkenti nedir?' --max-new-bytes 350
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

BOS, EOS, PAD = 256, 257, 258
VOCAB_SIZE = 259
IGNORE = -100
QUESTION_PREFIX = "Kullanıcı: "
ANSWER_PREFIX = "\nAsistan: "


def make_prefix(question: str) -> bytes:
    return (QUESTION_PREFIX + question.strip() + ANSWER_PREFIX).encode("utf-8")


def make_training_example(question: str, answer: str, max_len: int = 384):
    prefix = make_prefix(question)
    response = answer.strip().encode("utf-8")
    # No silent answer truncation: changing labels via cropping invalidates CE.
    if len(prefix) + len(response) + 1 > max_len:
        return None
    raw = [BOS] + list(prefix) + list(response) + [EOS]
    x = raw[:-1]
    y = raw[1:]
    prefix_len = 1 + len(prefix)
    # y[i] is the byte AFTER x[i]; supervise first answer byte onward.
    y[:prefix_len - 1] = [IGNORE] * (prefix_len - 1)
    return x, y


def collate(rows, device='cpu'):
    max_len = max(len(x) for x,_ in rows)
    x=torch.full((len(rows),max_len), PAD, dtype=torch.long,device=device)
    y=torch.full((len(rows),max_len), IGNORE,dtype=torch.long,device=device)
    for i,(a,b) in enumerate(rows):
        x[i,:len(a)] = torch.tensor(a,device=device)
        y[i,:len(b)] = torch.tensor(b,device=device)
    return x,y


class AssociativeTextMemory(nn.Module):
    """Differentiable normalized, additive key/value AMN memory.

    The parallel version computes the exact causal prefix of rank-one writes;
    step() uses only the previous (matrix, key_sum) state.  No self-attention
    or Transformer blocks are used.
    """
    def __init__(self,width=96,key_dim=24):
        super().__init__()
        self.key=nn.Linear(width,key_dim)
        self.query=nn.Linear(width,key_dim)
        self.value=nn.Linear(width,width)
        self.readout=nn.Linear(width,width)
        self.gate=nn.Linear(width,width)
        self.norm=nn.LayerNorm(width)
        self.width=width
        self.key_dim=key_dim

    def projection(self,x):
        # Positive keys permit a stable normalized associative readout.
        return F.elu(self.key(x)) + 1., F.elu(self.query(x)) + 1.,self.value(x)

    def forward(self,x):
        k,q,v=self.projection(x)
        # Memory prefix includes the CURRENT byte (valid for next-byte labels).
        sum_k=torch.cumsum(k,dim=1)
        state=torch.cumsum(k.unsqueeze(-1)*v.unsqueeze(-2),dim=1)
        read=torch.einsum('btk,btkd->btd',q,state)
        denom=(q*sum_k).sum(-1,keepdim=True).clamp_min(1e-5)
        read=read/denom
        return self.norm(x+torch.sigmoid(self.gate(x))*self.readout(read))

    def step(self,x,state=None,key_sum=None):
        k,q,v=self.projection(x)
        if state is None:
            state=torch.zeros(x.shape[0],self.key_dim,self.width,device=x.device,dtype=x.dtype)
            key_sum=torch.zeros(x.shape[0],self.key_dim,device=x.device,dtype=x.dtype)
        state=state+k.unsqueeze(-1)*v.unsqueeze(-2)
        key_sum=key_sum+k
        read=(q.unsqueeze(-2)@state).squeeze(-2)
        den=(q*key_sum).sum(-1,keepdim=True).clamp_min(1e-5)
        out=self.norm(x+torch.sigmoid(self.gate(x))*self.readout(read/den))
        return out,state,key_sum


class AMNOpenTurkish(nn.Module):
    def __init__(self,width=96,key_dim=24,layers=2):
        super().__init__()
        self.cfg={"width":width,"key_dim":key_dim,"layers":layers}
        self.embedding=nn.Embedding(VOCAB_SIZE,width,padding_idx=PAD)
        self.controller=nn.GRU(width,width,num_layers=layers,batch_first=True)
        self.memory=AssociativeTextMemory(width,key_dim)
        self.head=nn.Linear(width,VOCAB_SIZE)

    def forward(self,token_ids):
        x=self.embedding(token_ids)
        hidden,_=self.controller(x)
        memory=self.memory(hidden)
        return self.head(memory)

    def step(self,token,state=None):
        h,mat,ks=(None,None,None) if state is None else state
        emb=self.embedding(token[:,None])
        controller_out,h=self.controller(emb,h)
        mem,mat,ks=self.memory.step(controller_out[:,0],mat,ks)
        return self.head(mem),(h,mat,ks)

    @torch.no_grad()
    def complete(self,prompt,max_new_bytes=384,temperature=0.8,top_k=20,seed=13):
        self.eval()
        gen=torch.Generator(device='cpu').manual_seed(seed)
        initial=[BOS]+list(make_prefix(prompt))
        state=None
        for v in initial:
            logits,state=self.step(torch.tensor([v],device=next(self.parameters()).device),state)
        generated=[]
        for _ in range(max_new_bytes):
            sample=logits[0].float().cpu()
            sample[BOS]=-1e6;sample[PAD]=-1e6
            if temperature<=0:
                token=int(sample.argmax())
            else:
                sample=sample/max(temperature,1e-3)
                if top_k>0 and top_k<len(sample):
                    val,ix=torch.topk(sample,top_k)
                    probs=torch.softmax(val,dim=-1)
                    token=int(ix[torch.multinomial(probs,1,generator=gen).item()])
                else:
                    token=int(torch.multinomial(torch.softmax(sample,-1),1,generator=gen))
            if token==EOS:break
            generated.append(token)
            logits,state=self.step(torch.tensor([token],device=next(self.parameters()).device),state)
        raw=bytes(t for t in generated if t < 256)
        return {"text":raw.decode('utf-8',errors='replace'),
                "bytes_generated":len(raw),"valid_utf8":'\ufffd' not in raw.decode('utf-8',errors='replace'),
                "eos":token==EOS}


def save_checkpoint(path, model, *, extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"architecture":"AMNOpenTurkish-v1", "config":model.cfg,
                "state_dict":model.state_dict(), "metadata": extra or {}},path)


def load_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=True)
    assert ck['architecture']=='AMNOpenTurkish-v1'
    m=AMNOpenTurkish(**ck['config'])
    m.load_state_dict(ck['state_dict'],strict=True)
    return m.to(device),ck.get('metadata',{})


def main():
    cli=argparse.ArgumentParser()
    sub=cli.add_subparsers(dest='command',required=True)
    x=sub.add_parser('ask')
    x.add_argument('--checkpoint',default='results/open_turkish/best.pt')
    x.add_argument('--prompt',required=True)
    x.add_argument('--max-new-bytes',type=int,default=384)
    x.add_argument('--temperature',type=float,default=.7)
    x.add_argument('--top-k',type=int,default=20)
    x.add_argument('--seed',type=int,default=13)
    x=sub.add_parser('inspect')
    x.add_argument('--checkpoint',default='results/open_turkish/best.pt')
    args=cli.parse_args()
    model,metadata=load_checkpoint(args.checkpoint)
    if args.command=='inspect':
        print(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),
                          'architecture':model.cfg,'metadata':metadata},ensure_ascii=False,indent=2))
    else:
        output=model.complete(args.prompt,args.max_new_bytes,args.temperature,args.top_k,args.seed)
        print(json.dumps({'prompt':args.prompt,'response':output,'training':metadata},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
