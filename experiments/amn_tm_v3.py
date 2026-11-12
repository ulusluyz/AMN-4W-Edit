"""AMN-TM v3: native recurrent associative memory, no Transformer.

A 16M-ish parameter Turkish multi-objective model with
1) next-token modeling, 2) intent classification, 3) contrastive
question/evidence retrieval, 4) synthetic multi-turn instruction grounding,
5) final supervised answer generation.

This is a research prototype. An 8k vocabulary and 512-token context are
CONFIGURATION CAPABILITIES; training/evaluation evidence must state actual
sequence length, token exposure, model quality and data provenance.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from amn_open_turkish import AssociativeTextMemory

INTENTS=("explanation","procedure","comparison","calculation","writing","other")
IGNORE=-100
USER_PREFIX="Kullanıcı: "
ASSISTANT_PREFIX="\nAsistan: "

class AMNTMv3(nn.Module):
    def __init__(self,vocab_size=8192,width=544,layers=6,key_dim=128,
                 semantic_dim=192,pad_id=0):
        super().__init__()
        self.config=dict(vocab_size=vocab_size,width=width,layers=layers,
                         key_dim=key_dim,semantic_dim=semantic_dim,pad_id=pad_id)
        self.embedding=nn.Embedding(vocab_size,width,padding_idx=pad_id)
        self.gru=nn.GRU(width,width,num_layers=layers,batch_first=True)
        self.memory=AssociativeTextMemory(width,key_dim)
        self.intent_head=nn.Linear(width,len(INTENTS))
        self.query_head=nn.Linear(width,semantic_dim)
        self.evidence_head=nn.Linear(width,semantic_dim)
        self.output_bias=nn.Parameter(torch.zeros(vocab_size))

    def features(self,ids):
        x=self.embedding(ids)
        h,_=self.gru(x)
        return self.memory(h)

    def forward(self,ids):
        hidden=self.features(ids)
        return F.linear(hidden,self.embedding.weight,self.output_bias)

    def pooled(self,ids):
        hidden=self.features(ids)
        idx=ids.ne(self.config["pad_id"]).sum(-1).clamp_min(1)-1
        return hidden[torch.arange(ids.shape[0],device=ids.device),idx]

    def intent_logits(self,question_ids):
        return self.intent_head(self.pooled(question_ids))

    def retrieval_scores(self,question_ids,answer_ids):
        # Contrastive in-batch negatives: diagonal pairs are positives,
        # off-diagonals are other naturally sourced Turkish answers.
        queries=F.normalize(self.query_head(self.pooled(question_ids)),dim=-1)
        answers=F.normalize(self.evidence_head(self.pooled(answer_ids)),dim=-1)
        return queries @ answers.T / 0.12

    def retrieval_loss(self,question_ids,answer_ids):
        scores=self.retrieval_scores(question_ids,answer_ids)
        target=torch.arange(len(scores),device=scores.device)
        return 0.5*(F.cross_entropy(scores,target)+F.cross_entropy(scores.T,target))

    def step(self,token,state=None):
        previous,mat,ks=(None,None,None) if state is None else state
        x=self.embedding(token[:,None])
        h,previous=self.gru(x,previous)
        mem,mat,ks=self.memory.step(h[:,0],mat,ks)
        return F.linear(mem,self.embedding.weight,self.output_bias),(previous,mat,ks)

    @torch.no_grad()
    def generate(self,sp,question,max_new_tokens=120,temperature=0.7,top_k=32,seed=17):
        self.eval()
        device=next(self.parameters()).device
        source=[sp.bos_id()]+sp.encode(USER_PREFIX+question.strip()+ASSISTANT_PREFIX)
        if len(source)>512:
            raise ValueError("Prompt exceeds configured 512-token context")
        state=None
        for token in source:
            logits,state=self.step(torch.tensor([token],device=device),state)
        outputs=[]
        rng=torch.Generator(device="cpu").manual_seed(seed)
        eos=False
        for _ in range(max_new_tokens):
            scores=logits[0].detach().float().cpu().clone()
            for special in (sp.bos_id(),sp.pad_id()):
                if special>=0:scores[special]=-1e9
            if temperature<=0:
                selected=int(scores.argmax())
            else:
                scores=scores/max(temperature,1e-3)
                if top_k>0:
                    val,ix=scores.topk(min(top_k,scores.numel()))
                    selected=int(ix[torch.multinomial(val.softmax(-1),1,generator=rng).item()])
                else:
                    selected=int(torch.multinomial(scores.softmax(-1),1,generator=rng))
            if selected==sp.eos_id():
                eos=True
                break
            outputs.append(selected)
            logits,state=self.step(torch.tensor([selected],device=device),state)
        text=sp.decode(outputs)
        return dict(text=text,eos=eos,tokens=len(outputs))

def build_sft(sp,question,answer,max_len=160,supervise_prompt=False):
    prefix=[sp.bos_id()]+sp.encode(USER_PREFIX+question.strip()+ASSISTANT_PREFIX)
    suffix=sp.encode(answer.strip())+[sp.eos_id()]
    if len(prefix)+len(suffix)-1>max_len:
        return None
    full=prefix+suffix
    x=full[:-1]
    y=full[1:]
    if not supervise_prompt:
        y[:len(prefix)-1]=[IGNORE]*(len(prefix)-1)
    return x,y

def build_answer_lm(sp,answer,max_len=160):
    tokens=[sp.bos_id()]+sp.encode(answer.strip())+[sp.eos_id()]
    if len(tokens)-1>max_len:return None
    return tokens[:-1],tokens[1:]

def collate_lm(rows,pad_id):
    length=max(len(x) for x,_ in rows)
    x=torch.full((len(rows),length),pad_id,dtype=torch.long)
    y=torch.full((len(rows),length),IGNORE,dtype=torch.long)
    for j,(a,b) in enumerate(rows):
        x[j,:len(a)]=torch.tensor(a,dtype=torch.long)
        y[j,:len(b)]=torch.tensor(b,dtype=torch.long)
    return x,y

def collate_questions(sp,strings,max_len=80):
    encoded=[sp.encode(s)[:max_len-1] for s in strings]
    size=max(len(x) for x in encoded)+1
    tensor=torch.full((len(strings),size),sp.pad_id(),dtype=torch.long)
    for i,row in enumerate(encoded):
        tokens=[sp.bos_id()]+row
        tensor[i,:len(tokens)]=torch.tensor(tokens,dtype=torch.long)
    return tensor

def save(path,model,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"architecture":"AMNTMv3","config":model.config,
                "state_dict":model.state_dict(),"metadata":metadata},path)

def restore(path,device="cpu"):
    record=torch.load(path,map_location=device,weights_only=True)
    if record.get("architecture")!="AMNTMv3":
        raise ValueError("Wrong checkpoint architecture")
    model=AMNTMv3(**record["config"])
    model.load_state_dict(record["state_dict"],strict=True)
    return model.to(device),record.get("metadata",{})

def parameters(model):
    return sum(p.numel() for p in model.parameters())
