"""SentencePiece BPE + recurrent AMN text-to-text generator (no Transformer).

This is a separate tokenizer and model. It does not mutate the byte-level AMN
checkpoint or any previous AMN-4W experimental weights.
"""
import argparse,json,math
from pathlib import Path
import sentencepiece as spm
import torch
from torch import nn
from torch.nn import functional as F

from amn_open_turkish import AssociativeTextMemory

PREFIX='Kullanıcı: '
SUFFIX='\nAsistan: '
IGNORE=-100


def load_sp(path):
    sp=spm.SentencePieceProcessor()
    assert sp.load(str(path))
    return sp


def encode(sp,question,answer,max_len=192):
    p=[sp.bos_id()]+sp.encode(PREFIX+question.strip()+SUFFIX,out_type=int)
    a=sp.encode(answer.strip(),out_type=int)+[sp.eos_id()]
    if len(p)+len(a)-1>max_len:return None
    x=(p+a)[:-1]
    y=(p+a)[1:]
    y[:len(p)-1]=[IGNORE]*(len(p)-1)
    return x,y


def collate(rows,pad):
    n=max(len(x) for x,_ in rows)
    x=torch.full((len(rows),n),pad,dtype=torch.long)
    y=torch.full((len(rows),n),IGNORE,dtype=torch.long)
    for i,(a,b) in enumerate(rows):
        x[i,:len(a)]=torch.tensor(a)
        y[i,:len(b)]=torch.tensor(b)
    return x,y


class AMNSubwordLM(nn.Module):
    def __init__(self,vocab_size,width=160,key_dim=32,layers=2,pad_id=0):
        super().__init__()
        self.config={'vocab_size':vocab_size,'width':width,'key_dim':key_dim,'layers':layers,'pad_id':pad_id}
        self.embedding=nn.Embedding(vocab_size,width,padding_idx=pad_id)
        self.gru=nn.GRU(width,width,num_layers=layers,batch_first=True)
        self.memory=AssociativeTextMemory(width,key_dim)
        self.out=nn.Linear(width,vocab_size)

    def forward(self,tokens):
        x=self.embedding(tokens)
        h,_=self.gru(x)
        return self.out(self.memory(h))

    def step(self,token,state=None):
        hidden,mem,keys=(None,None,None) if state is None else state
        x=self.embedding(token[:,None])
        h,hidden=self.gru(x,hidden)
        read,mem,keys=self.memory.step(h[:,0],mem,keys)
        return self.out(read),(hidden,mem,keys)

    @torch.no_grad()
    def generate(self,sp,question,max_new_tokens=160,temperature=.7,top_k=20,seed=42):
        self.eval()
        source=[sp.bos_id()]+sp.encode(PREFIX+question.strip()+SUFFIX)
        state=None
        for token in source:
            next_logits,state=self.step(torch.tensor([token]),state)
        rng=torch.Generator(device='cpu').manual_seed(seed)
        output=[]
        stop=False
        for _ in range(max_new_tokens):
            scores=next_logits[0].cpu().float()
            for i in [sp.bos_id(),sp.pad_id()]:scores[i]=-1e9
            if temperature<=0:
                nxt=int(scores.argmax())
            else:
                scores=scores/max(temperature,1e-3)
                if top_k>0:
                    val,ix=scores.topk(min(top_k,len(scores)))
                    p=val.softmax(-1)
                    nxt=int(ix[torch.multinomial(p,1,generator=rng).item()])
                else:
                    p=scores.softmax(-1)
                    nxt=int(torch.multinomial(p,1,generator=rng))
            if nxt==sp.eos_id():
                stop=True
                break
            output.append(nxt)
            next_logits,state=self.step(torch.tensor([nxt]),state)
        txt=sp.decode(output)
        return {'text':txt,'tokens_generated':len(output),'eos':stop,'words':len(txt.split())}


def save(path,m,metadata):
    torch.save({'architecture':'AMNSubwordLM-v1','cfg':m.config,'state_dict':m.state_dict(),
                'metadata':metadata},path)


def restore(path):
    c=torch.load(path,map_location='cpu',weights_only=True)
    assert c['architecture']=='AMNSubwordLM-v1'
    model=AMNSubwordLM(**c['cfg']);model.load_state_dict(c['state_dict'],strict=True)
    model.eval()
    return model,c.get('metadata',{})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='results/open_turkish_sp/best.pt')
    p.add_argument('--tokenizer',default='results/open_turkish_sp/tokenizer.model')
    p.add_argument('--prompt',required=True)
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--max-new-tokens',type=int,default=160)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--top-k',type=int,default=20)
    a=p.parse_args()
    m,metadata=restore(a.checkpoint)
    sp=load_sp(a.tokenizer)
    response=m.generate(sp,a.prompt,a.max_new_tokens,a.temperature,a.top_k,a.seed)
    print(json.dumps({'prompt':a.prompt,'answer':response,'metadata':metadata},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
