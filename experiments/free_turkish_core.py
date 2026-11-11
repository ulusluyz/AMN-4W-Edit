"""AMN-4W Edit: finite-domain Turkish text and genuinely autoregressive long-answer decoding.

A separate learned neural tagger interprets Turkish input. Frozen Dual Upsert
retrieves structured relations; this GRU decoder emits answer words one by one.
Training references are a finite collection of handcrafted Turkish examples:
not general free-topic language understanding, infinite vocabulary, or original
literary composition. No label strings are used during generation.
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent))
from relational_qa import NAMES, VALUES, VOCAB, ID, V, FACTS, generate
from dual_upsert import DualAddressUpsertAMN

ROOT=Path(__file__).resolve().parents[1]
NAME_LIST=tuple(NAMES)
VALUE_LIST=tuple(VALUES['renk']) + tuple(VALUES['konum'])
REL_LIST=('renk','konum')
GENITIVE={
 'Ali': "Ali'nin",'Ayşe':"Ayşe'nin",'Can':"Can'ın",'Deniz':"Deniz'in",
 'Ece':"Ece'nin",'Efe':"Efe'nin",'Elif':"Elif'in",'Emre':"Emre'nin",
 'Mert':"Mert'in",'Mina':"Mina'nın",'Murat':"Murat'ın",'Nisa':"Nisa'nın",
 'Okan':"Okan'ın",'Selin':"Selin'in",'Sena':"Sena'nın",'Umut':"Umut'un",
 'Yaren':"Yaren'in",'Yiğit':"Yiğit'in",'Zeynep':"Zeynep'in",'Arda':"Arda'nın",
}
LOC_FORM={'ev':'evde','okul':'okulda','park':'parkta','ofis':'ofiste',
          'bahçe':'bahçede','kafe':'kafede','salon':'salonda','garaj':'garajda'}
LOC_END={'ev':'evdedir.','okul':'okuldadır.','park':'parktadır.',
         'ofis':'ofistedir.','bahçe':'bahçededir.','kafe':'kafededir.',
         'salon':'salondadır.','garaj':'garajdadır.'}
COLOR_END={'kırmızı':'kırmızıdır.','mavi':'mavidir.','yeşil':'yeşildir.',
           'sarı':'sarıdır.','mor':'mordur.','beyaz':'beyazdır.',
           'siyah':'siyahtır.','turuncu':'turuncudur.'}

# Patterns are DATA GENERATION only, not parser rules. Test uses held-out
# phrasings to probe compositional transfer, and wholly unseen fact bindings.
FACT_FORMS={
 'renk':[
  lambda n,v:f"{GENITIVE[n]} kaleminin rengi {v}.",
  lambda n,v:f"{n} için kalem rengi {v} olarak belirtildi.",
  lambda n,v:f"{n} adlı kişinin kullandığı kalem {v} renktedir.",
  lambda n,v:f"{GENITIVE[n]} kalemi {v} görünüyor.",
  lambda n,v:f"{GENITIVE[n]} kalemi {v} renkte.",
  lambda n,v:f"{n} bir {v} kalem kullanıyor.",
  lambda n,v:f"Kaleminin rengini sorarsan {n} için cevap {v}.",
  lambda n,v:f"{GENITIVE[n]} kaleminden söz ediliyor ve renk {v}.",
  lambda n,v:f"{n} hakkında kaleminin {v} olduğu bilgisi var.",
  lambda n,v:f"Renk bilgisi {n} için {v} olarak kaydedilmiş.",
  lambda n,v:f"{n}, {v} renginde bir kaleme sahip.",
  lambda n,v:f"{GENITIVE[n]} kullandığı kalem {v} renkle tarif ediliyor.",
 ],
 'konum':[
  lambda n,v:f"{n} şu anda {LOC_FORM[v]}.",
  lambda n,v:f"{n} bugün {LOC_FORM[v]} bulunuyor.",
  lambda n,v:f"{GENITIVE[n]} konumu {LOC_FORM[v]} olarak bildirildi.",
  lambda n,v:f"{n} şimdi {LOC_FORM[v]} vakit geçiriyor.",
  lambda n,v:f"{n} için yer bilgisi {LOC_FORM[v]}.",
  lambda n,v:f"{n} halen {LOC_FORM[v]} görünüyor.",
  lambda n,v:f"Bulunduğu yeri sorarsan {n} {LOC_FORM[v]}.",
  lambda n,v:f"Konum kaydında {n} {LOC_FORM[v]} olarak geçiyor.",
  lambda n,v:f"{n} için son konum {LOC_FORM[v]} kaydedildi.",
  lambda n,v:f"{GENITIVE[n]} şu an bulunduğu mekân {LOC_FORM[v]}.",
  lambda n,v:f"{n}, {LOC_FORM[v]} zaman geçiriyor.",
  lambda n,v:f"{n} {LOC_FORM[v]} olduğu bilgisi verildi.",
 ]
}
QUERY_FORMS={
 'renk':[
  lambda n:f"{GENITIVE[n]} kalemi ne renk?",
  lambda n:f"{GENITIVE[n]} kaleminin rengi nedir?",
  lambda n:f"{n} hangi renk kalem kullanıyor?",
  lambda n:f"{n} için kalemin rengini söyler misin?",
  lambda n:f"{GENITIVE[n]} kalemi hangi renktedir?",
  lambda n:f"{n} adlı kişinin kaleminin rengini açıklar mısın?",
  lambda n:f"Kalemi hakkında {n} için hangi renk yazılmış?",
  lambda n:f"{GENITIVE[n]} kalemi hangi renkte görünüyor?",
  lambda n:f"{n} kaleminin rengi hakkında ne biliniyor?",
  lambda n:f"Kullandığı kalemin rengi {n} için hangisi?",
  lambda n:f"{n} hangi renk ile ilişkilendiriliyor?",
  lambda n:f"{GENITIVE[n]} kullandığı kalemi tarif eder misin?",
 ],
 'konum':[
  lambda n:f"{n} nerede?",
  lambda n:f"{n} şu anda nerede bulunuyor?",
  lambda n:f"{GENITIVE[n]} konumu nedir?",
  lambda n:f"{n} bugün hangi yerde?",
  lambda n:f"{n} şimdi nerede, söyler misin?",
  lambda n:f"{n} adlı kişinin bulunduğu yeri açıklar mısın?",
  lambda n:f"{n} için hangi konum bildirilmiş?",
  lambda n:f"Bulunduğu yer olarak {n} nerede görünüyor?",
  lambda n:f"{n} nerede bulunmuş?",
  lambda n:f"{GENITIVE[n]} bulunduğu yer hakkında bilgi verir misin?",
  lambda n:f"{n} konum olarak hangi yerde?",
  lambda n:f"{n} son olarak nerede görüldü?",
 ]
}
# Style is controlled via explicit user option, but every word is emitted by a
# trained recurrent decoder; these targets are NOT used in decoding/inference.
STYLES=('kisa','aciklayici','detayli')

def reference_text(name, relation, value, style):
    g=GENITIVE[name]
    if relation=='renk':
        sentence=f"{g} kaleminin rengi {COLOR_END[value]}"
        if style=='kisa':return sentence
        if style=='aciklayici':
            return f"Metindeki bilgiye göre {sentence} Soruda sorulan özellik kalemin rengidir. Bu nedenle cevap {value} rengidir."
        return (f"Sorunuzu metinde verilen renk bilgisine dayanarak yanıtlıyorum. "
                f"{sentence} Başka kişilere ait kayıtlar bu sonucu değiştirmiyor. "
                f"Sonuç olarak {name} için belirtilen renk {value} olarak görünüyor.")
    sentence=f"{name} şu anda {LOC_END[value]}"
    if style=='kisa':return sentence
    if style=='aciklayici':
        return f"Verilen açıklamaya göre {sentence} Sorunuz bu kişinin nerede olduğuyla ilgilidir. Bu nedenle belirtilen yer {LOC_FORM[value]} konumudur."
    return (f"Metindeki konum bilgilerini karşılaştırarak yanıtlıyorum. "
            f"{sentence} Diğer kişilerin konumları farklı olabilir. "
            f"Sonuç olarak {name} için kayıtlı yer {LOC_FORM[value]} olarak belirtilmiştir.")

# Byte vocabulary accepts all UTF-8 (0=padding, 1=out-of-domain unk,
# 2..257 byte values). No rule-based morphological analysis.
PAD_BYTE=0

def split_clauses(context):
    # Punctuation-only segmentation, NOT person/relation/value extraction.
    # One factual assertion per sentence is required for the first release.
    segments=[s.strip() for s in re.split(r'(?<=[.!])\s+',context.strip()) if s.strip()]
    if not segments:return []
    return [s if s.endswith('.') else s+'.' for s in segments]

def sample_raw(split,n,seed,holdout_forms=False):
    """Generate distinct fact bindings and diverse Turkish surface realizations."""
    rng=random.Random(seed)
    records=[]
    for k in range(0,n,128):
        x,q,y=generate(split,min(128,n-k),4,rng)
        for facts,query,label in zip(x.tolist(),q.tolist(),y.tolist()):
            raw=[]
            for rec in facts:
                n0,rel,val=(VOCAB[j] for j in rec)
                options=FACT_FORMS[rel]
                slots=(10,11) if holdout_forms else tuple(range(10))
                raw.append(options[rng.choice(slots)](n0,val))
            n0,rel=(VOCAB[j] for j in query)
            opts=QUERY_FORMS[rel]
            slots=(10,11) if holdout_forms else tuple(range(10))
            question=opts[rng.choice(slots)](n0)
            records.append((raw,question,facts,query,label))
    return records

# Target vocabulary includes words in training labels, but inference uses
# their IDs predicted by the decoder. Unseen strings cannot be spoken.
TOKENS=('<pad>','<bos>','<eos>')
def all_target_sentences():
    return [reference_text(n,rel,value,style)
            for n in NAME_LIST for rel in REL_LIST for value in VALUES[rel] for style in STYLES]
WORDS=tuple(dict.fromkeys(TOKENS + tuple(word for s in all_target_sentences() for word in s.split())))
WI={w:i for i,w in enumerate(WORDS)}
MAX_ANSWER_WORDS=max(len(s.split()) for s in all_target_sentences())+1

def encode_answer(text):
    words=text.split()
    ids=[WI[w] for w in words]+[WI['<eos>']]
    target=torch.zeros(MAX_ANSWER_WORDS,dtype=torch.long)
    target[:len(ids)]=torch.tensor(ids)
    return target

class LongTurkishDecoder(nn.Module):
    def __init__(self,hidden=112,word_dim=64):
        super().__init__()
        self.subject=nn.Embedding(V,32)
        self.relation=nn.Embedding(V,16)
        self.style=nn.Embedding(len(STYLES),16)
        self.init=nn.Sequential(nn.Linear(V+32+16+16,hidden),nn.Tanh())
        self.embedding=nn.Embedding(len(WORDS),word_dim,padding_idx=WI['<pad>'])
        self.gru=nn.GRUCell(word_dim,hidden)
        self.head=nn.Linear(hidden,len(WORDS))

    def hidden(self,answer_logits,query,styles):
        cond=torch.cat([answer_logits.softmax(-1),
            self.subject(query[:,0]),self.relation(query[:,1]),self.style(styles)],dim=-1)
        return self.init(cond)

    def forward(self,answer_logits,query,styles,reference):
        h=self.hidden(answer_logits,query,styles)
        prev=torch.full((len(query),),WI['<bos>'],dtype=torch.long,device=query.device)
        y=[]
        for step in range(reference.shape[1]):
            h=self.gru(self.embedding(prev),h)
            y.append(self.head(h))
            prev=reference[:,step]
        return torch.stack(y,dim=1)

    @torch.no_grad()
    def generate(self,answer_logits,query,styles,max_words=None):
        self.eval();h=self.hidden(answer_logits,query,styles)
        prev=torch.full((len(query),),WI['<bos>'],dtype=torch.long,device=query.device)
        finished=[False]*len(query);result=[[] for _ in range(len(query))]
        for _ in range(max_words or MAX_ANSWER_WORDS+4):
            h=self.gru(self.embedding(prev),h)
            prev=self.head(h).argmax(-1)
            for i,p in enumerate(prev.tolist()):
                if finished[i]:continue
                token=WORDS[p]
                if token=='<eos>':finished[i]=True
                elif token not in ('<bos>','<pad>'):result[i].append(token)
            if all(finished):break
        return [' '.join(r) for r in result]


def load_dual(seed=0):
    base=DualAddressUpsertAMN(48)
    ckpt=torch.load(ROOT/f'results/relational_qa/equal50/dual_{seed}.pt',map_location='cpu',weights_only=True)
    base.load_state_dict(ckpt['state_dict'],strict=True)
    base.eval()
    for p in base.parameters():p.requires_grad_(False)
    return base

def prepare_decoder_data(base,split,n,seed):
    rng=random.Random(seed)
    feats,queries,styles,targets,reference=[],[],[],[],[]
    for step in range(0,n,128):
        x,q,y=generate(split,min(128,n-step),4,rng)
        with torch.no_grad():feats.append(base(x,q))
        queries.append(q)
        ss=torch.tensor([rng.randrange(len(STYLES)) for _ in range(len(y))])
        styles.append(ss)
        for a,b,style in zip(q,y,ss):
            name,rel=(VOCAB[int(t)] for t in a)
            sentence=reference_text(name,rel,VOCAB[int(b)],STYLES[int(style)])
            reference.append(sentence);targets.append(encode_answer(sentence))
    return torch.cat(feats),torch.cat(queries),torch.cat(styles),torch.stack(targets),reference

@torch.no_grad()
def decoder_eval(model,data,batch=256):
    actual=[]
    for start in range(0,len(data[0]),batch):
        sl=slice(start,start+batch)
        actual+=model.generate(data[0][sl],data[1][sl],data[2][sl])
    matches=[a==b for a,b in zip(actual,data[4])]
    return {'exact_count':sum(matches),'count':len(matches),'exact_accuracy':sum(matches)/len(matches),'samples':actual[:8]}

def train_decoder(args,outdir,base):
    train=prepare_decoder_data(base,'train',args.decoder_train,4000)
    dev=prepare_decoder_data(base,'dev',args.decoder_dev,5000)
    test=prepare_decoder_data(base,'test',args.decoder_test,6000)
    torch.manual_seed(args.seed+400);m=LongTurkishDecoder()
    opt=torch.optim.Adam(m.parameters(),lr=args.decoder_lr)
    gen=torch.Generator().manual_seed(92000+args.seed)
    best=-1.;best_state=None;best_step=0;history=[];streak=0
    start=time.perf_counter()
    for step in range(1,args.decoder_steps+1):
        ix=torch.randint(len(train[0]),(args.decoder_batch,),generator=gen)
        logits=m(train[0][ix],train[1][ix],train[2][ix],train[3][ix])
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),train[3][ix].reshape(-1),ignore_index=WI['<pad>'])
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        if step%args.decoder_eval_every==0 or step==1 or step==args.decoder_steps:
            ev=decoder_eval(m,dev)
            history.append({'step':step,'loss':float(loss.detach()),'dev_exact':ev['exact_accuracy']})
            if ev['exact_accuracy']>best:
                best=ev['exact_accuracy'];best_step=step
                best_state={k:v.detach().clone() for k,v in m.state_dict().items()}
            streak=(streak+1 if ev['exact_accuracy']==1.0 else 0)
            print('DECODER',step,'dev',round(ev['exact_accuracy'],4),flush=True)
            if streak>=2:break
    elapsed=time.perf_counter()-start
    m.load_state_dict(best_state)
    evaluation=decoder_eval(m,test)
    torch.save({'state_dict':m.state_dict(),'words':WORDS,'styles':STYLES,'seed':args.seed},outdir/'decoder.pt')
    return m,{'training_seconds':elapsed,'best_step':best_step,'dev_best':best,
              'test':evaluation,'history':history,'params':sum(p.numel() for p in m.parameters())}

@torch.no_grad()
def infer(context,question,style,parser,base,decoder,threshold=.67):
    if style not in STYLES:raise ValueError(f'Style must be one of {STYLES}')
    facts,query,confidence=parser.decode([context],[question],threshold=threshold)
    if not any(torch.equal(facts[0,k,:2],query[0]) for k in range(facts.size(1))):
        raise ValueError('Requested person/relation not found in given context')
    logits=base(facts,query)
    text=decoder.generate(logits,query,torch.tensor([STYLES.index(style)]))[0]
    return {'raw_context':context,'raw_question':question,'style':style,'generated':text,
            'confidence_min':confidence[0],
            'predicted_facts':[[VOCAB[int(z)] for z in line] for line in facts[0]],
            'predicted_query':[VOCAB[int(z)] for z in query[0]]}



def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/turkish_free_v1')
    p.add_argument('--decoder-steps',type=int,default=1600)
    p.add_argument('--decoder-eval-every',type=int,default=100)
    p.add_argument('--decoder-batch',type=int,default=64)
    p.add_argument('--decoder-train',type=int,default=15000)
    p.add_argument('--decoder-dev',type=int,default=600)
    p.add_argument('--decoder-test',type=int,default=1200)
    p.add_argument('--decoder-lr',type=float,default=0.003)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(1)
    directory=ROOT/args.output;directory.mkdir(parents=True,exist_ok=True)
    model,report=train_decoder(args,directory,load_dual())
    (directory/'decoder_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('DECODER_RESULT',json.dumps({'test':report['test'],'best_step':report['best_step']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()