"""Token-supervised neural semantic extraction; words parsed, meanings learned.

The input is raw Turkish text. A simple Unicode tokenizer identifies word
boundaries only; a neural token tagger learns person/value spans and relation
class from supervised training records. No regex extracts semantic fields at
inference. Scope is still the finite training domain; not open-domain Turkish.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from relational_qa import VOCAB, ID, VALUES
from free_turkish_core import (
    ROOT, NAME_LIST, VALUE_LIST, REL_LIST, STYLES, GENITIVE, LOC_FORM,
    sample_raw, split_clauses, load_dual, LongTurkishDecoder, infer
)
# Word boundaries only; no hand-written subject/value/relation parsing.
MAX_WORDS=24
WORD_PATTERN=re.compile(r"[^\W_]+(?:['’][^\W_]+)?",re.UNICODE)
def tokenize(sentence):
    return [s.casefold() for s in WORD_PATTERN.findall(sentence)]
def tensorize(sentences,vocab):
    result=torch.zeros(len(sentences),MAX_WORDS,dtype=torch.long)
    for i,text in enumerate(sentences):
        words=[vocab.get(w,1) for w in tokenize(text)][:MAX_WORDS]
        if words:result[i,:len(words)]=torch.tensor(words)
    return result
def build_vocab(records):
    vocab={'<pad>':0,'<unk>':1}
    for lines,question,*_ in records:
        for text in list(lines)+[question]:
            for tok in tokenize(text):
                if tok not in vocab:vocab[tok]=len(vocab)
    return vocab


def make_tags(sentence, name, relation, value=None):
    tokens=tokenize(sentence)[:MAX_WORDS]
    n_l=name.casefold()
    gen_l=GENITIVE[name].casefold()
    value_str=(LOC_FORM[value] if relation=='konum' else value).casefold() if value else None
    names=[0]*MAX_WORDS;values=[0]*MAX_WORDS
    for i,word in enumerate(tokens):
        if word==n_l or word==gen_l:
            names[i]=NAME_LIST.index(name)+1
        if value_str is not None and word==value_str:
            values[i]=VALUE_LIST.index(value)+1
    if not any(names):
        raise ValueError(f'Cannot locate label name token in: {sentence}')
    if value and not any(values):
        raise ValueError(f'Cannot locate label value token in: {sentence}')
    return names,values


def build_data(records,vocab):
    n=len(records)
    fact_strings=[text for rec in records for text in rec[0]]
    words=tensorize(fact_strings,vocab).reshape(n,4,MAX_WORDS)
    questions=tensorize([rec[1] for rec in records],vocab)
    n_tags=[];v_tags=[];rel_tags=[];q_tags=[];qrel=[]
    for lines,question,triples,query,answer in records:
        for raw,trip in zip(lines,triples):
            name,rel,value=(VOCAB[int(x)] for x in trip)
            name_t,value_t=make_tags(raw,name,rel,value)
            n_tags.append(name_t);v_tags.append(value_t)
            rel_tags.append(int(rel=='konum'))
        qname,qrelation=(VOCAB[int(x)] for x in query)
        qname_t,_=make_tags(question,qname,qrelation)
        q_tags.append(qname_t);qrel.append(int(qrelation=='konum'))
    labs=(torch.tensor(n_tags).reshape(n,4,MAX_WORDS),
          torch.tensor(v_tags).reshape(n,4,MAX_WORDS),
          torch.tensor(rel_tags).reshape(n,4),
          torch.tensor(q_tags).reshape(n,MAX_WORDS),
          torch.tensor(qrel))
    return words,questions,labs

class SpanConceptNet(nn.Module):
    def __init__(self,vocab_size,d=96):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,d,padding_idx=0)
        self.transform=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.LayerNorm(d))
        self.subject=nn.Linear(d,len(NAME_LIST)+1)
        self.value=nn.Linear(d,len(VALUE_LIST)+1)
        self.relation=nn.Linear(d,2)

    def token_logits(self,x):
        h=self.transform(self.emb(x))
        return self.subject(h),self.value(h),self.relation(h)

    @staticmethod
    def tag_to_class(scores,mask):
        # Score a concept by log P(concept) minus log P(background), maximized
        # across surface tokens. This is a learned span classifier.
        class_scores=scores[...,1:]-scores[...,0:1]
        class_scores=class_scores.masked_fill(~mask.unsqueeze(-1),-1e4)
        return class_scores.amax(-2)

    def forward(self,x,question=False):
        subj,val,rel=self.token_logits(x)
        mask=x.gt(1)
        subj_scores=self.tag_to_class(subj,mask)
        rel_scores=rel.masked_fill(~mask.unsqueeze(-1),-1e4).amax(-2)
        if question:return subj_scores,rel_scores
        val_scores=self.tag_to_class(val,mask)
        return subj_scores,rel_scores,val_scores

class TaggedNeuralParser(nn.Module):
    def __init__(self,vocab):
        super().__init__();self.vocab=dict(vocab)
        self.net=SpanConceptNet(len(vocab))

    def forward(self,facts,questions):
        b,n,l=facts.shape
        f=self.net(facts.reshape(b*n,l))
        # Constraint of the finite training ontology: all observed colour
        # values and all observed location values occupy disjoint label sets.
        # This keeps relation and learned value prediction consistent.
        value_scores=f[2]
        group_relation=torch.stack((value_scores[:,:8].amax(-1),
                                   value_scores[:,8:].amax(-1)),dim=-1)
        f=(f[0],group_relation,value_scores)
        q=self.net(questions,question=True)
        return tuple(v.reshape(b,n,-1) for v in f),q

    @torch.no_grad()
    def decode(self,contexts,questions,max_facts=8,threshold=.55):
        self.eval()
        if len(contexts)!=len(questions):raise ValueError('Mismatched raw inputs')
        lines=[split_clauses(t) for t in contexts]
        if not lines or any(not (1<=len(x)<=max_facts) for x in lines):
            raise ValueError('Expected 1..8 complete statements')
        if len(set(map(len,lines)))!=1:raise ValueError('Batch sentence count mismatch')
        n=len(lines[0]);b=len(lines)
        f=tensorize([s for row in lines for s in row],self.vocab).reshape(b,n,MAX_WORDS)
        q=tensorize(questions,self.vocab)
        (name,rel,value),(qname,qrel)=self(f,q)
        pred=[x.argmax(-1) for x in (name,rel,value)]
        scores=[x.softmax(-1).max(-1).values for x in (name,rel,value)]
        qpred=[x.argmax(-1) for x in (qname,qrel)]
        qscore=[x.softmax(-1).max(-1).values for x in (qname,qrel)]
        triples=[];queries=[];confs=[]
        for i in range(b):
            n0,r0=(int(x[i]) for x in qpred)
            expected_name=NAME_LIST[n0];expected_rel=REL_LIST[r0]
            qt=questions[i].casefold()
            if expected_name.casefold() not in qt:raise ValueError('Unknown person / question')
            if r0==0 and not any(s in qt for s in ('kalem','renk')):
                raise ValueError('Unsupported question type')
            if r0==1 and not any(s in qt for s in ('nerede','konum','yerde','yer ','bulunduğu','mekân')):
                raise ValueError('Unsupported question type')
            fs=[];conf=[float(x[i]) for x in qscore]
            for j,line in enumerate(lines[i]):
                a,r,v=(int(x[i,j]) for x in pred)
                person,relation,word=NAME_LIST[a],REL_LIST[r],VALUE_LIST[v]
                if word not in VALUES[relation]:raise ValueError('Inconsistent relation and value')
                if person.casefold() not in line.casefold() or word.casefold() not in line.casefold():
                    raise ValueError('Learned fields not supported by text')
                fs.append([ID[person],ID[relation],ID[word]])
                conf.extend(float(x[i,j]) for x in scores)
            if min(conf)<threshold:raise ValueError(f'Low confidence {min(conf):.3f}')
            triples.append(fs);queries.append([ID[expected_name],ID[expected_rel]]);confs.append(min(conf))
        return torch.tensor(triples),torch.tensor(queries),confs

def loss(model,inputs):
    fact,q,labs=inputs
    n,v,r,qn,qr=labs
    x=fact.reshape(-1,MAX_WORDS)
    name,val,rel=model.net.token_logits(x)
    qname,_,qrelation=model.net.token_logits(q)
    return (F.cross_entropy(name.reshape(-1,name.size(-1)),n.reshape(-1))+
            F.cross_entropy(val.reshape(-1,val.size(-1)),v.reshape(-1))+
            F.cross_entropy(qname.reshape(-1,qname.size(-1)),qn.reshape(-1))+
            F.cross_entropy(model.net(fact.reshape(-1,MAX_WORDS))[1],r.reshape(-1))+
            F.cross_entropy(model.net(q,question=True)[1],qr))

@torch.no_grad()
def evaluate(model,data,batch=128):
    model.eval();facts,questions,targ=data
    fields=0;fully=0
    for start in range(0,len(facts),batch):
        sl=slice(start,start+batch)
        a,b=model(facts[sl],questions[sl]);out=list(a)+list(b)
        tag1,tag2,rel,tagq,qr=[z[sl] for z in targ]
        # Label from highest-confidence token span must match actual subject/value.
        nlabel=tag1.amax(-1)-1
        vlabel=tag2.amax(-1)-1
        qlabel=tagq.amax(-1)-1
        truth=[nlabel,rel,vlabel,qlabel,qr]
        c=[p.argmax(-1).eq(z) for p,z in zip(out,truth)]
        fully+=int((c[0].all(-1)&c[1].all(-1)&c[2].all(-1)&c[3]&c[4]).sum())
        fields+=sum(int(z.sum()) for z in c)
    return {'episodes_correct':fully,'total':len(facts),'accuracy':fully/len(facts),
            'field_accuracy':fields/(len(facts)*14)}

def train(args):
    torch.set_num_threads(1)
    folder=ROOT/args.output;folder.mkdir(parents=True,exist_ok=True)
    raw=sample_raw('train',args.train_size,1111)
    vocab=build_vocab(raw)
    train=build_data(raw,vocab)
    dev=build_data(sample_raw('dev',args.dev_size,2222),vocab)
    test=build_data(sample_raw('test',args.test_size,3333),vocab)
    unseen=build_data(sample_raw('test',args.test_size,4444,holdout_forms=True),vocab)
    torch.manual_seed(args.seed)
    model=TaggedNeuralParser(vocab)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    gen=torch.Generator().manual_seed(5555+args.seed)
    best=-1.;best_state=None;best_step=0;history=[]
    for step in range(1,args.steps+1):
        model.train();ix=torch.randint(len(train[0]),(args.batch,),generator=gen)
        x=(train[0][ix],train[1][ix],tuple(y[ix] for y in train[2]))
        l=loss(model,x)
        opt.zero_grad(set_to_none=True);l.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%args.eval_every==0 or step==1 or step==args.steps:
            score=evaluate(model,dev)
            print('TAGGED',step,'dev',round(score['accuracy'],4),'fields',round(score['field_accuracy'],4),flush=True)
            history.append({'step':step,'dev':score['accuracy'],'training_loss':float(l.detach())})
            if score['accuracy']>best:
                best=score['accuracy'];best_state={k:v.detach().clone() for k,v in model.state_dict().items()};best_step=step
            if best==1. and step>=args.min_steps:break
    model.load_state_dict(best_state)
    metrics={'dev_best':best,'best_step':best_step,'word_vocab_size':len(vocab),
             'known_forms':evaluate(model,test),'unseen_forms':evaluate(model,unseen),'history':history}
    torch.save({'state_dict':model.state_dict(),'vocab':vocab},folder/'parser.pt')
    (folder/'parser_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
    print('FINAL_TAG',json.dumps({k:v for k,v in metrics.items() if k!='history'},ensure_ascii=False),flush=True)
    return metrics

def load_pipeline(folder):
    saved=torch.load(folder/'parser.pt',map_location='cpu',weights_only=True)
    p=TaggedNeuralParser(saved['vocab']);p.load_state_dict(saved['state_dict']);p.eval()
    m=LongTurkishDecoder()
    saved=torch.load(folder/'decoder.pt',map_location='cpu',weights_only=True)
    m.load_state_dict(saved['state_dict']);m.eval()
    return p,load_dual(),m

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('train','ask','demo'))
    p.add_argument('--output',default='results/free_turkish_tagged')
    p.add_argument('--decoder-from',default='results/free_turkish_v1/decoder.pt')
    p.add_argument('--train-size',type=int,default=12000)
    p.add_argument('--dev-size',type=int,default=800)
    p.add_argument('--test-size',type=int,default=1200)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--min-steps',type=int,default=150)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--context',default='')
    p.add_argument('--question',default='')
    p.add_argument('--style',default='detayli',choices=STYLES)
    p.add_argument('--threshold',type=float,default=.55)
    args=p.parse_args()
    folder=ROOT/args.output;folder.mkdir(parents=True,exist_ok=True)
    if args.command=='train':
        from shutil import copy2
        copy2(ROOT/args.decoder_from,folder/'decoder.pt')
        train(args)
    else:
        p,b,m=load_pipeline(folder)
        if args.command=='ask':
            r=infer(args.context,args.question,args.style,p,b,m,args.threshold)
            print(json.dumps(r,ensure_ascii=False,indent=2))
        else:
            for raw,q,*_ in sample_raw('test',8,77):
                try:print(infer(' '.join(raw),q,'detayli',p,b,m,args.threshold)['generated'])
                except ValueError as exc:print('REJECT',exc)

if __name__=='__main__':main()