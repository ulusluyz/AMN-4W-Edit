"""Learned autoregressive Turkish WORD decoder on top of frozen AMN-4W-Edit QA encoders.

The decoder generates every output word itself with greedy decoding from <bos>
to <eos>. During TRAINING, a restricted grammar supplies supervised Turkish
sentences. Structured triples are the input, NOT raw Turkish prose.
No pretrained encoder weights are changed. This is NOT a general Turkish LLM.

    python experiments/turkish_sentence_decoder.py --models dual_upsert amn transformer --seeds 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from relational_qa import VOCAB, NAMES, VALUES, FACTS, V, generate, AssociativeAMN, TransformerQA
from dual_upsert import DualAddressUpsertAMN

ROOT = Path(__file__).resolve().parents[1]
GENITIVE = {
    'Ali': "Ali'nin", 'Ayşe': "Ayşe'nin", 'Can': "Can'ın", 'Deniz': "Deniz'in",
    'Ece': "Ece'nin", 'Efe': "Efe'nin", 'Elif': "Elif'in", 'Emre': "Emre'nin",
    'Mert': "Mert'in", 'Mina': "Mina'nın", 'Murat': "Murat'ın", 'Nisa': "Nisa'nın",
    'Okan': "Okan'ın", 'Selin': "Selin'in", 'Sena': "Sena'nın", 'Umut': "Umut'un",
    'Yaren': "Yaren'in", 'Yiğit': "Yiğit'in", 'Zeynep': "Zeynep'in", 'Arda': "Arda'nın",
}
COLOR = {'kırmızı':'kırmızıdır.','mavi':'mavidir.','yeşil':'yeşildir.',
         'sarı':'sarıdır.','mor':'mordur.','beyaz':'beyazdır.',
         'siyah':'siyahtır.','turuncu':'turuncudur.'}
LOCATION = {'ev':'evdedir.','okul':'okuldadır.','park':'parktadır.',
            'ofis':'ofistedir.','bahçe':'bahçededir.','kafe':'kafededir.',
            'salon':'salondadır.','garaj':'garajdadır.'}
assert len(GENITIVE)==len(NAMES)
assert set(COLOR)==set(VALUES['renk']) and set(LOCATION)==set(VALUES['konum'])
WORDS = tuple(dict.fromkeys(('<pad>','<bos>','<eos>') + NAMES +
        tuple(GENITIVE.values()) + ('kaleminin','rengi','şu','anda') +
        tuple(COLOR.values()) + tuple(LOCATION.values())))
WORD_ID = {word:i for i,word in enumerate(WORDS)}

def reference_sentence(query, answer):
    """TRAINING and SCORING reference only; not used at generation time."""
    name, relation = (VOCAB[int(v)] for v in query)
    value=VOCAB[int(answer)]
    if relation=='renk':
        words=(GENITIVE[name],'kaleminin','rengi',COLOR[value])
    elif relation=='konum':
        words=(name,'şu','anda',LOCATION[value])
    else: raise ValueError('unknown relation')
    return ' '.join(words), torch.tensor([WORD_ID[w] for w in words]+[WORD_ID['<eos>']])

class SentenceDecoder(nn.Module):
    """GRU generates a complete 4-word Turkish sentence and EOS."""
    def __init__(self,hidden=96,word_dim=64):
        super().__init__()
        self.person=nn.Embedding(V,hidden//2)
        self.relation=nn.Embedding(V,hidden//2)
        self.init=nn.Sequential(nn.Linear(V+hidden,hidden),nn.Tanh())
        self.words=nn.Embedding(len(WORDS),word_dim)
        self.gru=nn.GRUCell(word_dim,hidden)
        self.output=nn.Linear(hidden,len(WORDS))
    def condition(self, base_logits, query):
        return self.init(torch.cat([base_logits.softmax(-1),
               self.person(query[:,0]),self.relation(query[:,1])],-1))
    def forward(self,base_logits,query,gold):
        state=self.condition(base_logits,query)
        prev=torch.full((len(query),),WORD_ID['<bos>'],dtype=torch.long,device=query.device)
        outputs=[]
        for i in range(gold.shape[1]):
            state=self.gru(self.words(prev),state)
            outputs.append(self.output(state))
            prev=gold[:,i]  # Training teacher forcing, not inference.
        return torch.stack(outputs,1)
    @torch.no_grad()
    def generate(self,base_logits,query,max_words=8):
        self.eval();state=self.condition(base_logits,query)
        prev=torch.full((len(query),),WORD_ID['<bos>'],dtype=torch.long,device=query.device)
        collected=[[] for _ in range(len(query))]
        finished=[False for _ in range(len(query))]
        for _ in range(max_words):
            state=self.gru(self.words(prev),state)
            prev=self.output(state).argmax(-1)
            for i,v in enumerate(prev.tolist()):
                if finished[i]:continue
                word=WORDS[v]
                if word=='<eos>':finished[i]=True
                elif word not in ('<bos>','<pad>'):collected[i].append(word)
            if all(finished):break
        return [' '.join(parts) for parts in collected]

def load_frozen(kind,seed):
    if kind=='dual_upsert':
        model=DualAddressUpsertAMN(48);src='dual'
    elif kind=='amn':
        model=AssociativeAMN(48);src='amn'
    else:
        model=TransformerQA(48,4);src='transformer'
    path=ROOT/'results'/'relational_qa'/'equal50'/f'{src}_{seed}.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

@torch.no_grad()
def make_data(base,split,n,seed):
    rng=random.Random(seed);features=[];queries=[];gold=[];texts=[];records=[]
    for start in range(0,n,256):
        x,q,y=generate(split,min(256,n-start),4,rng)
        features.append(base(x,q));queries.append(q)
        for a,b in zip(q,y):
            s,t=reference_sentence(a,b)
            texts.append(s);gold.append(t)
        if len(records)<12:
            for i in range(min(12-len(records),len(y))):
                records.append({'facts':[[VOCAB[int(c)] for c in trip] for trip in x[i]],
                                'query':[VOCAB[int(w)] for w in q[i]]})
    return torch.cat(features),torch.cat(queries),torch.stack(gold),texts,records

def one_run(kind,seed,args):
    torch.manual_seed(seed+527);random.seed(seed+527)
    base=load_frozen(kind,seed)
    train=make_data(base,'train',args.train_size,seed+10000)
    dev=make_data(base,'dev',args.dev_size,seed+20000)
    test=make_data(base,'test',args.test_size,seed+30000)
    model=SentenceDecoder()
    optim=torch.optim.Adam(model.parameters(),lr=args.lr)
    rng=torch.Generator().manual_seed(seed+40000)
    best=-1.;best_weights=None;best_step=0;streak=0;history=[]
    start=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train()
        ix=torch.randint(args.train_size,(args.batch,),generator=rng)
        prediction=model(train[0][ix],train[1][ix],train[2][ix])
        loss=F.cross_entropy(prediction.reshape(-1,prediction.size(-1)),
                             train[2][ix].reshape(-1))
        optim.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            actual=model.generate(dev[0],dev[1])
            accuracy=sum(x==y for x,y in zip(actual,dev[3]))/len(actual)
            elapsed=time.perf_counter()-start
            history.append({'step':step,'dev_sentence_exact':accuracy,
                            'loss':float(loss.detach()),'decoder_train_seconds':round(elapsed,3)})
            if accuracy>best:
                best=accuracy;best_step=step
                best_weights={k:v.clone() for k,v in model.state_dict().items()}
            streak=streak+1 if accuracy==1.0 else 0
            print('PROGRESS',kind,seed,step,round(accuracy,4),round(elapsed,2),flush=True)
            if streak==2:break
    elapsed=time.perf_counter()-start
    model.load_state_dict(best_weights)
    actual=model.generate(test[0],test[1])
    exact=sum(x==y for x,y in zip(actual,test[3]))
    examples=[{**test[4][i],'expected':test[3][i],
               'generated':actual[i],'exact':actual[i]==test[3][i]}
              for i in range(min(12,len(actual)))]
    result={'architecture':kind,'seed':seed,'frozen_backbone':True,
            'backbone_original_training_steps':50,
            'decoder_training_steps_used':step,'decoder_best_step':best_step,
            'decoder_train_seconds':round(elapsed,3),
            'decoder_params':sum(p.numel() for p in model.parameters()),
            'best_dev_sentence_accuracy':best,
            'test_correct_complete_sentences':exact,'test_sentence_count':len(actual),
            'test_accuracy':exact/len(actual),'examples':examples,'history':history,
            'limits':{'train_unique_triples':len(FACTS['train']),
                      'dev_unique_triples':len(FACTS['dev']),
                      'test_unique_triples':len(FACTS['test']),
                      'train_sampled_contexts':args.train_size,
                      'dev_sampled_contexts':args.dev_size,
                      'test_sampled_contexts':args.test_size,
                      'model_input':'structured person/relation/value triples',
                      'training_references':'limited templated Turkish word sequences',
                      'model_output':'word-by-word autoregressive neural generation',
                      'unsupported':'raw free-form Turkish input / unconstrained writing'}}
    return result,model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models',nargs='+',choices=['dual_upsert','amn','transformer'],
                   default=['dual_upsert','amn','transformer'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--steps',type=int,default=650)
    p.add_argument('--train-size',type=int,default=8192)
    p.add_argument('--dev-size',type=int,default=640)
    p.add_argument('--test-size',type=int,default=1200)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--lr',type=float,default=0.003)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--output',default='results/real_turkish_sentences')
    args=p.parse_args()
    torch.set_num_threads(1)
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    runs=[]
    for seed in args.seeds:
        for kind in args.models:
            metrics,model=one_run(kind,seed,args)
            (out/f'{kind}_{seed}.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
            torch.save({'state_dict':model.state_dict(),'word_vocab':WORDS,
                        'seed':seed,'architecture':kind},out/f'{kind}_{seed}.pt')
            runs.append(metrics)
            print('RESULT',kind,seed,metrics['test_correct_complete_sentences'],
                  '/',metrics['test_sentence_count'],flush=True)
    lines=['# Türkçe tam cümle: Modellerin gerçekten ürettiği metinler','',
           'Çıktılar yeni eğitilen ortak GRU çözücülerinin **kelime kelime ürettiği tahminlerdir**.',
           'Cevap sırasında cümle birleştiren şablon kullanılmadı. Eğitim hedefleri sınırlı Türkçe şablonlarla oluşturuldu.',
           '**Girdi serbest Türkçe değil, yapılandırılmış olgu üçlüleridir.**','',
           '| Model | Doğru tam cümle | Test | Öğrenilen decoder adımı |',
           '|---|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['architecture']} (seed {r['seed']}) | {r['test_correct_complete_sentences']} | {r['test_sentence_count']} | {r['decoder_training_steps_used']} |")
    for i in range(min(8,len(runs[0]['examples']))):
        lines += ['',f'## Örnek {i+1}',
                  'Olgu: '+', '.join('/'.join(x) for x in runs[0]['examples'][i]['facts']),
                  'Soru: '+ ' / '.join(runs[0]['examples'][i]['query']),
                  '**Beklenen cümle:** '+runs[0]['examples'][i]['expected']]
        for r in runs:
            x=r['examples'][i]
            lines.append(f"- **{r['architecture']} seed {r['seed']}:** {x['generated']} ({'doğru' if x['exact'] else 'yanlış'})")
    (out/'model_generated_sentences_tr.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
