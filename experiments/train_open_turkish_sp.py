"""Train independent AMN subword language generator on real Turkish Alpaca QA.

Uses only training text to learn a SentencePiece tokenizer; makes held-out test
measurements only AFTER selection using validation. No Transformer checkpoint.
"""
import argparse,json,math,random,time,hashlib
from pathlib import Path
import sentencepiece as spm
import numpy as np
import torch
from torch.nn import functional as F

from amn_open_turkish_sp import AMNSubwordLM,load_sp,encode,collate,save,restore,IGNORE


def records(folder,part):
    with (Path(folder)/(part+'.jsonl')).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def build_tokenizer(folder,training,words):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    source=folder/'training_sentences.txt'
    with source.open('w',encoding='utf-8') as f:
        for rec in training:
            # Newlines are normalized for SentencePiece text corpus only.
            f.write(('Kullanıcı: '+rec['prompt']+' Asistan: '+rec['answer']).replace('\n',' ')+'\n')
    spm.SentencePieceTrainer.train(input=str(source),model_prefix=str(folder/'tokenizer'),
        vocab_size=words,model_type='bpe',character_coverage=0.9995,
        byte_fallback=True,hard_vocab_limit=False,
        pad_id=0,bos_id=1,eos_id=2,unk_id=3,
        num_threads=2, max_sentence_length=2048)
    source.unlink(missing_ok=True)
    return load_sp(folder/'tokenizer.model')


def encode_split(sp,rows,max_len):
    accepted=[];dropped=0
    for x in rows:
        tok=encode(sp,x['prompt'],x['answer'],max_len=max_len)
        if tok is None:dropped+=1
        else:accepted.append(tok)
    return accepted,dropped


@torch.no_grad()
def validation(model,items,pad,batch=16,max_batches=15):
    model.eval();s=0.;n=0
    for start in range(0,min(len(items),batch*max_batches),batch):
        x,y=collate(items[start:start+batch],pad)
        logits=model(x)
        s+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),
                                 ignore_index=IGNORE,reduction='sum'))
        n+=int(y.ne(IGNORE).sum())
    return {'nll':s/max(1,n),'ppl':math.exp(min(s/max(1,n),30)),'tokens':n}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/open_turkish')
    p.add_argument('--output',default='results/open_turkish_sp')
    p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--width',type=int,default=160)
    p.add_argument('--key-dim',type=int,default=32)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--vocab-size',type=int,default=2500)
    p.add_argument('--max-len',type=int,default=192)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=.0018)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--eval-every',type=int,default=150)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--min-delta',type=float,default=.005)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True)
    train=records(a.data,'train');dev=records(a.data,'val');test=records(a.data,'test')
    corpus=json.loads((Path(a.data)/'manifest.json').read_text(encoding='utf-8'))
    sp=build_tokenizer(folder,train,a.vocab_size)
    t,td=encode_split(sp,train,a.max_len)
    v,vd=encode_split(sp,dev,a.max_len)
    h,hd=encode_split(sp,test,a.max_len)
    if len(t)<1000 or len(v)<80 or len(h)<80:
        raise RuntimeError(f'Not enough subword samples: {len(t),len(v),len(h)}')
    model=AMNSubwordLM(sp.vocab_size(),a.width,a.key_dim,a.layers,sp.pad_id())
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    rng=random.Random(a.seed+500)
    best=float('inf');step_best=0;bad=0;history=[]
    start=time.perf_counter()
    for step in range(1,a.steps+1):
        model.train()
        chosen=[t[rng.randrange(len(t))] for _ in range(a.batch)]
        x,y=collate(chosen,sp.pad_id())
        logits=model(x)
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=IGNORE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step%a.eval_every==0 or step==1 or step==a.steps:
            val=validation(model,v,sp.pad_id())
            elapsed=time.perf_counter()-start
            history.append({'step':step,'train_nll':float(loss.detach()),
                            'val_nll':val['nll'],'val_ppl':val['ppl'],'seconds':elapsed})
            print('STEP',step,'TRAIN',round(float(loss.detach()),4),
                  'VAL',round(val['nll'],4),'PPL',round(val['ppl'],3),
                  'SECONDS',round(elapsed,1),flush=True)
            if val['nll']<best-a.min_delta:
                best=val['nll'];step_best=step;bad=0
                save(folder/'best.pt',model,{
                    'best_step':step,'val_nll':best,'source':corpus,'seed':a.seed,
                    'train_examples':len(t),'dev_examples':len(v),
                    'params':sum(z.numel() for z in model.parameters())})
            else:bad+=1
            if bad>=a.patience:break
    saved,metadata=restore(folder/'best.pt')
    final=validation(saved,h,sp.pad_id(),max_batches=30)
    user_prompts=[
     'Bir insanın öğrenme sürecini açıklar mısın?',
     'Ağ bağlantısında gecikme neden artabilir?',
     'Bir kitabın ana fikri nasıl belirlenir?',
     'Yeni bir yazılım projesinde riskler nasıl azaltılır?',
     'Bir arkadaşına çevreyi korumayı anlat.',
     'Türkçede kelimelere neden ekler getirilir?',
    ]
    samples=[]
    for i,q in enumerate(user_prompts):
        ans=saved.generate(sp,q,max_new_tokens=160,temperature=.7,top_k=20,seed=a.seed+100+i)
        samples.append({'prompt':q,**ans})
        print('OUTPUT',json.dumps(samples[-1],ensure_ascii=False),flush=True)
    for i,row in enumerate(test[:4]):
        ans=saved.generate(sp,row['prompt'],max_new_tokens=160,temperature=.7,top_k=20,seed=a.seed+300+i)
        samples.append({'prompt':row['prompt'],'reference':row['answer'],**ans})
    stats={'architecture':saved.config,'params':sum(z.numel() for z in saved.parameters()),
        'tokenizer_sha256':hashlib.sha256((folder/'tokenizer.model').read_bytes()).hexdigest(),
        'source':corpus,'counts':{'train':len(t),'val':len(v),'test':len(h)},
        'tokenizer_dropped':{'train':td,'val':vd,'test':hd},
        'training_steps':step,'best_step':step_best,'train_seconds':time.perf_counter()-start,
        'validation_nll':best,'test':final,'history':history,'generated':samples,
        'limits':'Subword Turkish AMN from scratch; small model trained on real instruction text, not general Turkish proficiency'}
    (folder/'metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# AMN — Alt sözcük modeliyle gerçek Türkçe çıktı','',
           'Cevaplar hazır şablon değil, modelin serbestçe ürettiği tokenlardır.','',
           f'Parametre: {stats["params"]}; test NLL: {final["nll"]:.4f}; PPL: {final["ppl"]:.2f}.','']
    for i,s in enumerate(samples,1):
        lines += [f'## {i}. soru','**Soru:** '+s['prompt'],'','**Model cevabı:** '+(s['text'] or '(boş)'),
                  f'**Üretilen token:** {s["tokens_generated"]}; EOS: {s["eos"]}','']
    (folder/'generated_turkish.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
