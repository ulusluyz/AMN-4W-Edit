"""Train independent AMN text-to-text model on real Turkish instructions.

This is training from scratch, not fake training via a pretrained Transformer.
Only response bytes are supervised. All parameters stay local on the runner;
original AMN-4W weights are read-only and untouched.

Examples (repo root):
  python experiments/train_open_turkish.py --steps 1200 --output results/open_turkish
"""
from __future__ import annotations
import argparse, json, math, random, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from amn_open_turkish import (
 AMNOpenTurkish,make_training_example,collate,load_checkpoint,save_checkpoint,IGNORE
)

ROOT=Path(__file__).resolve().parents[1]


def read_split(folder,part,max_len):
    path=Path(folder)/(part+'.jsonl')
    with path.open(encoding='utf-8') as f:
        rows=[json.loads(x) for x in f if x.strip()]
    items=[]
    for r in rows:
        value=make_training_example(r['prompt'],r['answer'],max_len)
        if value is None:raise ValueError(f'Dataset preprocessing mismatch: {part}')
        items.append(value)
    return items,rows


def compute_loss(logits,y):
    return F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=IGNORE)


@torch.no_grad()
def evaluate(model,examples,batch=12,max_batches=16):
    model.eval();total_loss=0.;total_tokens=0
    for start in range(0,min(len(examples),batch*max_batches),batch):
        x,y=collate(examples[start:start+batch])
        l=model(x)
        num=int(y.ne(IGNORE).sum())
        if not num:continue
        total_loss+=float(F.cross_entropy(l.reshape(-1,l.size(-1)),y.reshape(-1),ignore_index=IGNORE,reduction='sum'))
        total_tokens+=num
    if not total_tokens:return {'nll':None,'ppl':None,'tokens':0}
    nll=total_loss/total_tokens
    return {'nll':nll,'ppl':math.exp(min(nll,30)),'tokens':total_tokens}


def run(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    train,train_rows=read_split(args.data,'train',args.max_len)
    val,val_rows=read_split(args.data,'val',args.max_len)
    test,test_rows=read_split(args.data,'test',args.max_len)
    manifest=json.loads((Path(args.data)/'manifest.json').read_text(encoding='utf-8'))
    model=AMNOpenTurkish(width=args.width,key_dim=args.key_dim,layers=args.layers)
    params=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    chooser=random.Random(args.seed+777)
    best=float('inf');best_step=0;bad_checks=0;history=[]
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train()
        rows=[train[chooser.randrange(len(train))] for _ in range(args.batch)]
        x,y=collate(rows)
        logits=model(x)
        loss=compute_loss(logits,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step%args.eval_every==0 or step==1 or step==args.steps:
            score=evaluate(model,val,args.eval_batch,args.eval_batches)
            elapsed=time.perf_counter()-started
            history.append({'step':step,'train_nll':float(loss.detach()),'val_nll':score['nll'],
                            'val_ppl':score['ppl'],'seconds':elapsed})
            print('STEP',step,'TRAIN',round(float(loss.detach()),4),
                  'VAL',round(score['nll'],4),'PPL',round(score['ppl'],2),
                  'ELAPSED',round(elapsed,1),flush=True)
            if score['nll'] < best-args.min_delta:
                best=score['nll'];best_step=step;bad_checks=0
                save_checkpoint(folder/'best.pt',model,extra={
                    'step':step,'val_nll':best,'data':manifest,'model_params':params,
                    'seed':args.seed,'train_s':elapsed})
            else:
                bad_checks+=1
            if bad_checks>=args.patience:break
    best_model,metadata=load_checkpoint(folder/'best.pt')
    final_eval=evaluate(best_model,test,args.eval_batch,args.test_batches)
    probes=[
        "Bir insanın öğrenme sürecini açıklayan kısa bir paragraf yazar mısın?",
        "Bir ağ bağlantısındaki gecikme neden artabilir?",
        "Bir kitabın ana fikri nasıl belirlenir?",
        "Yeni bir proje geliştirirken karşılaşılabilecek sorunları açıklar mısın?",
        "Bir arkadaşına çevreyi korumanın önemini anlat.",
        "Türkçede neden kelimelerin sonuna ekler getirilir?",
    ]
    samples=[]
    for i,prompt in enumerate(probes):
        ans=best_model.complete(prompt,max_new_bytes=args.sample_bytes,
                                temperature=args.temperature,top_k=args.top_k,seed=args.seed+10+i)
        samples.append({'prompt':prompt,**ans})
        print('GENERATED',json.dumps(samples[-1],ensure_ascii=False),flush=True)
    # heldout examples used only for inference after model selection
    for i,row in enumerate(test_rows[:4]):
        ans=best_model.complete(row['prompt'],max_new_bytes=args.sample_bytes,
                                temperature=args.temperature,top_k=args.top_k,seed=args.seed+100+i)
        samples.append({'prompt':row['prompt'], 'test_reference':row['answer'],**ans})
    result={'dataset':manifest,'train_examples':len(train),'dev_examples':len(val),
            'test_examples':len(test),'model_params':params,'architecture':model.cfg,
            'best_step':best_step,'steps_executed':step,'train_seconds':time.perf_counter()-started,
            'best_val_nll':best,'test':final_eval,'history':history,
            'generation':samples,
            'caveat':'Open-domain generative experiment; no general Turkish ability guarantee; no pretrained LLM'}
    (folder/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Gerçek Türkçe veride bağımsız AMN dil modeli üretimi','',
           'Bu metinler modelin bayt bayt ürettiği çıktılardır; cevap şablonu kullanılmadı.',
           'Model Transformer kullanmaz. Eğitim: Türkçe Alpaca (Apache-2.0), cevap token kaybı.',
           '',f'En iyi adım: {best_step}; parametre: {params}; doğrulama NLL: {best:.4f}; '
           f'test NLL: {final_eval["nll"]:.4f} ve test PPL: {final_eval["ppl"]:.2f}.','']
    for i,r in enumerate(samples,1):
        lines += [f'## Soru {i}',r['prompt'],'','**Model yanıtı:**',r['text'] or '(boş yanıt)',
                  '',f'UTF-8 geçerli: {r["valid_utf8"]}; EOS: {r["eos"]}','']
    (folder/'generated_turkish.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/open_turkish')
    p.add_argument('--output',default='results/open_turkish')
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--width',type=int,default=96)
    p.add_argument('--key-dim',type=int,default=24)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--batch',type=int,default=12)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--max-len',type=int,default=384)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--eval-batch',type=int,default=8)
    p.add_argument('--eval-batches',type=int,default=14)
    p.add_argument('--test-batches',type=int,default=24)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--min-delta',type=float,default=.01)
    p.add_argument('--sample-bytes',type=int,default=300)
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--top-k',type=int,default=12)
    run(p.parse_args())

if __name__=='__main__':main()
