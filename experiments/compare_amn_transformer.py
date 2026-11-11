"""Same sampled A->B->C data/optimizer/steps for AMN proxy and Transformer.
Not a comparison with the full AMN-4W v0.4 deployed model.
Generate datasets independently of the training RNG so both see identical cases.
"""
import argparse, json, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from amn4w_local_test import AMN4W, norm

torch.set_num_threads(1)

class TransformerBaseline(nn.Module):
    def __init__(self,V,M,d=64):
        super().__init__()
        self.embedding=nn.Embedding(V,d,padding_idx=0)
        self.position=nn.Embedding(3*M+1,d)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=128,dropout=0.0,batch_first=True,activation='gelu',norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        self.fc=nn.Linear(d,V)
    def forward(self,ctx,a):
        x=torch.cat([ctx,a[:,None]],dim=1)
        pos=self.position(torch.arange(x.shape[1],device=x.device))
        x=self.embedding(x)+pos
        return self.fc(self.encoder(x)[:,-1])

def generate(bs,V,M,rng):
    ctx=[];a=[];b=[];c=[];targets=[]
    for _ in range(bs):
        ids=rng.sample(range(2,V),3*M)
        j=rng.randrange(M)
        ctx.append(ids);a.append(ids[3*j]);b.append(ids[3*j+1]);c.append(ids[3*j+2]);targets.append(j)
    return [torch.tensor(z,dtype=torch.long) for z in (ctx,a,b,c,targets)]

def data_stream(seed,steps,bs,V,M):
    rng=random.Random(100000+seed)
    return [generate(bs,V,M,rng) for _ in range(steps)]

@torch.no_grad()
def evaluate(model,arch,dataset,V,M):
    model.eval()
    num=0; correct=0; unrelated_corr=0; target_corr=0
    p_true=[];p_unrelated=[];p_target=[]
    for ctx,a,b,c,j in dataset:
        B=ctx.shape[0];mask=ctx.new_tensor([0,1,1]*M).float().expand(B,-1)
        def logits(cctx,cmask):
            return model(cctx,a) if arch=='transformer' else model.query(model(cctx,cmask),a,3*M)[0]
        L=logits(ctx,mask)
        # Change one irrelevant chain, leave target chain and query intact.
        # Corruption is sampled independently; tokens are replaced by 0 (out of training support).
        unrelated=(j+1)%M
        corrupt_ctx=ctx.clone(); corrupt_mask=mask.clone()
        inds=unrelated[:,None]*3+torch.arange(3)[None,:]
        corrupt_ctx.scatter_(1,inds,0)
        corrupt_mask.scatter_(1,inds,0.)
        Lu=logits(corrupt_ctx,corrupt_mask)
        target_ctx=ctx.clone();target_mask=mask.clone()
        target_inds=j[:,None]*3+torch.arange(3)[None,:]
        target_ctx.scatter_(1,target_inds,0)
        target_mask.scatter_(1,target_inds,0.)
        Lt=logits(target_ctx,target_mask)
        num+=B;correct+=(L.argmax(-1)==c).sum().item();unrelated_corr+=(Lu.argmax(-1)==c).sum().item();target_corr+=(Lt.argmax(-1)==c).sum().item()
        for dest,logit in ((p_true,L),(p_unrelated,Lu),(p_target,Lt)):
            dest.extend(F.softmax(logit,dim=-1).gather(1,c[:,None]).squeeze(1).tolist())
    return dict(acc=correct/num,acc_unrelated_removed=unrelated_corr/num,acc_target_removed=target_corr/num,
                p_gold=sum(p_true)/num,p_gold_unrelated_removed=sum(p_unrelated)/num,p_gold_target_removed=sum(p_target)/num,
                unrelated_probability_abs_change=sum(abs(x-y) for x,y in zip(p_true,p_unrelated))/num,
                target_probability_abs_change=sum(abs(x-y) for x,y in zip(p_true,p_target))/num)

def run(seed,steps,V,M,bs,eval_size,arch,aux):
    torch.manual_seed(seed)
    model=(AMN4W(V,64,M) if arch=='amn' else TransformerBaseline(V,M))
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    train_set=data_stream(seed,steps,bs,V,M)
    eval_set=[generate(eval_size,V,M,random.Random(200000+seed))]  # shared for both architectures
    t=time.perf_counter();model.train();lossval=0
    for ctx,a,b,c,j in train_set:
        if arch=='amn':
            mask=ctx.new_tensor([0,1,1]*M).float().expand(ctx.shape[0],-1)
            S=model(ctx,mask); logits,(q1,r1,q2,r2)=model.query(S,a,3*M)
        else: logits=model(ctx,a)
        loss=F.cross_entropy(logits,c)
        if arch=='amn' and aux:
            with torch.no_grad():
                ka,kb=norm(model.fc_k(model.embedding(a))),norm(model.fc_k(model.embedding(b)))
                vb,vc=norm(model.fc_v(model.embedding(b))),norm(model.fc_v(model.embedding(c)))
            loss+=sum(1-F.cosine_similarity(x,y,dim=-1).mean() for x,y in ((q1,ka),(r1,vb),(q2,kb),(r2,vc)))
        opt.zero_grad();loss.backward();opt.step();lossval=loss.item()
    train_time=time.perf_counter()-t
    ev=evaluate(model,arch,eval_set,V,M)
    return {'arch':arch+('_aux' if aux else ''),'seed':seed,'params':sum(p.numel() for p in model.parameters()),'training_s':round(train_time,2),'loss':round(lossval,5),**ev}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=350);p.add_argument('--seeds',type=int,default=3);p.add_argument('--vocab',type=int,default=50);p.add_argument('--chains',type=int,default=2);p.add_argument('--batch',type=int,default=64);p.add_argument('--eval',type=int,default=500);p.add_argument('--out',default='/mnt/data/amn_transformer_results.jsonl');args=p.parse_args()
    print('torch',torch.__version__,'settings',vars(args),flush=True)
    path=Path(args.out);path.write_text('')
    for arch,aux in [('amn',False),('amn',True),('transformer',False)]:
        for seed in range(args.seeds):
            result=run(seed,args.steps,args.vocab,args.chains,args.batch,args.eval,arch,aux)
            with path.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
            print('DONE',json.dumps(result,ensure_ascii=False),flush=True)