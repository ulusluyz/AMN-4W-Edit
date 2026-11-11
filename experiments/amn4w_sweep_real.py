"""Independent, synthetic AMN-4W proxy sweep. Does NOT test GitHub v0.4 production architecture.
All outputs written to JSONL with experiment signature. Resumable by name, seed.
"""
import argparse, hashlib, json, os, random, statistics, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from amn4w_local_test import AMN4W, norm, batch, span_residual

CONFIGS={'baseline':(0.0,0.0),'leak0.5':(.5,0.0),'decorr5':(0.0,5.0),'decorr20':(0.0,20.0),'leak0.5+decorr5':(.5,5.0)}

def train_once(name, seed, steps, V, M, n_eval, bs=64, aux_w=1.0):
    torch.set_num_threads(1)
    start=time.time(); lw,dw=CONFIGS[name]
    torch.manual_seed(seed); random.seed(seed)
    model=AMN4W(V,64,M); opt=torch.optim.Adam(model.parameters(),lr=.002)
    lastloss=None
    for step in range(steps):
        ctx,wm,a,b,c=batch(bs,V,M)
        S=model(ctx,wm)
        logits,(q1,r1,q2,r2)=model.query(S,a,3*M)
        with torch.no_grad():
            ka=norm(model.fc_k(model.embedding(a))); kb=norm(model.fc_k(model.embedding(b)))
            vb=norm(model.fc_v(model.embedding(b))); vc=norm(model.fc_v(model.embedding(c)))
        loss=F.cross_entropy(logits,c)
        loss=loss+aux_w*sum(1-F.cosine_similarity(x,y,dim=-1).mean() for x,y in ((q1,ka),(r1,vb),(q2,kb),(r2,vc)))
        if lw:
            cand=ctx[:,2::3]; kc=norm(model.fc_k(model.embedding(cand)))
            z=torch.einsum('bij,bmj->bmi',S,kc)
            loss=loss+lw*z.square().sum(-1).mean()
        if dw:
            keys=ctx.reshape(bs,M,3)[:,:,:2].reshape(bs,-1)
            K=norm(model.fc_k(model.embedding(keys)))
            G=K@K.transpose(1,2)
            off=G-torch.diag_embed(torch.diagonal(G,dim1=1,dim2=2))
            loss=loss+dw*off.square().mean()
        if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss {name} seed {seed} step {step}')
        opt.zero_grad();loss.backward();opt.step();lastloss=float(loss.detach())
    with torch.no_grad():
        random.seed(10000+seed);torch.manual_seed(10000+seed)
        ctx,wm,a,b,c=batch(n_eval,V,M)
        S=model(ctx,wm);logits,(q1,r1,q2,r2)=model.query(S,a,3*M)
        ids=torch.arange(2,V); E=model.embedding(ids)
        Kall=norm(model.fc_k(E));Vall=norm(model.fc_v(E))
        nearK=lambda x:ids[(x@Kall.T).argmax(-1)]
        nearV=lambda x:ids[(x@Vall.T).argmax(-1)]
        final=(logits.argmax(-1)==c).float().mean().item()
        r2C=(nearV(r2)==c).float().mean().item()
        q2B=(nearK(q2)==b).float().mean().item()
        r1B=(nearV(r1)==b).float().mean().item()
        kc=norm(model.fc_k(model.embedding(ctx[:,2::3])))
        kb=norm(model.fc_k(model.embedding(b)))
        leak=((S@kc.transpose(1,2)).norm(dim=1).mean(-1) / (AMN4W.read(S,kb).norm(dim=-1)+1e-8)).mean().item()
        chains=ctx.reshape(n_eval,M,3)
        KA=norm(model.fc_k(model.embedding(chains[:,:,0])));KB=norm(model.fc_k(model.embedding(chains[:,:,1])))
        allK=torch.cat([KA,KB],1)
        G=allK@allK.transpose(1,2)
        off=(G-torch.diag_embed(torch.diagonal(G,dim1=1,dim2=2))).abs()
        maxcos=off.amax(dim=(1,2))
        rhos=[];rank_def=torch.zeros(n_eval,dtype=torch.bool)
        for i in range(allK.shape[1]):
            others=torch.cat((allK[:,:i],allK[:,i+1:]),dim=1)
            r,rank=span_residual(others.transpose(1,2),allK[:,i]);rhos.append(r.norm(dim=-1))
            rank_def|=rank < allK.shape[1]-1
        rho_min=torch.stack(rhos,1).min(1).values
        ktarget=norm(model.fc_k(model.embedding(b)))
        Sdel=AMN4W.erase(S,ktarget,torch.ones(n_eval,1))
        read0=AMN4W.read(S,allK.reshape(-1,64)).reshape(n_eval,2*M,64) if False else torch.einsum('bij,bnj->bni',S,allK)
        read1=torch.einsum('bij,bnj->bni',Sdel,allK)
        istarget=(chains[:,:,0]==a[:,None]); other=torch.cat((~istarget,~istarget),1)
        change=(read1.norm(dim=-1)-read0.norm(dim=-1)).abs()/(read0.norm(dim=-1)+1e-8)
        cos=F.cosine_similarity(read0,read1,dim=-1)
        return dict(name=name,seed=seed,leak_w=lw,decorr_w=dw,steps=steps,V=V,M=M,n_eval=n_eval,bs=bs,aux_w=aux_w,
                    final=final,r2_C=r2C,q2_kB=q2B,r1_B=r1B,leak=leak,erase_dnorm=change[other].mean().item(),
                    erase_cos=cos[other].mean().item(),maxcos=maxcos.mean().item(),maxcos_gt=(maxcos>.6).float().mean().item(),
                    rho_min=rho_min.mean().item(),rho_lt03=(rho_min<.3).float().mean().item(),rank_def=rank_def.float().mean().item(),
                    last_loss=lastloss,seconds=round(time.time()-start,2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='/mnt/data/amn4w_full_sweep.jsonl');p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--vocab',type=int,default=300);p.add_argument('--chains',type=int,default=4);p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--eval',type=int,default=1000);p.add_argument('--workers',type=int,default=4);p.add_argument('--configs',default=','.join(CONFIGS));a=p.parse_args()
    signature=hashlib.sha256(json.dumps({'model_sha256':hashlib.sha256(Path('/mnt/data/amn4w_local_test.py').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':a.steps,'V':a.vocab,'M':a.chains,'seeds':a.seeds,'n_eval':a.eval,'torch':torch.__version__,'configs':a.configs},sort_keys=True).encode()).hexdigest()[:16]
    old=[]
    if Path(a.out).exists():old=[json.loads(x) for x in Path(a.out).read_text().splitlines() if x.strip()]
    done={(x['name'],x['seed']) for x in old if x.get('exp_sig')==signature}
    tasks=[(name,seed) for name in a.configs.split(',') for seed in range(a.seeds) if (name,seed) not in done]
    print('SIG',signature,'TASKS',len(tasks),'torch',torch.__version__,flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(train_once,name,seed,a.steps,a.vocab,a.chains,a.eval):(name,seed) for name,seed in tasks}
        for f in as_completed(fs):
            name,seed=fs[f]
            try:res=f.result()
            except Exception as e:
                print('ERROR',name,seed,repr(e),flush=True);continue
            res['exp_sig']=signature
            with open(a.out,'a') as fh:fh.write(json.dumps(res)+'\n');fh.flush();os.fsync(fh.fileno())
            print('DONE',name,seed,'acc',res['final'],'leak',round(res['leak'],4),'time',res['seconds'],flush=True)
    rows=[json.loads(x) for x in Path(a.out).read_text().splitlines() if x.strip()]
    rows=[x for x in rows if x.get('exp_sig')==signature]
    print('COMPLETED',len(rows),'/',len(tasks)+len(done),'SIGNATURE',signature)
    for name in a.configs.split(','):
        sub=[r for r in rows if r['name']==name]
        if sub:
            print('SUMMARY',name,'n',len(sub), ' '.join(f'{k}={statistics.mean(x[k] for x in sub):.4f}' for k in ('final','r2_C','leak','erase_dnorm','maxcos','rho_min','rho_lt03')))
if __name__=='__main__':main()