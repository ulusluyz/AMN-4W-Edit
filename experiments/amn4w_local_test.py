import torch, random, math, time, json
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
def norm(x):return F.normalize(x,dim=-1)
class AMN4W(nn.Module):
 def __init__(self,V=50,d=64,M=2):
  super().__init__(); self.d=d
  self.embedding=nn.Embedding(V,d,padding_idx=0); self.pos_embedding=nn.Embedding(3*M+2,d)
  self.fc_k=nn.Linear(d,d); self.fc_v=nn.Linear(d,d); self.fc_eta=nn.Linear(d,1); self.fc_gamma=nn.Linear(d,1); self.fc_q=nn.Linear(d,d); self.fc_q2=nn.Linear(d,d)
  self.fc_out1=nn.Linear(d*2,d); self.fc_out2=nn.Linear(d,V)
 @staticmethod
 def read(S,k):return torch.bmm(S,k.unsqueeze(-1)).squeeze(-1)
 @staticmethod
 def write(S,k,v,eta):return S+torch.bmm((eta*(v-AMN4W.read(S,k))).unsqueeze(-1),k.unsqueeze(1))
 @staticmethod
 def erase(S,k,strength):return S-torch.bmm((strength*AMN4W.read(S,k)).unsqueeze(-1),k.unsqueeze(1))
 def forward(self,x,mask):
  b,L=x.shape; tok=self.embedding(x); prev=F.pad(tok,(0,0,1,0))[:,:-1]; S=tok.new_zeros(b,self.d,self.d)
  for t in range(1,L):
   tt=tok[:,t]; k=norm(self.fc_k(prev[:,t])); v=norm(self.fc_v(tt)); eta=torch.sigmoid(self.fc_eta(tt))*mask[:,t:t+1]
   S=self.write(S,k,v,eta)
  return S
 def query(self,S,a,pos):
  tok=self.embedding(a); q1=norm(self.fc_q(tok));r1=self.read(S,q1);q2=norm(self.fc_q2(r1));r2=self.read(S,q2)
  p=self.pos_embedding(torch.full_like(a,pos)); logits=self.fc_out2(F.relu(self.fc_out1(torch.cat([tok+p,r2],-1))))
  return logits,(q1,r1,q2,r2)

def batch(bs,V,M):
 seq=[];aa=[];bb=[];cc=[]
 for _ in range(bs):
  ids=random.sample(range(2,V),3*M); ch=[ids[3*i:3*i+3] for i in range(M)]; j=random.randrange(M)
  seq.append([v for c in ch for v in c]);aa.append(ch[j][0]);bb.append(ch[j][1]);cc.append(ch[j][2])
 return (torch.tensor(seq),torch.tensor([[0,1,1]*M]*bs,dtype=torch.float32),torch.tensor(aa),torch.tensor(bb),torch.tensor(cc))

def span_residual(others,k,rtol=None):
 tol=rtol if rtol is not None else max(others.shape[-2:])*torch.finfo(others.dtype).eps
 U,s,_=torch.linalg.svd(others.double(),full_matrices=False); keep=s>tol*s[:,:1];Uk=U*keep.unsqueeze(1);k=k.double();r=k-torch.einsum('bdm,bm->bd',Uk,torch.einsum('bdm,bd->bm',Uk,k));return r,keep.sum(-1)
def dual_write(S,K,k,v,eta,min_rho=1e-6):
 r,_=span_residual(K.transpose(1,2),k); rho=r.norm(dim=-1);ok=rho>=min_rho;kd=r/rho.clamp(min=min_rho).square().unsqueeze(-1)
 S64=S.double();k64=k.double();upd=eta.double()*(v.double()-AMN4W.read(S64,k64));candidate=S64+torch.bmm(upd.unsqueeze(-1),kd.unsqueeze(1));return torch.where(ok[:,None,None],candidate,S64).to(S.dtype),rho,ok

def dual_erase(S,K,idx,min_rho=1e-6):
 b,N,d=K.shape;ar=torch.arange(b);keep=torch.ones(b,N,dtype=torch.bool);keep[ar,idx]=False;other=K[keep].view(b,N-1,d).transpose(1,2); k=K[ar,idx];r,_=span_residual(other,k);rho=r.norm(dim=-1);ok=rho>=min_rho;kd=r/rho.clamp(min=min_rho).square().unsqueeze(-1);Ss=S.double();candidate=Ss-torch.bmm(AMN4W.read(Ss,k.double()).unsqueeze(-1),kd.unsqueeze(1));return torch.where(ok[:,None,None],candidate,Ss).to(S.dtype),rho,ok

def numeric():
 torch.manual_seed(42); b=200;d=64;N=8;K=norm(torch.randn(b,N,d));V=norm(torch.randn(b,N,d));S=torch.zeros(b,d,d);one=torch.ones(b,1)
 for i in range(N):S=AMN4W.write(S,K[:,i],V[:,i],one)
 ix=torch.zeros(b,dtype=torch.long);se,rho,ok=dual_erase(S,K,ix)
 kept=(torch.einsum('bij,bnj->bni',se-S,K[:,1:])).abs().max().item();target=AMN4W.read(se,K[:,0]).norm(dim=-1).max().item()
 # include duplicate key rank deficiency
 K2=K.clone();K2[:,2]=K2[:,1];S2=torch.zeros_like(S)
 for i in range(N):S2=AMN4W.write(S2,K2[:,i],V[:,i],one)
 sd,rhod,okd=dual_erase(S2,K2,ix); keptdup=(torch.einsum('bij,bnj->bni',sd-S2,K2[:,1:])).abs().max().item()
 # comparable write c=.9,.99
 data={}
 for c in [.3,.9,.99,.9999]:
  u=norm(torch.randn(b,d));u=norm(u-(u*K[:,1]).sum(-1,keepdim=True)*K[:,1]);k=norm(c*K[:,1]+math.sqrt(1-c*c)*u);v=norm(torch.randn(b,d))
  std=AMN4W.write(S,k,v,one);dual,r,ok=dual_write(S,K,k,v,one)
  z0=AMN4W.read(S,K[:,1]); zs=AMN4W.read(std,K[:,1]);zd=AMN4W.read(dual,K[:,1]);
  data[str(c)]={'std_neighbor_cos':F.cosine_similarity(z0,zs).mean().item(),'dual_neighbor_cos':F.cosine_similarity(z0,zd).mean().item(),'dual_max_neighbor_change':(torch.einsum('bij,bnj->bni',dual-S,K)).abs().max().item(),'dual_update_norm':(dual-S).flatten(1).norm(dim=1).mean().item(),'rho_mean':r.mean().item(),'ok':ok.float().mean().item()}
 return {'dual_erase_target_max':target,'dual_erase_other_max':kept,'duplicated_other_max':keptdup,'duplicated_ok':okd.float().mean().item(),'write':data}

def train(seed,leak_w,decor_w,steps=350,V=50,M=2):
 torch.manual_seed(seed);random.seed(seed);m=AMN4W(V,64,M);opt=torch.optim.Adam(m.parameters(),lr=.002)
 for _ in range(steps):
  ctx,wm,a,b,c=batch(64,V,M);S=m(ctx,wm); logits,(q1,r1,q2,r2)=m.query(S,a,3*M)
  with torch.no_grad():
   ka=norm(m.fc_k(m.embedding(a)));kb=norm(m.fc_k(m.embedding(b)));vb=norm(m.fc_v(m.embedding(b)));vc=norm(m.fc_v(m.embedding(c)))
  aux=sum((1-F.cosine_similarity(x,y,dim=-1).mean()) for x,y in [(q1,ka),(r1,vb),(q2,kb),(r2,vc)])
  loss=F.cross_entropy(logits,c)+aux
  if leak_w:
   cand=ctx[:,2::3];kc=norm(m.fc_k(m.embedding(cand)));z=torch.einsum('bij,bmj->bmi',S,kc);loss=loss+leak_w*z.square().sum(-1).mean()
  if decor_w:
   keys=norm(m.fc_k(m.embedding(ctx.view(64,M,3)[:,:,:2].reshape(64,-1))));G=keys@keys.transpose(1,2);off=G-torch.diag_embed(torch.diagonal(G,dim1=1,dim2=2));loss=loss+decor_w*off.square().mean()
  opt.zero_grad();loss.backward();opt.step()
 with torch.no_grad():
  random.seed(10000+seed);torch.manual_seed(10000+seed);ctx,wm,a,b,c=batch(500,V,M);S=m(ctx,wm);logits,(q1,r1,q2,r2)=m.query(S,a,3*M)
  ids=torch.arange(2,V); kval=norm(m.fc_k(m.embedding(ids)));vval=norm(m.fc_v(m.embedding(ids)))
  final=(logits.argmax(-1)==c).float().mean().item();rc=(ids[(r2@vval.T).argmax(-1)]==c).float().mean().item()
  cand=ctx[:,2::3]; kc=norm(m.fc_k(m.embedding(cand))); kb=norm(m.fc_k(m.embedding(b)))
  leak=((S@kc.transpose(1,2)).norm(dim=1).mean(-1)/(AMN4W.read(S,kb).norm(dim=-1)+1e-8)).mean().item()
  keys=norm(m.fc_k(m.embedding(ctx.view(500,M,3)[:,:,:2].reshape(500,-1))));G=keys@keys.transpose(1,2);off=(G-torch.diag_embed(torch.diagonal(G,dim1=1,dim2=2))).abs();maxcos=off.amax(dim=(1,2)).mean().item()
 return {'final':final,'r2_C':rc,'leak':leak,'maxcos':maxcos}

if __name__=='__main__':
 t=time.time();print('version',torch.__version__,flush=True)
 print('NUMERIC',json.dumps(numeric()),flush=True)
 for name,lw,dw in [('baseline',0,0),('leak0.5',.5,0),('decorr5',0,5),('combined',.5,5)]:
  for seed in range(3):
   r=train(seed,lw,dw);print('TRAIN',name,seed,json.dumps(r),'elapsed',round(time.time()-t,1),flush=True)
 print('TOTAL',round(time.time()-t,1),flush=True)