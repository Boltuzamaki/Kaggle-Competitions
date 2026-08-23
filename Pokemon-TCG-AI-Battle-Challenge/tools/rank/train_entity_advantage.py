"""Entity/action Transformer for causal router-deviation advantage."""
from __future__ import annotations
import argparse, pickle, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_state_ranker import N_CARDS,N_TYPES,N_CTX,N_ZONES,CTX_DIM,NUM_DIM,STATE_NUM,collate,norms

class EntityAdvantage(nn.Module):
    def __init__(self,d=128,layers=3,heads=8):
        super().__init__(); self.d=d
        self.card=nn.Embedding(N_CARDS,d); self.zone=nn.Embedding(N_ZONES,d)
        self.otype=nn.Embedding(N_TYPES,d); self.sctx=nn.Embedding(N_CTX,d)
        self.snum=nn.Linear(STATE_NUM,d); self.onum=nn.Linear(NUM_DIM,d); self.ctx=nn.Linear(CTX_DIM,d)
        self.kind=nn.Embedding(4,d)
        layer=nn.TransformerEncoderLayer(d,heads,4*d,.1,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,layers); self.norm=nn.LayerNorm(d)
        self.head=nn.Sequential(nn.Linear(4*d,2*d),nn.GELU(),nn.Dropout(.1),nn.Linear(2*d,3))
    def forward(self,b):
        ctx,ctxid,scid,szone,snum,smask,cids,types,nums,mask,y=b; B=len(y)
        idx=y.clamp_max(cids.shape[1]-1); ar=torch.arange(B,device=y.device)
        chosen=self.card(cids[ar,idx])+self.otype(types[ar,idx])+self.onum(nums[ar,idx])+self.kind.weight[1]
        base=self.card(cids[:,0])+self.otype(types[:,0])+self.onum(nums[:,0])+self.kind.weight[2]
        ct=self.ctx(ctx)+self.sctx(ctxid)+self.kind.weight[0]
        state=self.card(scid)+self.zone(szone)+self.snum(snum)+self.kind.weight[3]
        seq=torch.cat([ct[:,None],chosen[:,None],base[:,None],state],1)
        pad=torch.cat([torch.zeros(B,3,dtype=torch.bool,device=y.device),~smask],1)
        z=self.norm(self.encoder(seq,src_key_padding_mask=pad))
        return self.head(torch.cat([z[:,0],z[:,1],z[:,2],z[:,1]-z[:,2]],-1))

def metrics(probs,truth):
    pred=probs.argmax(-1); recalls=[]
    for c in range(3):
        m=truth==c; recalls.append(float((pred[m]==c).float().mean()) if m.any() else 0)
    best={'threshold':float('inf'),'precision':0.,'coverage':0.,'positives':0}
    for t in torch.linspace(.5,.99,50):
        use=probs[:,2]>=t; n=int(use.sum())
        if n<25: continue
        precision=float((truth[use]==2).float().mean())
        if precision>=.7 and n>best['positives']:
            best={'threshold':float(t),'precision':precision,'coverage':n/len(truth),'positives':n}
    return sum(recalls)/3,best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=24); ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--seed',type=int,default=31); ap.add_argument('--layers',type=int,default=3); a=ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); rows=pickle.load(open(a.data,'rb')); stats=norms(rows)
    groups=sorted({r['episode'] for r in rows}); random.shuffle(groups); hold=set(groups[:max(1,len(groups)//5)])
    train=[r for r in rows if r['episode'] not in hold]; val=[r for r in rows if r['episode'] in hold]
    # Balanced sampling makes rare beneficial/harmful interventions visible.
    buckets={v:[r for r in train if r['advantage']==v] for v in (-1,0,1)}
    dev='cuda' if torch.cuda.is_available() else 'cpu'; model=EntityAdvantage(layers=a.layers).to(dev)
    opt=torch.optim.AdamW(model.parameters(),3e-4,weight_decay=1e-4); best=-1
    for ep in range(a.epochs):
        n=max(map(len,buckets.values())); epoch_rows=[]
        for v,b in buckets.items(): epoch_rows += random.choices(b,k=n)
        random.shuffle(epoch_rows); model.train()
        for i in range(0,len(epoch_rows),a.batch):
            rb=epoch_rows[i:i+a.batch]; b=[x.to(dev) for x in collate(rb,stats,maxs=80)]
            y=torch.tensor([r['advantage']+1 for r in rb],device=dev); logits=model(b)
            loss=F.cross_entropy(logits,y,label_smoothing=.03); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        model.eval(); ps=[]; ys=[]
        with torch.no_grad():
            for i in range(0,len(val),a.batch):
                rb=val[i:i+a.batch]; b=[x.to(dev) for x in collate(rb,stats,maxs=80)]
                ps.append(model(b).softmax(-1).cpu()); ys.extend(r['advantage']+1 for r in rb)
        balanced,gate=metrics(torch.cat(ps),torch.tensor(ys))
        if balanced>best: best=balanced; torch.save({'model':model.state_dict(),'stats':stats,'balanced_acc':balanced,'gate':gate,'layers':a.layers},a.out)
        print(ep+1,{'balanced_acc':balanced,'gate':gate},flush=True)

if __name__=='__main__': main()
