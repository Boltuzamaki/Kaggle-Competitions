"""Train a listwise elite-action ranker with pooled full-state card tokens."""
from __future__ import annotations

import argparse, os, pickle, random, statistics as st
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CARDS, N_TYPES, N_CTX, N_ZONES = 1400, 20, 64, 10
CTX_DIM, NUM_DIM, STATE_NUM = 21, 8, 2


class StateRanker(nn.Module):
    def __init__(self, d=96, h=256):
        super().__init__()
        self.card = nn.Embedding(N_CARDS, d)
        self.otype = nn.Embedding(N_TYPES, 16)
        self.sctx = nn.Embedding(N_CTX, 16)
        self.zone = nn.Embedding(N_ZONES, 16)
        self.ctx = nn.Sequential(nn.Linear(CTX_DIM, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.state = nn.Sequential(nn.Linear(d+16+STATE_NUM, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.opt = nn.Sequential(nn.Linear(d+16+NUM_DIM, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(3*h+16, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, ctx, ctxid, scid, szone, snum, smask, cids, types, nums, mask):
        B, O = cids.shape
        c = self.ctx(ctx)
        stok = self.state(torch.cat([self.card(scid), self.zone(szone), snum], -1))
        den = smask.sum(1, keepdim=True).clamp_min(1).float()
        state = (stok * smask.unsqueeze(-1)).sum(1) / den
        opt = self.opt(torch.cat([self.card(cids), self.otype(types), nums], -1))
        sid = self.sctx(ctxid)
        joint = torch.cat([opt, c[:,None,:].expand(-1,O,-1),
                           state[:,None,:].expand(-1,O,-1), sid[:,None,:].expand(-1,O,-1)], -1)
        return self.score(joint).squeeze(-1).masked_fill(~mask, -1e9)


def norms(rows):
    cmu=[st.mean(r['ctx'][i] for r in rows) for i in range(CTX_DIM)]
    csd=[max(st.pstdev(r['ctx'][i] for r in rows),1e-3) for i in range(CTX_DIM)]
    opts=[x for r in rows for x in r['nums']]
    nmu=[st.mean(x[i] for x in opts) for i in range(NUM_DIM)]
    nsd=[max(st.pstdev(x[i] for x in opts),1e-3) for i in range(NUM_DIM)]
    states=[x[2] for r in rows for x in r.get('state',[])] or [[0,0]]
    smu=[st.mean(x[i] for x in states) for i in range(STATE_NUM)]
    ssd=[max(st.pstdev(x[i] for x in states),1e-3) for i in range(STATE_NUM)]
    return dict(ctx_mu=cmu,ctx_sd=csd,num_mu=nmu,num_sd=nsd,state_mu=smu,state_sd=ssd)


def collate(batch, stats, maxo=60, maxs=100):
    B=len(batch); O=min(max(len(r['cids']) for r in batch),maxo); S=max(1,min(max(len(r.get('state',[])) for r in batch),maxs))
    ctx=torch.zeros(B,CTX_DIM); ctxid=torch.zeros(B,dtype=torch.long)
    scid=torch.zeros(B,S,dtype=torch.long); szone=torch.zeros(B,S,dtype=torch.long); snum=torch.zeros(B,S,STATE_NUM); smask=torch.zeros(B,S,dtype=torch.bool)
    cids=torch.zeros(B,O,dtype=torch.long); types=torch.zeros(B,O,dtype=torch.long); nums=torch.zeros(B,O,NUM_DIM); mask=torch.zeros(B,O,dtype=torch.bool); y=torch.zeros(B,dtype=torch.long)
    for i,r in enumerate(batch):
        ctx[i]=torch.tensor([(v-stats['ctx_mu'][j])/stats['ctx_sd'][j] for j,v in enumerate(r['ctx'][:CTX_DIM])])
        ctxid[i]=min(int(r.get('ctxid',0)),N_CTX-1)
        for j,(cid,z,n) in enumerate(r.get('state',[])[:S]):
            scid[i,j]=min(max(cid,0),N_CARDS-1); szone[i,j]=min(max(z,0),N_ZONES-1)
            snum[i,j]=torch.tensor([(n[k]-stats['state_mu'][k])/stats['state_sd'][k] for k in range(STATE_NUM)]); smask[i,j]=1
        for j in range(min(len(r['cids']),O)):
            cids[i,j]=min(max(r['cids'][j],0),N_CARDS-1); types[i,j]=min(max(r['types'][j],0),N_TYPES-1)
            nums[i,j]=torch.tensor([(r['nums'][j][k]-stats['num_mu'][k])/stats['num_sd'][k] for k in range(NUM_DIM)]); mask[i,j]=1
        y[i]=r['y']
    return ctx,ctxid,scid,szone,snum,smask,cids,types,nums,mask,y


def gate_report(pred, margin, targets, precision_floor=0.75):
    """Find a conservative threshold for learned overrides of option zero."""
    candidates = []
    for threshold in torch.unique(margin[pred != 0]).sort(descending=True).values.tolist():
        use = (pred != 0) & (margin >= threshold)
        n = int(use.sum())
        if n < 25:
            continue
        precision = float((pred[use] == targets[use]).float().mean())
        if precision >= precision_floor:
            candidates.append((n, precision, float(threshold)))
    if not candidates:
        return {'threshold': float('inf'), 'overrides': 0, 'precision': 0.0,
                'coverage': 0.0}
    n, precision, threshold = max(candidates)
    return {'threshold': threshold, 'overrides': n, 'precision': precision,
            'coverage': n / len(targets)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch',type=int,default=384); ap.add_argument('--nonzero-weight',type=float,default=1.0); ap.add_argument('--gate-precision',type=float,default=0.75); a=ap.parse_args()
    rows=pickle.load(open(a.data,'rb')); stats=norms(rows)
    groups=sorted(set(r['episode'] for r in rows)); random.Random(7).shuffle(groups); vg=set(groups[:max(1,len(groups)//5)])
    train=[r for r in rows if r['episode'] not in vg]; val=[r for r in rows if r['episode'] in vg]
    dev='cuda' if torch.cuda.is_available() else 'cpu'; model=StateRanker().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    base=sum(r['y']==0 for r in val)/len(val); best=0
    print(len(train),len(val),'baseline',base,flush=True)
    for ep in range(a.epochs):
        model.train(); random.shuffle(train)
        for i in range(0,len(train),a.batch):
            t=[x.to(dev) for x in collate(train[i:i+a.batch],stats)]; losses=F.cross_entropy(model(*t[:-1]),t[-1],reduction='none'); weights=torch.where(t[-1]!=0,a.nonzero_weight,1.0); loss=(losses*weights).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        model.eval(); cor=t3=tot=0; all_pred=[]; all_margin=[]; all_y=[]
        with torch.no_grad():
            for i in range(0,len(val),a.batch):
                t=[x.to(dev) for x in collate(val[i:i+a.batch],stats)]; s=model(*t[:-1]); y=t[-1]; pred=s.argmax(-1); cor+=(pred==y).sum().item(); t3+=(s.topk(min(3,s.shape[1]),-1).indices==y[:,None]).any(-1).sum().item(); tot+=len(y); all_pred.append(pred.cpu()); all_margin.append((s.gather(1,pred[:,None]).squeeze(1)-s[:,0]).cpu()); all_y.append(y.cpu())
        acc=cor/tot
        gate=gate_report(torch.cat(all_pred),torch.cat(all_margin),torch.cat(all_y),a.gate_precision)
        if acc>best: best=acc; torch.save({'model':model.state_dict(),'stats':stats,'acc':acc,'gate':gate},a.out)
        print(ep+1,acc,t3/tot,gate,'saved' if acc==best else '',flush=True)
    print('best',best,'lift',best-base)


if __name__=='__main__': main()
