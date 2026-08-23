"""Train a three-way causal action-advantage classifier."""
from __future__ import annotations

import argparse, pickle, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_state_ranker import (N_CARDS, N_TYPES, N_CTX, N_ZONES, CTX_DIM,
                                NUM_DIM, STATE_NUM, collate, norms)


class AdvantageNet(nn.Module):
    def __init__(self, d=96, h=256):
        super().__init__()
        self.card=nn.Embedding(N_CARDS,d); self.otype=nn.Embedding(N_TYPES,16)
        self.sctx=nn.Embedding(N_CTX,16); self.zone=nn.Embedding(N_ZONES,16)
        self.ctx=nn.Sequential(nn.Linear(CTX_DIM,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.state=nn.Sequential(nn.Linear(d+16+STATE_NUM,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.opt=nn.Sequential(nn.Linear(d+16+NUM_DIM,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(5*h+16,h),nn.ReLU(),nn.Dropout(.1),nn.Linear(h,3))

    def forward(self, batch):
        ctx,ctxid,scid,szone,snum,smask,cids,types,nums,mask,y=batch
        c=self.ctx(ctx); stok=self.state(torch.cat([self.card(scid),self.zone(szone),snum],-1))
        state=(stok*smask.unsqueeze(-1)).sum(1)/smask.sum(1,keepdim=True).clamp_min(1)
        opts=self.opt(torch.cat([self.card(cids),self.otype(types),nums],-1))
        idx=y.clamp_max(opts.shape[1]-1); chosen=opts[torch.arange(len(y),device=y.device),idx]
        base=opts[:,0]; sid=self.sctx(ctxid)
        return self.head(torch.cat([chosen,base,chosen-base,c,state,sid],-1))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch',type=int,default=384); a=ap.parse_args()
    rows=pickle.load(open(a.data,'rb')); stats=norms(rows)
    groups=sorted({r['episode'] for r in rows}); random.Random(17).shuffle(groups); hold=set(groups[:max(1,len(groups)//5)])
    train=[r for r in rows if r['episode'] not in hold]; val=[r for r in rows if r['episode'] in hold]
    dev='cuda' if torch.cuda.is_available() else 'cpu'; model=AdvantageNet().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-4)
    counts=torch.tensor([sum(r['advantage']==v for r in train) for v in (-1,0,1)],dtype=torch.float)
    weights=(counts.sum()/counts.clamp_min(1)).sqrt(); weights/=weights.mean(); weights=weights.to(dev)
    best=-1.0
    for epoch in range(a.epochs):
        model.train(); random.shuffle(train)
        for i in range(0,len(train),a.batch):
            rows_b=train[i:i+a.batch]; batch=[x.to(dev) for x in collate(rows_b,stats)]
            target=torch.tensor([r['advantage']+1 for r in rows_b],device=dev)
            loss=F.cross_entropy(model(batch),target,weight=weights); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        model.eval(); pred=[]; truth=[]; probs=[]
        with torch.no_grad():
            for i in range(0,len(val),a.batch):
                rows_b=val[i:i+a.batch]; batch=[x.to(dev) for x in collate(rows_b,stats)]
                p=model(batch).softmax(-1); pred.extend(p.argmax(-1).cpu().tolist()); probs.extend(p[:,2].cpu().tolist()); truth.extend([r['advantage']+1 for r in rows_b])
        positive=[i for i,p in enumerate(probs) if p>=.8]; precision=sum(truth[i]==2 for i in positive)/max(1,len(positive))
        balanced=sum(sum(pred[i]==c for i,t in enumerate(truth) if t==c)/max(1,sum(t==c for t in truth)) for c in range(3))/3
        if balanced>best:
            best=balanced; torch.save({'model':model.state_dict(),'stats':stats,'balanced_acc':balanced,'positive_precision_08':precision,'positive_coverage_08':len(positive)/len(val)},a.out)
        print(epoch+1,{'balanced_acc':balanced,'positive_precision_08':precision,'positive_coverage_08':len(positive)/len(val)},flush=True)


if __name__=='__main__': main()
