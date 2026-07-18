"""Two-stage low-memory model: tiny structural context, then free final output."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from train_task099_quadratic import ROOT, load_data, stats


class ContextModel(torch.nn.Module):
    def __init__(self, rank=5, basis=3, context=3):
        super().__init__(); self.rank=rank; self.basis=basis; self.context=context
        p=np.arange(30,dtype=np.float32)
        S=np.stack([np.cos(np.pi*(p+.5)*k/30) for k in range(basis)])
        self.S=torch.nn.Parameter(torch.from_numpy(S))
        def P(*shape,scale=.25): return torch.nn.Parameter(torch.randn(*shape)*scale)
        self.ck=P(context,10); self.ci=P(context,basis); self.cj=P(context,basis)
        self.co=P(rank,10); self.cc=P(rank,10); self.cd=P(rank,context)
        self.qa=P(rank,basis); self.qb=P(rank,basis); self.qr=P(rank,basis); self.qc=P(rank,basis)
        # Start one context feature close to a constant: sum of all one-hot cells.
        with torch.no_grad():
            self.ck[0].fill_(.1); self.ci[0].zero_(); self.ci[0,0]=.1
            self.cj[0].zero_(); self.cj[0,0]=.1

    def forward(self,x):
        S=self.S[:,:10]
        fi=self.ci@S; fj=self.cj@S
        context=torch.einsum('nkij,dk,di,dj->nd',x,self.ck,fi,fj)
        fa,fb,fr,fc=[q@S for q in (self.qa,self.qb,self.qr,self.qc)]
        source=torch.einsum('noab,ta,tb->nto',x,fa,fb)
        gate=context@self.cd.T
        cell=torch.einsum('nkrc,tk->ntrc',x,self.cc)
        return torch.einsum('nto,nt,ntrc,to,tr,tc->norc',source,gate,cell,self.co,fr,fc)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rank',type=int,default=5); ap.add_argument('--basis',type=int,default=3)
    ap.add_argument('--context',type=int,default=3); ap.add_argument('--steps',type=int,default=25000)
    ap.add_argument('--batch',type=int,default=64); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--lr',type=float,default=.012); ap.add_argument('--resume')
    ap.add_argument('--output',default='other_model_onnx/task099_context.npz'); a=ap.parse_args()
    torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x,y=load_data(dev); m=ContextModel(a.rank,a.basis,a.context).to(dev)
    if a.resume:
        saved=np.load(ROOT/a.resume)
        with torch.no_grad():
            for n,p in m.named_parameters(): p.copy_(torch.from_numpy(saved[n]).to(dev))
    opt=torch.optim.Adam(m.parameters(),lr=a.lr)
    best=None
    for step in range(a.steps+1):
        ix=torch.randint(len(x),(min(a.batch,len(x)),),device=dev); z=m(x[ix]); yy=y[ix]
        present=x[ix].sum((-1,-2),keepdim=True)>0; neg=(yy<.5)&present
        loss=torch.nn.functional.softplus(-z[yy>.5]).mean()+torch.nn.functional.softplus(z[neg]).mean()
        if step>4000:
            signed=torch.where(yy>.5,z,-z); active=(yy>.5)|neg
            loss=loss+.25*torch.relu(.15-signed[active]).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),30); opt.step()
        if step%250==0:
            with torch.no_grad(): result=stats(m(x),y)
            print(step,float(loss),result,flush=True)
            if best is None or result[0]<best[0]:
                best=result; arr={n:p.detach().cpu().numpy() for n,p in m.named_parameters()}; arr.update(stats=np.asarray(result),rank=a.rank,basis=a.basis,context=a.context); np.savez(ROOT/a.output,**arr)
            if result[0]==0 and result[1]>=.08: break
    print('best',best,'saved',ROOT/a.output,flush=True)

if __name__=='__main__': main()
