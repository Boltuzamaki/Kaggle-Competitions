"""Strict outer-fold residual priors keyed by paired typewell identity/family."""
from pathlib import Path
import glob,hashlib,json,collections
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/typewell_identity_residual_prior_v1';OUT.mkdir(parents=True,exist_ok=True)
v=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=v['wells'].astype(str);C=v['true'];lens=v['length'];L=C.shape[1]
z6=np.load(ROOT/'exp/results/generative_curve_prior_expert_errors_v6/oof.npz');assert np.array_equal(W,z6['wells'].astype(str));V=z6['pred']
def rs(x,n=256):return np.interp(np.linspace(0,1,n),np.linspace(0,1,len(x)),np.asarray(x,float))
def info(w):
 d=pd.read_csv(ROOT/'data/train'/f'{w}__typewell.csv').sort_values('TVT');a=d[['TVT','GR']].to_numpy(float);full=hashlib.sha256(np.nan_to_num(np.round(a,6),nan=9e30).tobytes()).hexdigest()
 gr=rs(a[:,1]);gr=(gr-np.nanmedian(gr))/(np.nanstd(gr)+1e-6);norm=hashlib.sha256(np.round(gr,3).tobytes()).hexdigest()
 g=d.Geology.astype(str).to_numpy();ix=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1,len(g)];labs=tuple(g[ix[:-1]]);tops=d.TVT.to_numpy()[ix[:-1]];th=np.diff(np.r_[tops,d.TVT.iloc[-1]]);th=th/(th.sum()+1e-9);family=hashlib.sha256(repr((labs,tuple(np.round(th,1)))).encode()).hexdigest()
 feat=np.r_[gr,rs(a[:,0]-a[0,0])/(a[-1,0]-a[0,0]+1e-6)];return full,norm,family,np.nan_to_num(feat)
I=[info(w) for w in W];full=np.array([x[0] for x in I]);norm=np.array([x[1] for x in I]);fam=np.array([x[2] for x in I]);F=np.array([x[3] for x in I]);folds=list(GroupKFold(5).split(F,groups=W));fold_id=np.full(len(W),-1,int);P={m:np.zeros_like(C) for m in ['exact','family','near3','near8']};support={m:np.zeros(len(W),int) for m in P};distance=np.full(len(W),np.inf)
for k,(tr,va) in enumerate(folds):
 fold_id[va]=k
 for q in va:
  for mode,key in [('exact',full),('family',fam)]:
   jj=tr[key[tr]==key[q]];support[mode][q]=len(jj)
   if len(jj):P[mode][q]=C[jj].mean(0)
  d=((F[tr]-F[q])**2).mean(1);oo=np.argsort(d);distance[q]=float(np.sqrt(d[oo[0]]))
  for mode,n in [('near3',3),('near8',8)]:
   jj=tr[oo[:n]];dd=np.sqrt(d[oo[:n]]);ww=np.exp(-dd/(np.median(dd)+1e-6));ww/=ww.sum();P[mode][q]=ww@C[jj];support[mode][q]=n

# Exact row-weighted reconstruction and fixed support shrinkage grid around v6.
s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=s['y'];groups=s['groups'].astype(str);base=.1*s['accepted']+.9*s['replacement'];cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)];ixs={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}
def row(A):
 o=np.empty_like(y)
 for w,c in zip(W,A):
  ix=ixs[w];o[ix]=np.interp(np.linspace(0,1,len(ix)),np.linspace(0,1,L),c)
 return o
rv=row(V);R={m:row(P[m]) for m in P};defw={m:np.empty_like(y) for m in P};rd=np.empty_like(y);rf=np.empty(len(y),int)
for i,w in enumerate(W):
 for m in P:defw[m][ixs[w]]=support[m][i]
 rd[ixs[w]]=distance[i];rf[ixs[w]]=fold_id[i]
def rm(p):return float(np.sqrt(np.mean((y-p)**2)))
grid=[]
for mode in P:
 for shrink in [0.25,.5,1,2,4,8]:
  sw=defw[mode]/(defw[mode]+shrink)
  if mode.startswith('near'):
   # Near priors are distance-shrunk using a training-independent smooth scale.
   sw*=np.exp(-rd/0.5)
  for alpha in [.05,.1,.2,.35,.5,.75,1]:grid.append({'mode':mode,'shrink':shrink,'alpha':alpha,'covered_wells':int((support[mode]>0).sum()),'rmse':rm(base+rv+alpha*sw*R[mode])})
g=pd.DataFrame(grid).sort_values('rmse');g.to_csv(OUT/'grid.csv',index=False)
counts={}
for name,key in [('full',full),('normalized',norm),('formation_family_0p1',fam)]:
 c=collections.Counter(key);counts[name]={'groups':len(c),'duplicate_groups':sum(n>1 for n in c.values()),'grouped_wells':sum(n for n in c.values() if n>1),'max_group':max(c.values()),'size_histogram':dict(collections.Counter(c.values()))}
bb=g.iloc[0];mode=bb['mode'];sw=defw[mode]/(defw[mode]+float(bb['shrink']));sw*=np.exp(-rd/0.5) if str(mode).startswith('near') else 1;bp=base+rv+float(bb['alpha'])*sw*R[mode]
summary={'wells':len(W),'baseline':rm(base),'v6':rm(base+rv),'identity_counts':counts,'outer_exact_supported':int((support['exact']>0).sum()),'outer_family_supported':int((support['family']>0).sum()),'near_distance_quantiles':{str(q):float(np.quantile(distance,q)) for q in [0,.01,.05,.1,.5,1]},'best':bb.to_dict(),'best_fold_scores':[float(np.sqrt(np.mean((y[rf==k]-bp[rf==k])**2))) for k in range(5)],'v6_fold_scores':[float(np.sqrt(np.mean((y[rf==k]-(base+rv)[rf==k])**2))) for k in range(5)],'gate':'>0.05 over v6'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/'oof.npz',wells=W,**P,exact_support=support['exact'],family_support=support['family'],near_distance=distance);print(json.dumps(summary,indent=2))
