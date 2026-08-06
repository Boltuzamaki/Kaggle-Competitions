"""Honest outer-train-only exact/near overlap retrieval and path transfer.

Matching uses only complete trajectory/GR/typewell covariates. Candidate TVT is
read only for outer-training wells. A candidate surface U=TVT+Z is shifted by
the validation well's visible prefix, then transferred on normalized MD.
"""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/outer_train_overlap_transfer_v1';OUT.mkdir(parents=True,exist_ok=True)
L=64
def rs(v):
 v=np.asarray(v,float);ok=np.isfinite(v)
 if not ok.any():return np.zeros(L)
 v=np.interp(np.arange(len(v)),np.flatnonzero(ok),v[ok]);return np.interp(np.linspace(0,1,L),np.linspace(0,1,len(v)),v)
def hash_raw(h,tw):
 # Exact raw-file identity independent of TVT/formation/IDs.
 a=np.c_[h.MD,h.X,h.Y,h.Z,h.GR].astype('<f8');b=np.c_[tw.TVT,tw.GR].astype('<f8')
 return hashlib.sha256(np.nan_to_num(np.r_[a.ravel(),b.ravel()],nan=9.96921e36).tobytes()).hexdigest()

s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);groups=s['groups'].astype(str);wells=np.unique(groups)
records=[]
for w in wells:
 h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(ROOT/'data/train'/f'{w}__typewell.csv').sort_values('TVT')
 # Absolute pose plus centered shape. Absolute terms make true colocated/copy
 # wells close; centered terms distinguish trajectory and acquisition texture.
 x,y,z,g=[h[c].to_numpy(float) for c in ['X','Y','Z','GR']];md=h.MD.to_numpy(float)
 fp=np.r_[rs(x)/50000,rs(y)/50000,rs(z)/10000,rs(g)/100,
          rs(x-x[0])/5000,rs(y-y[0])/5000,rs(z-z[0])/500,rs(g-np.nanmedian(g))/50,
          rs(tw.TVT-tw.TVT.iloc[0])/500,rs(tw.GR-np.nanmedian(tw.GR))/50,
          len(h)/5000,(md[-1]-md[0])/5000]
 records.append((w,h,tw,np.nan_to_num(fp),hash_raw(h,tw)))
W=np.array([r[0] for r in records]);F=np.array([r[3] for r in records]);folds=list(GroupKFold(5).split(F,groups=W));fold_id=np.full(len(W),-1)
pred_by_w={}; report=[]
for k,(tr,va) in enumerate(folds):
 sc=StandardScaler().fit(F[tr]);a=sc.transform(F[tr]);b=sc.transform(F[va]);d=((b[:,None]-a[None])**2).mean(2);order=np.argsort(d,axis=1)
 for q,v in enumerate(va):
  w,h,tw,fp,hh=records[v];ps=np.flatnonzero(h.TVT_input.isna())[0];uq=(h.MD-h.MD.iloc[0])/(h.MD.iloc[-1]-h.MD.iloc[0]+1e-9)
  best=None
  # Fingerprint top-8 only; visible prefix chooses compatibility, never suffix truth.
  for rank,jj in enumerate(order[q,:8],1):
   t=tr[jj];wt,ht,twt,fpt,hht=records[t];ut=(ht.MD-ht.MD.iloc[0])/(ht.MD.iloc[-1]-ht.MD.iloc[0]+1e-9)
   surf=ht.TVT.to_numpy(float)+ht.Z.to_numpy(float);cand=np.interp(uq,ut,surf)-h.Z.to_numpy(float)
   shift=float(np.nanmedian(h.TVT_input.iloc[:ps]-cand[:ps]));cand+=shift
   pr=float(np.sqrt(np.nanmean((h.TVT_input.iloc[:ps]-cand[:ps])**2)))
   item=(pr,float(d[q,jj]),rank,t,wt,cand,hht==hh)
   if best is None or item[:3]<best[:3]:best=item
  pr,dist,rank,t,wt,cand,exact=best
  # Student/meta targets are delta from last-known TVT, not absolute TVT.
  pred_by_w[w]=cand[ps:]-float(h.TVT_input.iloc[ps-1]);fold_id[v]=k
  report.append({'well':w,'fold':k,'match':wt,'fingerprint_distance':dist,'fingerprint_rank_used':rank,'prefix_rmse':pr,'exact_hash':bool(exact),'rows':len(h)-ps})

# Exact-row reconstruction in Student artifact order.
y=s['y'];base=.1*s['accepted']+.9*s['replacement'];p=base.copy();row_report=pd.DataFrame(report).set_index('well');row_dist=np.empty(len(y));row_pr=np.empty(len(y));row_fold=np.empty(len(y),int);row_exact=np.zeros(len(y),bool)
cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)]
for a,b in zip(cuts[:-1],cuts[1:]):
 w=groups[a];p[a:b]=pred_by_w[w];r=row_report.loc[w];row_dist[a:b]=r.fingerprint_distance;row_pr[a:b]=r.prefix_rmse;row_fold[a:b]=int(r.fold);row_exact[a:b]=bool(r.exact_hash)
def rm(q,m=None):
 if m is None:m=np.ones(len(y),bool)
 return float(np.sqrt(np.mean((y[m]-q[m])**2)))
grid=[]
for prlim in [0.25,.5,1,2,4,8,16]:
 for qtile in [0,.001,.005,.01,.02,.05,.1,.2,1]:
  dl=float(np.quantile(row_report.fingerprint_distance,qtile)) if qtile else -1
  use=(row_pr<=prlim)&(row_dist<=dl) if qtile else row_exact&(row_pr<=prlim)
  out=base.copy();out[use]=p[use]
  grid.append({'prefix_rmse_limit':prlim,'distance_quantile':qtile,'distance_limit':dl,'wells_covered':int(len(np.unique(groups[use]))),'rows_covered':int(use.sum()),'coverage':float(use.mean()),'rmse':rm(out),**{f'f{k}':rm(out,row_fold==k) for k in range(5)}})
grid=pd.DataFrame(grid).sort_values('rmse');row_report.reset_index().to_csv(OUT/'matches.csv',index=False);grid.to_csv(OUT/'gate_grid.csv',index=False)
summary={'wells':len(W),'rows':len(y),'baseline':rm(base),'exact_hash_pairs':int(row_report.exact_hash.sum()),'nearest_distance_quantiles':{str(q):float(row_report.fingerprint_distance.quantile(q)) for q in [0,.001,.01,.05,.1,.5,1]},'prefix_rmse_quantiles':{str(q):float(row_report.prefix_rmse.quantile(q)) for q in [0,.01,.05,.1,.5,1]},'best':grid.iloc[0].to_dict(),'protocol':'outer GroupKFold; full legal query covariates; candidate TVT outer-train only; visible-prefix shift/gate'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
