"""Row-count/difficulty weighted functional residual manifold, strict GKF."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_weighted_v3';OUT.mkdir(parents=True,exist_ok=True)
z=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz')
W=z['wells'];C=z['true'];Q=z['query'];lens=z['length'].astype(float);L=C.shape[1]
configs=[('equal',0,False,False),('sqrt_len',.5,False,False),('len',1,False,False),('len_tail',1,True,False),('len_heel',1,False,True)]
loc=np.linspace(.85,1.15,L); folds=list(GroupKFold(5).split(Q,groups=W)); preds={n:np.zeros_like(C) for n,_,_,_ in configs}
def wpca_fit(Y,weight,k=24):
 weight=weight/weight.mean();mu=np.average(Y,axis=0,weights=weight); _,_,vt=np.linalg.svd((Y-mu)*np.sqrt(weight[:,None]),full_matrices=False)
 return mu,vt[:k]
for fold,(tr,va) in enumerate(folds):
 sc=StandardScaler().fit(Q[tr]);aa=sc.transform(Q[tr]);bb=sc.transform(Q[va])
 qp=PCA(n_components=min(48,len(tr)-1),whiten=True,random_state=43).fit(aa);a=qp.transform(aa);b=qp.transform(bb)
 diff=np.sqrt(np.mean(C[tr]**2,1)); med=np.median(diff)
 for name,pow_,tail,heel in configs:
  ww=(lens[tr]/np.median(lens[tr]))**pow_
  if tail: ww*=np.clip(diff/(med+1e-6),.5,2.)
  sw=np.sqrt(loc) if heel else np.ones(L)
  if name=='equal':
   cp=PCA(n_components=24,random_state=44).fit(C[tr]);mu,V=cp.mean_,cp.components_;latent=cp.transform(C[tr])
  else:
   mu,V=wpca_fit(C[tr]*sw,ww); latent=(C[tr]*sw-mu)@V.T
  m=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=12,max_features=.6,n_jobs=-1,random_state=45).fit(a,latent,sample_weight=ww)
  preds[name][va]=((m.predict(b)@V+mu)/sw)
def score(P,idx): return float(np.sqrt(np.average(np.mean((C[idx]-P[idx])**2,1),weights=lens[idx])))
rows=[]
for name,_,_,_ in configs:
 fs=[]
 for k,(_,va) in enumerate(folds): fs.append({'fold':k,'base':score(np.zeros_like(C),va),'model':score(preds[name],va),'gain':score(np.zeros_like(C),va)-score(preds[name],va)})
 hard=np.argsort(np.sqrt(np.mean(C*C,1)))[-int(.2*len(C)):]
 rows.append({'config':name,'pooled':score(preds[name],np.arange(len(C))),'folds':fs,'wins':sum(x['gain']>0 for x in fs),'tail20':score(preds[name],hard),'tail20_base':score(np.zeros_like(C),hard)})
best=min(rows,key=lambda x:x['pooled']);np.savez_compressed(OUT/'oof.npz',wells=W,pred=preds[best['config']],true=C,length=lens,config=best['config'])
out={'baseline':score(np.zeros_like(C),np.arange(len(C))),'grid':rows,'best':best,'protocol':'predeclared 5 configs; outer-train-only weights'}
(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['oof_sha256']=hashlib.sha256((OUT/'oof.npz').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
