"""Pairwise expert ranking from causal prefix-error features."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/causal_pairwise_router_v9';OUT.mkdir(parents=True,exist_ok=True)
z=np.load(ROOT/'exp/results/causal_prefix_expert_router_v5/oof_weights.npz',allow_pickle=True);W=z['wells'].astype(str);G=z['G'];nrow=z['nrow'];names=z['names'];X=z['X']
b=np.load(ROOT/'exp/results/prefix_backtest_moe/backtest_features.npz',allow_pickle=True);bm={w:i for i,w in enumerate(b['wells'].astype(str))};BT=b['features'][[bm[w] for w in W]];X=np.c_[X,BT];K=G.shape[1];loss=np.sqrt(np.diagonal(G,axis1=1,axis2=2));e0=np.zeros(K);e0[0]=.1;e0[-1]=.9;base=np.tile(e0,(len(W),1));folds=list(GroupKFold(5).split(X,groups=W));rankW={t:np.zeros_like(base) for t in [1.,2.,4.]};knn=np.zeros_like(base);fid=np.zeros(len(W),int)
for fo,(tr,va) in enumerate(folds):
 sc=StandardScaler().fit(X[tr]);a=sc.transform(X[tr]);q=sc.transform(X[va]);strength=np.zeros((len(va),K))
 for i in range(K):
  for j in range(i+1,K):
   lab=(loss[tr,i]<loss[tr,j]).astype(int)
   if lab.min()==lab.max():p=np.full(len(va),lab[0],float)
   else:
    m=ExtraTreesClassifier(n_estimators=180,min_samples_leaf=15,max_features=.55,n_jobs=-1,random_state=9000+fo*100+i*K+j,class_weight='balanced').fit(a,lab);p=m.predict_proba(q)[:,list(m.classes_).index(1)]
   strength[:,i]+=p;strength[:,j]+=1-p
 for t in rankW:
  u=np.exp((strength-strength.max(1,keepdims=True))/t);rankW[t][va]=u/u.sum(1,keepdims=True)
 # Deterministic causal-analogue winner from nearest 15 prefix-error profiles.
 dist=((q[:,None]-a[None])**2).mean(2);near=np.argsort(dist,axis=1)[:,:15]
 for jj,v in enumerate(va):knn[v,np.argmin(np.average(loss[tr[near[jj]]],axis=0,weights=1/(np.sqrt(dist[jj,near[jj]])+1e-3)))]=1
 fid[va]=fo;print('fold',fo,flush=True)
def score(w,mask=None):
 if mask is None:mask=np.ones(len(W),bool)
 return float(np.sqrt(np.sum(nrow[mask]*np.einsum('ni,nij,nj->n',w[mask],G[mask],w[mask]))/nrow[mask].sum()))
grid=[]
for method,R in [('knn',knn)]+[(f'bt_t{t:g}',rankW[t]) for t in rankW]:
 for a in [.05,.1,.2,.35]:
  q=(1-a)*base+a*R;fg=[score(base,fid==k)-score(q,fid==k) for k in range(5)];grid.append({'method':method,'blend':a,'rmse':score(q),'fold_gains':fg,'wins':sum(x>0 for x in fg)})
best=min(grid,key=lambda x:x['rmse']);np.savez_compressed(OUT/'oof_weights.npz',wells=W,knn=knn,fold=fid,**{f'bt_t{t:g}':rankW[t] for t in rankW});out={'baseline':score(base),'candidate_oracle':float(np.sqrt(np.sum(nrow*np.min(loss,1)**2)/nrow.sum())),'grid':grid,'best':best};(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['sha256']=hashlib.sha256((OUT/'oof_weights.npz').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
