"""Ablate causal expert-family backtest signatures in v1 manifold ET."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_expert_errors_v6';OUT.mkdir(parents=True,exist_ok=True)
v=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=v['wells'].astype(str);C=v['true'];Q=v['query'];lens=v['length']
b=np.load(ROOT/'exp/results/prefix_backtest_moe/backtest_features.npz',allow_pickle=True);bw=b['wells'].astype(str);bm={w:i for i,w in enumerate(bw)};BT=b['features'][[bm[w] for w in W]]
sets={'base':Q,'add':np.c_[Q,BT],'replace':BT};pred={n:np.zeros_like(C) for n in sets};folds=list(GroupKFold(5).split(Q,groups=W))
for k,(tr,va) in enumerate(folds):
 cp=PCA(24,random_state=44).fit(C[tr]);lat=cp.transform(C[tr])
 for n,X in sets.items():
  sc=StandardScaler().fit(X[tr]);aa=sc.transform(X[tr]);bb=sc.transform(X[va]);pc=PCA(min(48,len(tr)-1),whiten=True,random_state=43).fit(aa)
  m=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=12,max_features=.6,n_jobs=-1,random_state=45).fit(pc.transform(aa),lat);pred[n][va]=cp.inverse_transform(m.predict(pc.transform(bb)))
 print('fold',k,flush=True)
def score(p,ix):return float(np.sqrt(np.average(np.mean((C[ix]-p[ix])**2,1),weights=lens[ix])))
grid=[]
for n,p in pred.items():
 fg=[score(np.zeros_like(C),va)-score(p,va) for _,va in folds];grid.append({'model':n,'score':score(p,np.arange(len(C))),'fold_gains':fg,'wins':sum(x>0 for x in fg)})
best=min(grid,key=lambda x:x['score']);np.savez_compressed(OUT/'oof.npz',wells=W,pred=pred[best['model']],base=pred['base'],true=C,length=lens,model=best['model'])
out={'baseline_zero':score(np.zeros_like(C),np.arange(len(C))),'grid':grid,'best':best,'bt_features':BT.shape[1],'protocol':'strict outer GKF; predeclared base/add/replace'};(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['sha256']=hashlib.sha256((OUT/'oof.npz').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
