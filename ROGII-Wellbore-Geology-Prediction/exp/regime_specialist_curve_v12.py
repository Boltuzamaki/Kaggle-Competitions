"""Strict outer-fold residual-curve regimes and specialists."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_regimes_v12';OUT.mkdir(parents=True,exist_ok=True)
v=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=v['wells'].astype(str);C=v['true'];lens=v['length'];Q=v['query'];b=np.load(ROOT/'exp/results/prefix_backtest_moe/backtest_features.npz',allow_pickle=True);bm={w:i for i,w in enumerate(b['wells'].astype(str))};X=np.c_[Q,b['features'][[bm[w] for w in W]]];folds=list(GroupKFold(5).split(X,groups=W));K=3
soft=np.zeros_like(C);hard=np.zeros_like(C);oracle=np.zeros_like(C);globalp=np.zeros_like(C);acc=[];foldmeta=[]
for fo,(tr,va) in enumerate(folds):
 sc=StandardScaler().fit(X[tr]);aa=sc.transform(X[tr]);bb=sc.transform(X[va]);xp=PCA(48,whiten=True,random_state=43).fit(aa);a=xp.transform(aa);q=xp.transform(bb);cp=PCA(24,random_state=44).fit(C[tr]);lat=cp.transform(C[tr]);km=KMeans(K,n_init=30,random_state=120+fo).fit(lat);lab=km.labels_;true_lab=np.argmin(((cp.transform(C[va])[:,None]-km.cluster_centers_[None])**2).sum(2),1)
 clf=ExtraTreesClassifier(n_estimators=800,min_samples_leaf=18,max_features=.65,n_jobs=-1,random_state=220+fo,class_weight='balanced').fit(a,lab);pro=clf.predict_proba(q);pro=np.clip(pro,1e-4,1);pro=pro**(1/1.5);pro/=pro.sum(1,keepdims=True);acc.append(float(np.mean(pro.argmax(1)==true_lab)))
 gm=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=12,max_features=.6,n_jobs=-1,random_state=45).fit(a,lat);globalp[va]=cp.inverse_transform(gm.predict(q));sp=[]
 for k in range(K):
  ii=np.where(lab==k)[0];m=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=max(5,min(12,len(ii)//8)),max_features=.65,n_jobs=-1,random_state=330+fo*10+k).fit(a[ii],lat[ii]);sp.append(cp.inverse_transform(m.predict(q)))
 sp=np.stack(sp,1);soft[va]=np.einsum('nk,nkl->nl',pro,sp);hard[va]=sp[np.arange(len(va)),pro.argmax(1)];oi=np.argmin(np.mean((sp-C[va,None])**2,2),1);oracle[va]=sp[np.arange(len(va)),oi];foldmeta.append({'fold':fo,'sizes':np.bincount(lab,minlength=K).tolist(),'accuracy':acc[-1]});print(foldmeta[-1],flush=True)
def score(p,ix=None):
 if ix is None:ix=np.arange(len(C))
 return float(np.sqrt(np.average(np.mean((C[ix]-p[ix])**2,1),weights=lens[ix])))
hardix=np.argsort(np.sqrt(np.mean(C*C,1)))[-int(.2*len(C)):];models={'global':globalp,'soft':soft,'hard':hard,'specialist_oracle':oracle};grid=[]
for n,p in models.items():grid.append({'model':n,'pooled':score(p),'folds':[score(p,va) for _,va in folds],'tail20':score(p,hardix)})
out={'zero':score(0*C),'regime_accuracy':acc,'mean_accuracy':float(np.mean(acc)),'models':grid,'foldmeta':foldmeta};np.savez_compressed(OUT/'oof.npz',wells=W,soft=soft,hard=hard,oracle=oracle,globalp=globalp,true=C,length=lens);out['sha256']=hashlib.sha256((OUT/'oof.npz').read_bytes()).hexdigest();(OUT/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
