"""Strict GKF prediction of coherent residual datum/slope around legal L2 v1."""
from pathlib import Path
import json, hashlib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

R=Path(__file__).resolve().parents[1]
OUT=R/'exp/results/residual_bias_slope_direct_v1';OUT.mkdir(parents=True,exist_ok=True)
C=np.load(R/'exp/kaggle_tabicl_v6_fpca_dataset/legal_v6_fpca_cache.npz',allow_pickle=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True)
L=np.load(R/'exp/results/legal_level2_all_oof_v1/oof.npz',allow_pickle=True)
wells=C['wells'].astype(str); X=np.c_[C['query'],C['backtest']].astype(float)
y=S['y']; groups=S['groups'].astype(str); student=.1*S['accepted']+.9*S['replacement']
center=student+L['anchored_difference']; assert np.array_equal(groups,L['groups'].astype(str))
# Saved row arrays retain the feature-table order, which is not contiguous by
# well.  Sort all row-level quantities together before functional aggregation.
order=np.argsort(groups,kind='stable'); y=y[order]; center=center[order]; groups=groups[order]
starts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1]; ends=np.r_[starts[1:],len(groups)]
assert np.array_equal(groups[starts],wells)
coef=[]
for a,b in zip(starts,ends):
    x=np.linspace(-1,1,b-a); coef.append(np.polyfit(x,y[a:b]-center[a:b],1)[::-1]) # bias,slope
coef=np.asarray(coef)
fold=np.empty(len(wells),int)
for k,(_,va) in enumerate(GroupKFold(5).split(X,groups=wells)): fold[va]=k

models={
 'ridge30':lambda: Ridge(30), 'ridge100':lambda:Ridge(100),
 'et8':lambda:ExtraTreesRegressor(n_estimators=700,min_samples_leaf=8,max_features=.6,n_jobs=-1,random_state=881),
 'et16':lambda:ExtraTreesRegressor(n_estimators=700,min_samples_leaf=16,max_features=.8,n_jobs=-1,random_state=882),
 'rf10':lambda:RandomForestRegressor(n_estimators=500,min_samples_leaf=10,max_features=.7,n_jobs=-1,random_state=883),
}
preds={n:np.zeros_like(coef) for n in models}
for k in range(5):
 tr=fold!=k;va=~tr;sc=StandardScaler().fit(X[tr]); Z=sc.transform(X); pc=PCA(96,whiten=False,random_state=7).fit(Z[tr]); Z=pc.transform(Z)
 for n,mk in models.items():
  m=mk();m.fit(Z[tr],coef[tr]);preds[n][va]=m.predict(Z[va])

def expand(c):
 out=np.empty(len(y))
 for i,(a,b) in enumerate(zip(starts,ends)):out[a:b]=c[i,0]+c[i,1]*np.linspace(-1,1,b-a)
 return out
def score(c):return float(np.sqrt(np.mean((y-(center+expand(c)))**2)))
grid=[]; best=(score(np.zeros_like(coef)),'zero',0,None)
for n,p in preds.items():
 for alpha in [.05,.1,.2,.35,.5,.75,1.0]:
  q=score(alpha*p); fs=[]
  for k in range(5):
   m=np.isin(groups,wells[fold==k]);fs.append(float(np.sqrt(np.mean((y[m]-(center[m]+expand(alpha*p)[m]))**2))))
  grid.append(dict(model=n,alpha=alpha,rmse=q,folds=fs))
  if q<best[0]:best=(q,n,alpha,p)
out=np.zeros_like(coef) if best[3] is None else best[2]*best[3]
np.savez_compressed(OUT/'oof.npz',wells=wells,bias_slope=out,fold=fold)
sha=hashlib.sha256((OUT/'oof.npz').read_bytes()).hexdigest()
summary={'baseline':score(np.zeros_like(coef)),'bias_oracle':score(np.c_[coef[:,0],np.zeros(len(coef))]),'line_oracle':score(coef),'best':{'rmse':best[0],'model':best[1],'alpha':best[2]},'grid':grid,'sha256':sha}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['best']|{'baseline':summary['baseline'],'bias_oracle':summary['bias_oracle'],'line_oracle':summary['line_oracle']},indent=2))
