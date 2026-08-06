"""Strict GKF robust GPU boosting of immutable causal-prefix features to suffix FPCA."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz'
OUT=ROOT/'exp/results/causal_prefix_boost_fpca_v1';OUT.mkdir(parents=True,exist_ok=True)
Z=np.load(SRC,allow_pickle=True);W=Z['wells'].astype(str);C=Z['true'];Q=np.nan_to_num(Z['query']); folds=list(GroupKFold(5).split(Q,groups=W))
configs=[dict(name='huber_d3',max_depth=3,min_child_weight=12,reg_lambda=20,learning_rate=.035,n_estimators=450),
         dict(name='huber_d5',max_depth=5,min_child_weight=20,reg_lambda=35,learning_rate=.025,n_estimators=550)]
curves={c['name']:np.zeros_like(C) for c in configs}
for fi,(tr,va) in enumerate(folds):
 cp=PCA(20,random_state=301).fit(C[tr]);A=cp.transform(C)
 for cfg in configs:
  pa=np.zeros((len(va),20))
  for j in range(20):
   kw={k:v for k,v in cfg.items() if k!='name'}
   m=XGBRegressor(**kw,objective='reg:pseudohubererror',subsample=.8,colsample_bytree=.65,
        tree_method='hist',device='cuda',random_state=310+fi*31+j,n_jobs=2)
   m.fit(Q[tr],A[tr,j],verbose=False);pa[:,j]=m.predict(Q[va])
  curves[cfg['name']][va]=cp.inverse_transform(pa)
  print(fi,cfg['name'],flush=True)

s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=s['y'];groups=s['groups'].astype(str);base=.1*s['accepted']+.9*s['replacement']
cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)];ixs={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}
def row(P):
 o=np.empty_like(y)
 for w,c in zip(W,P):
  ix=ixs[w];o[ix]=np.interp(np.linspace(0,1,len(ix)),np.linspace(0,1,len(c)),c)
 return o
def rm(p):return float(np.sqrt(np.mean((y-p)**2)))
et=row(Z['et']); rp={n:row(p) for n,p in curves.items()}; grid=[]
for n,p in rp.items():
 for a in [0,.1,.2,.35,.5,.75,1]:grid.append(dict(model=n,et=0,alpha=a,rmse=rm(base+a*p)))
 # Immutable ET full plus small additive boost correction.
 for a in [-.2,-.1,-.05,.025,.05,.1,.2,.35]:grid.append(dict(model=n,et=1,alpha=a,rmse=rm(base+et+a*p)))
d=pd.DataFrame(grid).sort_values('rmse');d.to_csv(OUT/'blend_addone_grid.csv',index=False)
np.savez_compressed(OUT/'predictions.npz',wells=W,**curves)
err=y-(base+et);cors={n:float(np.corrcoef(p,err)[0,1]) for n,p in rp.items()};cors['models']=float(np.corrcoef(rp['huber_d3'],rp['huber_d5'])[0,1])
summary={'device':'cuda','input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'baseline':rm(base),'et':rm(base+et),'best':d.iloc[0].to_dict(),'correlations_with_et_residual':cors,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
