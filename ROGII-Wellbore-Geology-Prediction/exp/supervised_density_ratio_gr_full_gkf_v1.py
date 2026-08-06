"""Frozen strict 5-fold replay of supervised GR density-ratio emission."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
R=Path(__file__).resolve().parents[1];SRC=R/'exp/results/legal_level2_all_oof_v1/oof.npz';OUT=R/'exp/results/supervised_density_ratio_gr_full_gkf_v1';OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];g=S['groups'].astype(str);center=.1*S['accepted']+.9*S['replacement']+np.load(SRC,allow_pickle=True)['anchored_difference'];cuts=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1,len(g)];ixs={g[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])};W=np.sort(np.unique(g));folds=list(GroupKFold(5).split(W,groups=W));labels=np.array([(a,b) for a in [-4,-2,0,2,4] for b in [-4,-2,0,2,4]]);pred=center.copy()
def affine(x,y):
 q=np.isfinite(x)&np.isfinite(y);return np.linalg.lstsq(np.c_[x[q],np.ones(q.sum())],y[q],rcond=None)[0]
def load(w):
 h=pd.read_csv(R/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(R/'data/train'/f'{w}__typewell.csv').sort_values('TVT').dropna(subset=['TVT','GR']);k=h.TVT_input.notna().sum();tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();pt=np.interp(h.TVT_input.iloc[:k],tt,tg);co=affine(pt,h.GR.iloc[:k]);return h,k,tt,co[0]*tg+co[1]
def feat(hg,tg,sl,dz):
 r=np.nan_to_num(hg-tg);dr=np.gradient(r);c=[r,abs(r),r*r,dr,hg,tg,np.full(len(r),sl),dz]
 for w in [5,21,81]:c += [uniform_filter1d(r,w),np.sqrt(uniform_filter1d(r*r,w)+1e-8)]
 return np.column_stack(c)
# Cache fold-reusable supervised samples once.
cache=[]
for j,w in enumerate(W):
 h,k,tt,tg=load(w);ix=ixs[w];q=h.iloc[k:];true=y[ix]+h.TVT_input.iloc[k-1];hg=np.nan_to_num(q.GR.to_numpy(),nan=np.nanmedian(q.GR));dz=np.gradient(q.Z.to_numpy());take=np.arange(0,len(q),max(10,len(q)//250));xx=[];lab=[]
 for off in [0,-6,-3,-1.5,1.5,3,6]:xx.append(feat(hg,np.interp(true+off,tt,tg),float(np.mean(np.gradient(true))),dz)[take]);lab.append(np.full(len(take),off==0,int))
 cache.append((np.vstack(xx),np.concatenate(lab))); 
 if (j+1)%100==0:print('cache',j+1,flush=True)
for fo,(tr,va) in enumerate(folds):
 X=np.vstack([cache[i][0] for i in tr]);Y=np.concatenate([cache[i][1] for i in tr]);m=LGBMClassifier(n_estimators=500,num_leaves=15,max_depth=5,learning_rate=.035,min_child_samples=300,subsample=.8,colsample_bytree=.8,reg_lambda=30,verbosity=-1,n_jobs=4,random_state=1501).fit(X,Y)
 for i in va:
  w=W[i];h,k,tt,tg=load(w);ix=ixs[w];q=h.iloc[k:];f=np.linspace(0,1,len(ix));paths=np.array([center[ix]+a+b*f for a,b in labels]);hg=np.nan_to_num(q.GR.to_numpy(),nan=np.nanmedian(q.GR));dz=np.gradient(q.Z.to_numpy());take=np.arange(0,len(q),max(5,len(q)//400));scores=[]
  for p in paths:
   F=feat(hg,np.interp(float(h.TVT_input.iloc[k-1])+p,tt,tg),float(np.mean(np.gradient(p))),dz);pr=np.clip(m.predict_proba(F[take])[:,1],1e-5,1-1e-5);scores.append(np.mean(np.log(pr/(1-pr))))
  score=np.asarray(scores)-((labels[:,0]/4)**2+(labels[:,1]/4)**2);temp=np.std(score)*.5+1e-8;ww=np.exp(np.clip((score-score.max())/temp,-30,0));ww/=ww.sum();pred[ix]=ww@paths
 print('fold',fo,flush=True)
def rm(p,m=None):
 if m is None:m=np.ones(len(y),bool)
 return float(np.sqrt(np.mean((y[m]-p[m])**2)))
fm={w:f for f,(_,v) in enumerate(folds) for w in W[v]};rf=np.array([fm[w] for w in g]);fr=[{'fold':f,'baseline':rm(center,rf==f),'density':rm(pred,rf==f),'gain':rm(center,rf==f)-rm(pred,rf==f)} for f in range(5)];summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'locked':{'prior':1,'temperature':.5,'blend':1,'classifier_trees':500},'baseline':rm(center),'density':rm(pred),'gain':rm(center)-rm(pred),'folds':fr,'fold_wins':sum(x['gain']>0 for x in fr),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};np.savez_compressed(OUT/'oof.npz',groups=g,prediction=pred,correction=pred-center);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
