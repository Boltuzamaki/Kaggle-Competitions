"""Whole-well-cross-fitted heteroscedastic vertical->LWD GR emission pilot."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

R=Path(__file__).resolve().parents[1]
OUT=R/'exp/results/angle_conditional_emission_pilot_v1';OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True)
L=np.load(R/'exp/results/legal_level2_all_oof_v1/oof.npz',allow_pickle=True)
groups=S['groups'].astype(str); y=S['y']; accepted=S['accepted']; replacement=S['replacement']
center=.1*accepted+.9*replacement+L['anchored_difference']
W=np.array(sorted(set(groups))); folds=list(GroupKFold(5).split(W,groups=W))
wf={w:f for f,(_,va) in enumerate(folds) for w in W[va]}
cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)]
ixs={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}

def norm(x):
 x=np.asarray(x,float);lo,hi=np.nanpercentile(x,[1,99]);return np.clip((x-lo)/(hi-lo+1e-6),0,1)
def load(w):
 h=pd.read_csv(R/'data/train'/f'{w}__horizontal_well.csv');t=pd.read_csv(R/'data/train'/f'{w}__typewell.csv').dropna(subset=['TVT','GR']).sort_values('TVT')
 m=h.TVT_input.isna().to_numpy();md=h.MD.to_numpy(float)[m];hg=norm(pd.to_numeric(h.GR,errors='coerce').interpolate(limit_direction='both'))[m]
 tt=t.TVT.to_numpy(float);tg=norm(t.GR.to_numpy(float));step=np.median(np.diff(tt));d1=np.gradient(tg,tt);d2=np.gradient(d1,tt)
 return md,hg,tt,tg,d1,d2
cache={w:load(w) for w in W}
def feat(w,path):
 md,hg,tt,tg,d1,d2=cache[w];v=np.gradient(path,md);q0=np.interp(path,tt,tg);q1=np.interp(path,tt,d1);q2=np.interp(path,tt,d2)
 # Shape/angle conditioning; observed GR is never a feature, only emission target.
 return np.c_[q0,q1,q2,np.abs(q1),v,np.abs(v),np.gradient(v,md)]

rng=np.random.RandomState(2408); pilot=set(rng.choice(W,240,replace=False)); rows=[]
for fo in range(5):
 trainw=np.array([w for w in W if wf[w]!=fo]); valw=np.array([w for w in W if wf[w]==fo and w in pilot])
 X=[];Y=[]
 for w in trainw:
  ix=ixs[w];xx=feat(w,y[ix]);yy=cache[w][1];take=np.linspace(0,len(ix)-1,min(350,len(ix))).astype(int);X.append(xx[take]);Y.append(yy[take])
 X=np.concatenate(X);Y=np.concatenate(Y)
 mean=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=80,max_features=.8,n_jobs=-1,random_state=fo).fit(X,Y)
 resid=Y-mean.predict(X);var=ExtraTreesRegressor(n_estimators=60,min_samples_leaf=120,max_features=.8,n_jobs=-1,random_state=100+fo).fit(X,np.log(resid*resid+0.0025))
 for w in valw:
  ix=ixs[w];obs=cache[w][1]; cand={'center':center[ix],'accepted':accepted[ix],'replacement':replacement[ix]};scores={}
  for n,p in cand.items():
   xx=feat(w,p);mu=mean.predict(xx);vv=np.exp(var.predict(xx));scores[n]=float(np.mean((obs-mu)**2/vv+np.log(vv)))
  pick=min(scores,key=scores.get);e0=float(np.sum((y[ix]-center[ix])**2));ep=float(np.sum((y[ix]-cand[pick])**2))
  rows.append({'fold':fo,'well':w,'rows':len(ix),'pick':pick,'base_sse':e0,'picked_sse':ep,**{f'nll_{k}':v for k,v in scores.items()}})
df=pd.DataFrame(rows);df.to_csv(OUT/'by_well.csv',index=False)
base=float(np.sqrt(df.base_sse.sum()/df.rows.sum()));picked=float(np.sqrt(df.picked_sse.sum()/df.rows.sum()))
summary={'wells':len(df),'rows':int(df.rows.sum()),'base':base,'picked':picked,'gain':base-picked,'well_wins':int((df.picked_sse<df.base_sse).sum()),'picks':df['pick'].value_counts().to_dict(),'folds':[],'protocol':'outer whole-well GKF; emission models exclude validation fold; 240-well seed-2408 pilot; candidates fixed before likelihood'}
for f,d in df.groupby('fold'):summary['folds'].append({'fold':int(f),'base':float(np.sqrt(d.base_sse.sum()/d.rows.sum())),'picked':float(np.sqrt(d.picked_sse.sum()/d.rows.sum())),'wells':len(d)})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
