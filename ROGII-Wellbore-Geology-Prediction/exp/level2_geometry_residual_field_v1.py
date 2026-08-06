"""Strict GKF smooth trajectory/prefix-state residual field around 8.086 L2."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

R=Path(__file__).resolve().parents[1];SRC=R/'exp/results/legal_level2_all_oof_v1/oof.npz';OUT=R/'exp/results/level2_geometry_residual_field_v1';OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];groups=S['groups'].astype(str);student=.1*S['accepted']+.9*S['replacement'];L2=np.load(SRC,allow_pickle=True)['anchored_difference'];center=student+L2;target=y-center
W=np.sort(np.unique(groups));cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)];ixs={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}
X=np.empty((len(y),15),np.float32);state=[];sample=[]
for wi,w in enumerate(W):
 ix=ixs[w];h=pd.read_csv(R/'data/train'/f'{w}__horizontal_well.csv',usecols=['MD','X','Y','Z','GR','TVT_input']);k=int(h.TVT_input.notna().sum());q=h.iloc[k:];md=q.MD.to_numpy();x=q.X.to_numpy();yy=q.Y.to_numpy();z=q.Z.to_numpy();gr=q.GR.to_numpy();t=(md-md[0])/max(md[-1]-md[0],1)
 dx=np.gradient(x);dy=np.gradient(yy);dz=np.gradient(z);dm=np.gradient(md)+1e-6;azi=np.unwrap(np.arctan2(dy,dx));inc=np.arctan2(np.hypot(dx,dy),-dz+1e-6);curv=np.gradient(azi)/dm;dip=np.gradient(z)/dm
 p=h.iloc[:k];pm=p.MD.to_numpy();pt=p.TVT_input.to_numpy();pz=p.Z.to_numpy();n=min(200,k);surf_slope=np.polyfit(pm[-n:]-pm[-1],(pt+pz)[-n:]-(pt.iloc[-1] if hasattr(pt,'iloc') else pt[-1]),1)[0] if n>5 else 0
 er=np.diff(pt)-np.diff(pz); em=float(np.mean(er[-min(200,len(er)):])) if len(er) else 0; es=float(np.std(er[-min(200,len(er)):])) if len(er) else 0
 st=np.array([x[0],yy[0],z[0],np.mean(np.cos(azi)),np.mean(np.sin(azi)),np.mean(inc),np.std(inc),np.mean(curv),np.std(curv),surf_slope,em,es,np.mean(gr),np.std(gr)])
 state.append(st); X[ix]=np.c_[t,t*t,np.sin(np.pi*t),np.cos(azi),np.sin(azi),inc,dip,curv,np.gradient(curv)/dm,gr, np.tile(st[9:],(len(ix),1))]
 step=max(1,len(ix)//300);sample.extend(ix[::step].tolist())
state=np.asarray(state);med=np.nanmedian(state,axis=0);state=np.nan_to_num(np.where(np.isfinite(state),state,med));sample=np.asarray(sample);folds=list(GroupKFold(5).split(W,groups=W));wf={w:f for f,(_,v) in enumerate(folds) for w in W[v]};rf=np.array([wf[w] for w in groups]);pred=np.zeros(len(y));field=np.zeros(len(y))
for fo,(trw,vaw) in enumerate(folds):
 trset=set(W[trw]);tr=np.array([i for i in sample if groups[i] in trset]);va=np.concatenate([ixs[w] for w in W[vaw]])
 m=LGBMRegressor(objective='huber',n_estimators=650,num_leaves=15,max_depth=5,learning_rate=.025,min_child_samples=250,subsample=.8,colsample_bytree=.75,reg_lambda=30,verbosity=-1,n_jobs=4,random_state=810+fo).fit(X[tr],target[tr]);field[va]=m.predict(X[va])
 # Smooth field within each validation well, then add outer-train analogue random curves.
 sc=StandardScaler().fit(state[trw]);a=sc.transform(state[trw]);b=sc.transform(state[vaw])
 for jj,i in enumerate(vaw):
  w=W[i];ix=ixs[w];fp=gaussian_filter1d(field[ix],20);field[ix]=fp
  dd=((a-b[jj])**2).mean(1);nn=np.argsort(dd)[:12];ww=np.exp(-np.sqrt(dd[nn])/(np.median(np.sqrt(dd[nn]))+1e-6));ww/=ww.sum();u=np.linspace(0,1,len(ix));re=np.zeros(len(ix))
  for wt,ti in zip(ww,trw[nn]):
   tx=ixs[W[ti]];rr=target[tx]-gaussian_filter1d(m.predict(X[tx]),20);re+=wt*np.interp(u,np.linspace(0,1,len(tx)),gaussian_filter1d(rr,20))
  pred[ix]=fp+re
 print('fold',fo,flush=True)
def rm(p,m=None):
 if m is None:m=np.ones(len(y),bool)
 return float(np.sqrt(np.mean((y[m]-p[m])**2)))
rows=[]
for kind,p0 in [('field',field),('field_analogue',pred)]:
 for a in [0,.05,.1,.2,.35,.5,.75,1]:
  p=center+a*p0;fs=[rm(p,rf==f) for f in range(5)];rows.append(dict(kind=kind,alpha=a,rmse=rm(p),**{f'f{f}':v for f,v in enumerate(fs)}))
d=pd.DataFrame(rows).sort_values('rmse');d.to_csv(OUT/'grid.csv',index=False);np.savez_compressed(OUT/'oof.npz',groups=groups,field=field,field_analogue=pred)
summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'baseline':rm(center),'best':d.iloc[0].to_dict(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
