"""Calibrated GR patch density-ratio emission; disjoint train/pilot/confirm."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.ndimage import uniform_filter1d
from lightgbm import LGBMClassifier

R=Path(__file__).resolve().parents[1];SRC=R/'exp/results/legal_level2_all_oof_v1/oof.npz';OUT=R/'exp/results/supervised_density_ratio_gr_emission_v1';OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];g=S['groups'].astype(str);center=.1*S['accepted']+.9*S['replacement']+np.load(SRC,allow_pickle=True)['anchored_difference'];cuts=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1,len(g)];ixs={g[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])};allw=set(ixs);pilot=[w for w in pd.read_csv(R/'exp/results/setchell_complete_path/pilot_wells.csv').well.astype(str) if w in allw];confirm=[w for w in pd.read_csv(R/'exp/results/setchell_complete_path/confirmation_wells.csv').well.astype(str) if w in allw];train=sorted(allw-set(pilot)-set(confirm))
def affine(x,y):
 q=np.isfinite(x)&np.isfinite(y);return np.linalg.lstsq(np.c_[x[q],np.ones(q.sum())],y[q],rcond=None)[0]
def load(w):
 h=pd.read_csv(R/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(R/'data/train'/f'{w}__typewell.csv').sort_values('TVT').dropna(subset=['TVT','GR']);k=h.TVT_input.notna().sum();tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();pt=np.interp(h.TVT_input.iloc[:k],tt,tg);co=affine(pt,h.GR.iloc[:k]);return h,tw,k,tt,co[0]*tg+co[1]
def feats(hg,tg,slope,dz):
 r=np.nan_to_num(hg-tg);dr=np.gradient(r);cols=[r,abs(r),r*r,dr,hg,tg,np.full(len(r),slope),dz]
 for win in [5,21,81]:cols += [uniform_filter1d(r,win),np.sqrt(uniform_filter1d(r*r,win)+1e-8)]
 return np.column_stack(cols)
# Completely disjoint supervised emission training.
XX=[];YY=[]
for j,w in enumerate(train):
 h,tw,k,tt,tg=load(w);q=h.iloc[k:];true=y[ixs[w]]+h.TVT_input.iloc[k-1];hg=np.nan_to_num(q.GR.to_numpy(),nan=np.nanmedian(q.GR));dz=np.gradient(q.Z.to_numpy());take=np.arange(0,len(q),max(10,len(q)//250))
 for off in [0,-6,-3,-1.5,1.5,3,6]:
  ref=np.interp(true+off,tt,tg);F=feats(hg,ref,float(np.mean(np.gradient(true))),dz);XX.append(F[take]);YY.append(np.full(len(take),off==0,int))
 if (j+1)%100==0:print('train features',j+1,flush=True)
XX=np.vstack(XX);YY=np.concatenate(YY);model=LGBMClassifier(n_estimators=500,num_leaves=15,max_depth=5,learning_rate=.035,min_child_samples=300,subsample=.8,colsample_bytree=.8,reg_lambda=30,verbosity=-1,n_jobs=4,random_state=1501).fit(XX,YY)
datums=np.array([-4,-2,0,2,4]);tilts=np.array([-4,-2,0,2,4]);labels=np.array([(a,b) for a in datums for b in tilts])
def store(wells,tag):
 st={}
 for j,w in enumerate(wells):
  h,tw,k,tt,tg=load(w);ix=ixs[w];q=h.iloc[k:];f=np.linspace(0,1,len(ix));paths=np.array([center[ix]+a+b*f for a,b in labels]);hg=np.nan_to_num(q.GR.to_numpy(),nan=np.nanmedian(q.GR));dz=np.gradient(q.Z.to_numpy());take=np.arange(0,len(q),max(5,len(q)//400));scores=[]
  for p in paths:
   ref=np.interp(float(h.TVT_input.iloc[k-1])+p,tt,tg);F=feats(hg,ref,float(np.mean(np.gradient(p))),dz);prob=np.clip(model.predict_proba(F[take])[:,1],1e-5,1-1e-5);scores.append(float(np.mean(np.log(prob/(1-prob)))))
  st[w]=(y[ix],center[ix],paths,np.asarray(scores));
  if (j+1)%30==0:print(tag,j+1,flush=True)
 return st
def evaluate(st,cfg):
 yy=[];bb=[];pp=[];wins=0
 for yt,c,p,s in st.values():
  score=s-cfg['prior']*((labels[:,0]/4)**2+(labels[:,1]/4)**2);temp=np.std(score)*cfg['temp']+1e-8;ww=np.exp(np.clip((score-score.max())/temp,-30,0));ww/=ww.sum();x=(1-cfg['blend'])*c+cfg['blend']*(ww@p);wins+=np.mean((yt-x)**2)<np.mean((yt-c)**2);yy.append(yt);bb.append(c);pp.append(x)
 yy=np.concatenate(yy);bb=np.concatenate(bb);pp=np.concatenate(pp);rm=lambda a,b:float(np.sqrt(np.mean((a-b)**2)));return rm(yy,pp),rm(yy,bb),wins/len(st)
ps=store(pilot,'pilot');grid=[]
for prior in [.01,.05,.2,1]:
 for temp in [.25,.5,1,2]:
  for blend in [.1,.2,.35,.5,1]:
   c={'prior':prior,'temp':temp,'blend':blend};a,b,w=evaluate(ps,c);grid.append({**c,'rmse':a,'baseline':b,'win_rate':w})
d=pd.DataFrame(grid).sort_values('rmse');d.to_csv(OUT/'pilot_grid.csv',index=False);lock={k:float(d.iloc[0][k]) for k in ['prior','temp','blend']};cs=store(confirm,'confirmation');a,b,w=evaluate(cs,lock);gain=b-a;summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'train_wells':len(train),'train_samples':len(YY),'pilot_baseline':float(d.iloc[0].baseline),'pilot':float(d.iloc[0].rmse),'locked':lock,'confirmation_baseline':b,'confirmation':a,'gain':gain,'win_rate':w,'passed':bool(gain>=.03 and w>=.55),'gate':'>=.03 and >=55% wins','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
