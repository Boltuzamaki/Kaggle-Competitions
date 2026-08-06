"""Organizer-consistent episodic Student-residual augmentation pilot."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_episodic_v10';OUT.mkdir(parents=True,exist_ok=True);L=128
v=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');allW=v['wells'].astype(str);sel=np.sort(np.random.RandomState(1001).choice(len(allW),200,False));W=allW[sel];true=v['true'][sel];lens=v['length'][sel]
s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);groups=s['groups'].astype(str);base=.1*s['accepted']+.9*s['replacement'];yy=s['y']
def rs(x):return np.interp(np.linspace(0,1,L),np.linspace(0,1,len(x)),np.asarray(x,float))
def feat(h,end,start,future_end,real):
 q=h.iloc[:end];tv=q.TVT.to_numpy();z=q.Z.to_numpy();md=q.MD.to_numpy();er=np.r_[0,np.diff(tv)-np.diff(z)];out=[]
 for x in [er,np.cumsum(er),np.gradient(tv)/(np.gradient(md)+1e-6),np.gradient(tv+z)/(np.gradient(md)+1e-6)]:
  a=rs(x);out.extend(a.reshape(16,8).mean(1));out.extend([a.mean(),a.std(),a[-1],np.mean(a[-16:]),np.std(a[-16:])])
 f=h.iloc[start:future_end]
 for c in ['MD','Z','X','Y','GR']:
  a=rs(f[c]);a=a-a[0] if c!='GR' else a;out.extend(a.reshape(16,8).mean(1));out.extend([a.mean(),a.std(),a[-1]-a[0]])
 out.extend([len(q),len(f),real]);return np.nan_to_num(out)
examples={};realX=[]
for w in W:
 h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv');ps=int(h.TVT_input.notna().sum());m=groups==w;n=m.sum();assert n>20
 rx=feat(h,ps,ps,len(h),1);realX.append(rx);ep=[]
 for frac in [.25,.5,.7]:
  j=max(5,min(n-10,int(n*frac)));cut=ps+j; ep.append((feat(h,cut,cut,len(h),0),rs(yy[m][j:]-base[m][j:])))
 examples[w]=ep
realX=np.asarray(realX);pred0=np.zeros_like(true);pred1=np.zeros_like(true);folds=list(GroupKFold(5).split(realX,groups=W))
for fo,(tr,va) in enumerate(folds):
 cp=PCA(20,random_state=44).fit(true[tr]);lat=cp.transform(true[tr]);sc=StandardScaler().fit(realX[tr]);a=sc.transform(realX[tr]);q=sc.transform(realX[va])
 m=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=6,max_features=.65,n_jobs=-1,random_state=10).fit(a,lat);pred0[va]=cp.inverse_transform(m.predict(q))
 ex=[];ey=[];wt=[]
 for i in tr:
  ex.append(realX[i]);ey.append(true[i]);wt.append(4.)
  for x,y in examples[W[i]]:ex.append(x);ey.append(y);wt.append(1.)
 ex=np.asarray(ex);ey=np.asarray(ey);sc2=StandardScaler().fit(ex);cp2=PCA(20,random_state=44).fit(ey,sample_weight=None) if False else PCA(20,random_state=44).fit(ey);lat2=cp2.transform(ey)
 mm=ExtraTreesRegressor(n_estimators=600,min_samples_leaf=10,max_features=.65,n_jobs=-1,random_state=11).fit(sc2.transform(ex),lat2,sample_weight=wt);pred1[va]=cp2.inverse_transform(mm.predict(sc2.transform(realX[va])));print('fold',fo,flush=True)
def score(p):return float(np.sqrt(np.average(np.mean((true-p)**2,1),weights=lens)))
out={'wells':200,'zero':score(0*true),'real_only':score(pred0),'episodic':score(pred1),'gain':score(pred0)-score(pred1),'gate':score(pred0)-score(pred1)>.05,'unit_check':'targets are target-immutable_student on identical suffix rows; causal features end before continuation'}
np.savez_compressed(OUT/'pilot_200.npz',wells=W,real=pred0,episodic=pred1,true=true,length=lens);out['sha256']=hashlib.sha256((OUT/'pilot_200.npz').read_bytes()).hexdigest();(OUT/'pilot_200.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
