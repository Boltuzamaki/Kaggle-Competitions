"""Deterministic random-convolution features over causal prefix error paths."""
from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_rocket_v8';OUT.mkdir(parents=True,exist_ok=True)
v=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=v['wells'].astype(str);C=v['true'];lens=v['length'];Q=v['query'];b=np.load(ROOT/'exp/results/prefix_backtest_moe/backtest_features.npz',allow_pickle=True);bm={w:i for i,w in enumerate(b['wells'].astype(str))};BT=b['features'][[bm[w] for w in W]]
n=int(sys.argv[1]) if len(sys.argv)>1 else 200
if n<len(W):ix=np.sort(np.random.RandomState(880).choice(len(W),n,False));W,C,lens,Q,BT=W[ix],C[ix],lens[ix],Q[ix],BT[ix]
def sequences(w):
 h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv',usecols=['MD','Z','TVT_input']);h=h[h.TVT_input.notna()];md=h.MD.to_numpy();z=h.Z.to_numpy();tv=h.TVT_input.to_numpy();S=tv+z;dmd=np.r_[1,np.diff(md)];ds=np.r_[0,np.diff(S)];dz=np.r_[0,np.diff(z)];dt=np.r_[0,np.diff(tv)];chs=[dt-dz,ds]
 for win in [10,25,50,100,200,400,800,1200,1600]:
  sl=pd.Series(ds/dmd).rolling(win,min_periods=max(3,win//5)).median().shift(1).fillna(0).to_numpy();chs.append(dt-(sl*dmd-dz))
 return np.stack([np.interp(np.linspace(0,1,256),np.linspace(0,1,len(x)),x) for x in chs])
rng=np.random.RandomState(881);kern=[]
for _ in range(192):
 ch=rng.randint(11);ln=int(rng.choice([7,9,15]));d=int(rng.choice([1,2,4,8]));k=rng.randn(ln);k-=k.mean();k/=np.linalg.norm(k)+1e-9;kern.append((ch,d,k))
def rocket(s):
 out=[]
 for ch,d,k in kern:
  x=s[ch];span=d*(len(k)-1)+1
  if span>len(x):out.extend([0,0]);continue
  q=np.array([np.dot(x[i:i+span:d],k) for i in range(len(x)-span+1)]);out.extend([(q>0).mean(),q.max()])
 return out
R=np.asarray([rocket(sequences(w)) for w in W],np.float32);base=np.c_[Q,BT];sets={'control':base,'rocket_add':np.c_[base,R],'rocket_only':R};pred={k:np.zeros_like(C) for k in sets};folds=list(GroupKFold(5).split(base,groups=W))
for fo,(tr,va) in enumerate(folds):
 cp=PCA(24,random_state=44).fit(C[tr]);lat=cp.transform(C[tr])
 for name,X in sets.items():
  sc=StandardScaler().fit(X[tr]);aa=sc.transform(X[tr]);bb=sc.transform(X[va]);pc=PCA(min(48,len(tr)-1),whiten=True,random_state=43).fit(aa)
  m=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=max(5,12*n//765),max_features=.6,n_jobs=-1,random_state=45).fit(pc.transform(aa),lat);pred[name][va]=cp.inverse_transform(m.predict(pc.transform(bb)))
 print('fold',fo,flush=True)
def score(p):return float(np.sqrt(np.average(np.mean((C-p)**2,1),weights=lens)))
scores={k:score(p) for k,p in pred.items()};out={'wells':len(W),'scores':scores,'gain_vs_control':scores['control']-scores['rocket_add'],'gate':scores['control']-scores['rocket_add']>.05}
np.savez_compressed(OUT/f'pilot_{n}.npz',wells=W,pred=pred['rocket_add'],control=pred['control'],true=C,length=lens,rocket=R);out['sha256']=hashlib.sha256((OUT/f'pilot_{n}.npz').read_bytes()).hexdigest();(OUT/f'pilot_{n}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
