"""Strict supervised Mahalanobis embedding for analogue curve retrieval."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1];V6=ROOT/'exp/results/generative_curve_prior_expert_errors_v6/oof.npz'
OUT=ROOT/'exp/results/v6_supervised_metric_analogue_v1';OUT.mkdir(parents=True,exist_ok=True)
v=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=v['wells'].astype(str);C=v['true'];Q=v['query'];lens=v['length']
b=np.load(ROOT/'exp/results/prefix_backtest_moe/backtest_features.npz',allow_pickle=True);bm={w:i for i,w in enumerate(b['wells'].astype(str))};BT=b['features'][[bm[w] for w in W]];X=np.nan_to_num(np.c_[Q,BT])
folds=list(GroupKFold(5).split(X,groups=W));configs=[(8,300,4),(8,1000,8),(16,300,8),(16,1000,12),(24,1000,16)]
pred={c:np.zeros_like(C) for c in configs};conf={c:np.zeros(len(W)) for c in configs}
for fi,(tr,va) in enumerate(folds):
 sc=StandardScaler().fit(X[tr]);a=sc.transform(X[tr]);q=sc.transform(X[va]);fp=PCA(64,whiten=True,random_state=701).fit(a);a=fp.transform(a);q=fp.transform(q)
 for rank,reg,k in configs:
  cp=PCA(rank,whiten=True,random_state=702).fit(C[tr]);zt=cp.transform(C[tr])
  m=Ridge(alpha=reg).fit(a,zt);zq=m.predict(q)
  dd=((zq[:,None,:]-zt[None,:,:])**2).mean(2);nn=np.argpartition(dd,k,axis=1)[:,:k]
  for j,ii in enumerate(nn):
   d=np.sqrt(dd[j,ii]);temp=np.median(d)+1e-6;ww=np.exp(-d/temp);ww/=ww.sum();pred[(rank,reg,k)][va[j]]=ww@C[tr[ii]]
   conf[(rank,reg,k)][va[j]]=float(1/(1+d.mean()))
 print('fold',fi,flush=True)

s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=s['y'];groups=s['groups'].astype(str);base=.1*s['accepted']+.9*s['replacement'];z6=np.load(V6,allow_pickle=True)
cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)];ixs={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}
def row(A):
 o=np.empty_like(y)
 for w,c in zip(W,A):
  ix=ixs[w];o[ix]=np.interp(np.linspace(0,1,len(ix)),np.linspace(0,1,len(c)),c)
 return o
def rm(p):return float(np.sqrt(np.mean((y-p)**2)))
rv6=row(z6['pred']);grid=[]
for c,P in pred.items():
 rp=row(P)
 for gate in ['none','confidence']:
  if gate=='confidence':
   cg=conf[c];cg=(cg-np.quantile(cg,.1))/(np.quantile(cg,.9)-np.quantile(cg,.1)+1e-8);cg=np.clip(cg,0,1);rg=np.empty_like(y)
   for w,x in zip(W,cg):rg[ixs[w]]=x
  else:rg=1
  for v6w in [0,.5,1]:
   for a0 in [.1,.2,.35,.5,.75,1]:grid.append(dict(rank=c[0],reg=c[1],k=c[2],gate=gate,v6=v6w,alpha=a0,rmse=rm(base+v6w*rv6+a0*rp*rg)))
d=pd.DataFrame(grid).sort_values('rmse');d.to_csv(OUT/'grid.csv',index=False);best=d.iloc[0].to_dict();key=(int(best['rank']),int(best['reg']),int(best['k']))
np.savez_compressed(OUT/'oof.npz',wells=W,pred=pred[key],confidence=conf[key])
summary={'v6_sha256':hashlib.sha256(V6.read_bytes()).hexdigest(),'baseline':rm(base),'v6':rm(base+rv6),'best':best,'gate':'>0.05 improvement over v6 required for escalation','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
