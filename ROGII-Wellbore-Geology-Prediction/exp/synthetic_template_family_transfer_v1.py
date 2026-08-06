"""Affine-invariant raw motif audit and strict outer-train residual transfer."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/synthetic_template_family_transfer_v1';OUT.mkdir(parents=True,exist_ok=True)
z=np.load(ROOT/'exp/results/generative_curve_prior_expert_errors_v6/oof.npz',allow_pickle=True)
W=z['wells'].astype(str);C=z['true'].astype(float);V6=z['pred'].astype(float);L=z['length'].astype(float);N=128

def interp(a,n=N):
 a=np.asarray(a,float);x=np.linspace(0,1,len(a));ok=np.isfinite(a)
 if ok.sum()<2:return np.zeros(n)
 return np.interp(np.linspace(0,1,n),x[ok],a[ok])
def robust(a):
 a=np.asarray(a,float);m=np.median(a);s=np.subtract(*np.percentile(a,[75,25]));return np.clip((a-m)/(s+1e-6),-6,6)
def motif(w):
 h=pd.read_csv(ROOT/f'data/train/{w}__horizontal_well.csv',usecols=['MD','X','Y','Z','GR'])
 t=pd.read_csv(ROOT/f'data/train/{w}__typewell.csv',usecols=['TVT','GR']).sort_values('TVT')
 # XY translation/rotation/isotropic-scale invariant trajectory coordinates.
 xy=h[['X','Y']].to_numpy(float);xy-=xy[0];u=xy[-1]/(np.linalg.norm(xy[-1])+1e-9);v=np.array([-u[1],u[0]])
 along=xy@u;cross=xy@v;scale=max(np.ptp(along),np.ptp(cross),1.)
 traj=np.r_[interp(along/scale),interp(cross/scale),interp(robust(h.Z)),interp(robust(np.gradient(h.Z.to_numpy(float))))]
 # GR offset/gain invariant; missingness retained separately because it can reveal generator templates.
 hg=h.GR.to_numpy(float);miss=~np.isfinite(hg);hi=interp(hg);hgr=np.r_[robust(hi),robust(np.gradient(hi)),interp(miss.astype(float))]
 tg=interp(t.GR.to_numpy(float));tgr=np.r_[robust(tg),robust(np.gradient(tg))]
 return traj.astype(np.float32),hgr.astype(np.float32),tgr.astype(np.float32)

T,H,R=[],[],[]
for i,w in enumerate(W):
 a,b,c=motif(w);T.append(a);H.append(b);R.append(c)
 if (i+1)%100==0:print('loaded',i+1,flush=True)
T,H,R=map(np.stack,(T,H,R))

# Descriptive all-well nearest distances in each invariant block.
def nearest(A):
 A=StandardScaler().fit_transform(A);A=PCA(min(48,A.shape[0]-1),whiten=True,random_state=9301).fit_transform(A)
 D=cdist(A,A)/np.sqrt(A.shape[1]);np.fill_diagonal(D,np.inf);return D.min(1),D.argmin(1)
audit={}
for name,A in [('trajectory',T),('horizontal_gr',H),('typewell_gr',R),('combined',np.c_[T,H,R])]:
 d,j=nearest(A);audit[name]={'quantiles':{str(q):float(np.quantile(d,q)) for q in (0,.001,.005,.01,.025,.05,.1,.5,1)},
  'pairs_below_0p25':int((d<.25).sum()),'pairs_below_0p5':int((d<.5).sum())}

# Strict outer-fold embedding and neighbor retrieval. Preprocessing/PCA and donor
# curves are fitted/read only on outer-train wells. IDs and formations never enter X.
X=np.c_[T,H,R];splits=list(GroupKFold(5).split(X,groups=W));KMAX=16
don=np.zeros((len(W),KMAX),int);dist=np.zeros((len(W),KMAX));fold=np.full(len(W),-1,np.int8)
for f,(tr,va) in enumerate(splits):
 sc=StandardScaler().fit(X[tr]);A=sc.transform(X[tr]);B=sc.transform(X[va]);pc=PCA(96,whiten=True,random_state=9400+f).fit(A)
 A=pc.transform(A);B=pc.transform(B);D=cdist(B,A)/np.sqrt(A.shape[1]);jj=np.argpartition(D,KMAX-1,axis=1)[:,:KMAX]
 dd=np.take_along_axis(D,jj,1);o=np.argsort(dd,axis=1);jj=np.take_along_axis(jj,o,1);dd=np.take_along_axis(dd,o,1)
 don[va]=tr[jj];dist[va]=dd;fold[va]=f

def score(P,m=None):
 if m is None:m=np.ones(len(W),bool)
 return float(np.sqrt(np.average(np.mean((C[m]-P[m])**2,1),weights=L[m])))
grid=[];saved={}
for k in (1,2,4,8,16):
 for temp in (.1,.2,.4,.8,1.6):
  wt=np.exp(-(dist[:,:k]-dist[:,:1])/temp);wt/=wt.sum(1,keepdims=True)
  Q=np.einsum('nk,nkl->nl',wt,C[don[:,:k]])
  # Confidence gates use only raw-motif distance.
  for gate in (.35,.5,.7,1.,1.5,np.inf):
   conf=np.exp(-dist[:,0]/.5) if np.isinf(gate) else (dist[:,0]<=gate).astype(float)
   for alpha in (.05,.1,.2,.3,.5):
    P=V6+alpha*conf[:,None]*(Q-V6);key=(k,temp,gate,alpha);saved[key]=P
    grid.append({'k':k,'temp':temp,'gate':str(gate),'alpha':alpha,'coverage':float((conf>0).mean()),'rmse':score(P),
      'folds':[score(P,fold==f) for f in range(5)]})
best=min(grid,key=lambda q:q['rmse']);key=(best['k'],best['temp'],float(best['gate']),best['alpha']);P=saved[key]
base_folds=[score(V6,fold==f) for f in range(5)];best['fold_gains']=[a-b for a,b in zip(base_folds,best['folds'])];best['wins']=sum(x>0 for x in best['fold_gains'])
pair=pd.DataFrame({'well':W,'donor':W[don[:,0]],'distance':dist[:,0],
 'residual_curve_rmse':np.sqrt(np.mean((C-C[don[:,0]])**2,axis=1))}).sort_values('distance')
summary={'wells':len(W),'blocks':{'trajectory':T.shape[1],'horizontal_gr':H.shape[1],'typewell_gr':R.shape[1]},'audit':audit,
 'v6':score(V6),'nearest_distance_quantiles':{str(q):float(np.quantile(dist[:,0],q)) for q in (0,.01,.05,.1,.5,.9,1)},
 'outer_pairs_below':{str(q):int((dist[:,0]<q).sum()) for q in (.75,.8,.85,.9)},
 'best':best,'v6_folds':base_folds,'protocol':'strict outer GKF; fold-local raw-motif scaling/PCA; donor residual curves outer-train only; no IDs/formations/public predictions'}
np.savez_compressed(OUT/'oof.npz',wells=W,true=C,v6=V6,pred=P,length=L,fold=fold,donor=don,distance=dist)
pd.DataFrame(grid).sort_values('rmse').to_csv(OUT/'grid.csv',index=False);pair.to_csv(OUT/'nearest_pairs.csv',index=False);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
