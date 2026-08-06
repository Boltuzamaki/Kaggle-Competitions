"""Full 765-well replay of the frozen PDE+GR configuration; no retuning."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold

R=Path(__file__).resolve().parents[1];SRC=R/'exp/results/legal_level2_all_oof_v1/oof.npz';OUT=R/'exp/results/boundary_kinematic_pde_gr_full_locked_v1';OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];g=S['groups'].astype(str);center=.1*S['accepted']+.9*S['replacement']+np.load(SRC,allow_pickle=True)['anchored_difference'];cuts=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1,len(g)];ixs={g[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])};W=np.sort(np.unique(g));scales=np.array([0,.02,.05,.1,.2]);pred=np.empty_like(y)
def plane(X,z):
 X=X-X[-1];z=z-z[-1];A=np.c_[X,np.ones(len(X))];keep=np.ones(len(X),bool)
 for _ in range(4):
  co=np.linalg.solve(A[keep].T@A[keep]+np.diag([100,100,1e-6]),A[keep].T@z[keep]);r=z-A@co;mad=1.4826*np.median(abs(r-np.median(r)))+1e-5;keep=abs(r-np.median(r))<3*mad
 return co
def affine(x,y):
 q=np.isfinite(x)&np.isfinite(y);return np.linalg.lstsq(np.c_[x[q],np.ones(q.sum())],y[q],rcond=None)[0]
for j,w in enumerate(W):
 ix=ixs[w];h=pd.read_csv(R/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(R/'data/train'/f'{w}__typewell.csv').sort_values('TVT').dropna(subset=['TVT','GR']);k=h.TVT_input.notna().sum();q=h.iloc[k:];last=h.iloc[k-1];p=h.iloc[max(0,k-200):k];co=plane(p[['X','Y']].to_numpy(),(p.TVT_input+p.Z).to_numpy());dxy=q[['X','Y']].to_numpy()-p[['X','Y']].to_numpy()[-1];phys=dxy@co[:2]-(q.Z.to_numpy()-last.Z);corr=np.clip(phys-center[ix],-25,25);paths=center[ix][None]+scales[:,None]*corr
 tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();pt=np.interp(h.TVT_input.iloc[:k],tt,tg);ac=affine(pt,h.GR.iloc[:k].to_numpy());pg=ac[0]*np.interp(float(last.TVT_input)+paths,tt,tg)+ac[1];res=q.GR.to_numpy()[None]-pg;pr=h.GR.iloc[:k].to_numpy()-(ac[0]*pt+ac[1]);sig=max(20,1.4826*np.nanmedian(abs(pr-np.nanmedian(pr))));loss=np.nanmean(2*np.log1p((res/sig)**2/3),axis=1);score=-loss-.02*(scales/.1)**2;temp=np.std(score)*.35+1e-8;ww=np.exp(np.clip((score-score.max())/temp,-30,0));ww/=ww.sum();pred[ix]=ww@paths
 if (j+1)%100==0:print(j+1,flush=True)
def rm(p,m=None):
 if m is None:m=np.ones(len(y),bool)
 return float(np.sqrt(np.mean((y[m]-p[m])**2)))
folds=list(GroupKFold(5).split(W,groups=W));fm={w:f for f,(_,v) in enumerate(folds) for w in W[v]};rf=np.array([fm[w] for w in g]);fr=[{'fold':f,'baseline':rm(center,rf==f),'pde':rm(pred,rf==f),'gain':rm(center,rf==f)-rm(pred,rf==f)} for f in range(5)]
summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'locked_config':{'window':200,'ridge':100,'nu':3,'prior':.02,'temperature':.35,'scales':scales.tolist()},'baseline':rm(center),'pde':rm(pred),'gain':rm(center)-rm(pred),'folds':fr,'fold_wins':sum(x['gain']>0 for x in fr),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};np.savez_compressed(OUT/'oof.npz',groups=g,prediction=pred,correction=pred-center);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
