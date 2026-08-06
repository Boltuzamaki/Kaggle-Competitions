"""Fold-safe deterministic boundary/horizon calibration of immutable v6 OOF curves."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold,KFold

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'exp/results/generative_curve_prior_expert_errors_v6/oof.npz'
OUT=ROOT/'exp/results/v6_physical_boundary_calibration_v1';OUT.mkdir(parents=True,exist_ok=True)
Z=np.load(SRC,allow_pickle=True);W=Z['wells'].astype(str);P=Z['pred'];C=Z['true'];lens=Z['length'];G=np.linspace(0,1,128)
s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=s['y'];groups=s['groups'].astype(str);base=.1*s['accepted']+.9*s['replacement']
cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)];ixs={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}
# Legal boundary correction: extrapolate the first 32 base suffix predictions back to last observed MD.
A=[]
for w in W:
 b=base[ixs[w]]
 h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv',usecols=['MD','TVT_input']);k=int(h.TVT_input.notna().sum())
 md=h.MD.to_numpy(); n=min(32,len(b)); co=np.polyfit(md[k:k+n]-md[k],b[:n],1)
 base_boundary=np.polyval(co,md[k-1]-md[k]);A.append(float(-base_boundary))
A=np.asarray(A); folds=list(GroupKFold(5).split(W,groups=W));cal=np.zeros_like(P);choices=[]

def design(p,a,tau,sig,gamma):
 q=gaussian_filter1d(p,sig,axis=1) if sig else p.copy(); taper=(1-G)**gamma
 e=np.exp(-G/tau); # all candidate curves equal observed anchor exactly at G=0
 return np.stack([(1-e)*q*taper,(1-e)*q*G*taper],axis=2), a[:,None]*e
def coef_fit(tr,tau,sig,gamma):
 X,anc=design(P[tr],A[tr],tau,sig,gamma);Y=C[tr]-anc
 ww=np.repeat(lens[tr],128);xx=X.reshape(-1,2);yy=Y.ravel(); sw=np.sqrt(ww/ww.mean())
 return np.linalg.solve((xx*sw[:,None]).T@(xx*sw[:,None])+20*np.eye(2),(xx*sw[:,None]).T@(yy*sw))
def apply(ix,co,tau,sig,gamma):
 X,anc=design(P[ix],A[ix],tau,sig,gamma);return anc+X@co

for fo,(tr,va) in enumerate(folds):
 best=None; inner=list(KFold(4,shuffle=True,random_state=601+fo).split(tr))
 for tau in [.03,.08,.15,.3]:
  for sig in [0,2,5]:
   for gamma in [0,.5,1]:
    se=den=0
    for x,z in inner:
     co=coef_fit(tr[x],tau,sig,gamma);pp=apply(tr[z],co,tau,sig,gamma)
     se+=float((((C[tr[z]]-pp)**2)*lens[tr[z],None]).sum());den+=int(lens[tr[z]].sum())*128
    score=np.sqrt(se/den)
    if best is None or score<best[0]:best=(score,tau,sig,gamma)
 co=coef_fit(tr,*best[1:]);cal[va]=apply(va,co,*best[1:]);choices.append({'fold':fo,'inner_rmse':best[0],'tau':best[1],'sigma':best[2],'gamma':best[3],'coef':co.tolist()});print(choices[-1],flush=True)

def row(Q):
 o=np.empty_like(y)
 for w,c in zip(W,Q):
  ix=ixs[w];o[ix]=np.interp(np.linspace(0,1,len(ix)),G,c)
 return o
rp=row(P);rc=row(cal);wf={w:f for f,(_,v) in enumerate(folds) for w in W[v]};rf=np.array([wf[w] for w in groups])
def rm(p,m=None):
 if m is None:m=np.ones(len(y),bool)
 return float(np.sqrt(np.mean((y[m]-p[m])**2)))
rows=[]
for name,r in [('raw',rp),('calibrated',rc)]:
 for a in [.5,.75,1,1.25]:
  p=base+a*r;fs=[rm(p,rf==f) for f in range(5)];rows.append(dict(model=name,alpha=a,rmse=rm(p),**{f'f{f}':q for f,q in enumerate(fs)}))
d=pd.DataFrame(rows).sort_values('rmse');d.to_csv(OUT/'scores.csv',index=False);np.savez_compressed(OUT/'oof.npz',wells=W,calibrated=cal,anchor=A)
summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'baseline':rm(base),'best':d.iloc[0].to_dict(),'choices':choices,
 'anchor_abs_median':float(np.median(abs(A))),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
