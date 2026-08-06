"""Strict outer-GKF functional BLUP from observed prefix errors to suffix residual."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold,KFold
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz'
OUT=ROOT/'exp/results/functional_gaussian_prefix_conditioning_v1';OUT.mkdir(parents=True,exist_ok=True)
Z=np.load(SRC,allow_pickle=True);W=Z['wells'].astype(str);C=Z['true'];lens=Z['length'];L=128
def rs(x):return np.interp(np.linspace(0,1,L),np.linspace(0,1,len(x)),np.nan_to_num(x))
P=[]
for w in W:
 h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv',usecols=['MD','Z','TVT_input','GR']);h=h[h.TVT_input.notna()]
 tv=h.TVT_input.to_numpy();z=h.Z.to_numpy();md=h.MD.to_numpy();er=np.r_[0,np.diff(tv)-np.diff(z)]
 cum=np.cumsum(er);cum-=cum[-1] # observed structural error, anchored at boundary
 slope=np.gradient(tv)/(np.gradient(md)+1e-6)-np.gradient(z)/(np.gradient(md)+1e-6)
 # Multi-scale error-function observations, all strictly before prediction start.
 P.append(np.r_[rs(cum),rs(er),rs(slope)])
P=np.asarray(P);outer=list(GroupKFold(5).split(P,groups=W));pred=np.zeros_like(C);chosen=[]

def fit_predict(tr,va,rank,lam):
 pm=P[tr].mean(0);ps=P[tr].std(0)+1e-6; X=(P[tr]-pm)/ps; Xv=(P[va]-pm)/ps
 xp=PCA(rank,random_state=401).fit(X);U=xp.transform(X);Uv=xp.transform(Xv)
 cm=C[tr].mean(0);Y=C[tr]-cm
 # Gaussian conditional mean / ridge-stabilized empirical covariance BLUP.
 B=np.linalg.solve(U.T@U+lam*np.eye(rank),U.T@Y)
 return cm+Uv@B

for fo,(tr,va) in enumerate(outer):
 best=None
 inner=list(KFold(4,shuffle=True,random_state=410+fo).split(tr))
 for rank in [8,16,32,48]:
  for lam in [1,10,50,200,1000]:
   se=n=0
   for a,b in inner:
    pp=fit_predict(tr[a],tr[b],rank,lam);se+=float(((C[tr[b]]-pp)**2*lens[tr[b],None]).sum());n+=int(lens[tr[b]].sum())*L
   score=np.sqrt(se/n)
   if best is None or score<best[0]:best=(score,rank,lam)
 pred[va]=fit_predict(tr,va,best[1],best[2]);chosen.append({'fold':fo,'inner_rmse':best[0],'rank':best[1],'lambda':best[2]});print(chosen[-1],flush=True)

s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=s['y'];groups=s['groups'].astype(str);base=.1*s['accepted']+.9*s['replacement']
cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)];ixs={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}
def row(A):
 o=np.empty_like(y)
 for w,c in zip(W,A):
  ix=ixs[w];o[ix]=np.interp(np.linspace(0,1,len(ix)),np.linspace(0,1,len(c)),c)
 return o
rp=row(pred);et=row(Z['et']);wf={w:f for f,(_,v) in enumerate(outer) for w in W[v]};rf=np.array([wf[w] for w in groups])
def rm(p,m=None):
 if m is None:m=np.ones(len(y),bool)
 return float(np.sqrt(np.mean((y[m]-p[m])**2)))
grid=[]
for e in [0,1]:
 for a in [-.2,-.1,-.05,0,.025,.05,.1,.2,.35,.5,.75,1]:
  p=base+e*et+a*rp;fs=[rm(p,rf==f) for f in range(5)];grid.append(dict(et=e,alpha=a,rmse=rm(p),**{f'f{f}':q for f,q in enumerate(fs)}))
d=pd.DataFrame(grid).sort_values('rmse');d.to_csv(OUT/'blend_addone_grid.csv',index=False);np.savez_compressed(OUT/'predictions.npz',wells=W,curve=pred,prefix=P)
summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'baseline':rm(base),'et':rm(base+et),'best':d.iloc[0].to_dict(),'chosen':chosen,
 'blup_et_residual_corr':float(np.corrcoef(rp,y-(base+et))[0,1]),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
