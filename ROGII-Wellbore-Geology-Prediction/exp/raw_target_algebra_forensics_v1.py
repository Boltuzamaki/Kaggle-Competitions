"""Raw-row algebra/precision/mask forensics; markers are diagnostic labels only."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

R=Path(__file__).resolve().parents[1];SRC=R/'exp/results/legal_level2_all_oof_v1/oof.npz';OUT=R/'exp/results/raw_target_algebra_forensics_v1';OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];g=S['groups'].astype(str);center=.1*S['accepted']+.9*S['replacement']+np.load(SRC,allow_pickle=True)['anchored_difference'];err=y-center
W=np.sort(np.unique(g));cuts=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1,len(g)];ixs={g[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])};M=['ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA'];features=[];markers=[];T=[];lens=[]
def stats(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 if not len(v):return [0]*8
 d=np.diff(v);return [v[0],v[-1],np.mean(v),np.std(v),np.mean(d) if len(d) else 0,np.std(d) if len(d) else 0,np.mean(abs(v-np.round(v,2))),len(np.unique(np.round(d,4)))/max(1,len(d))]
for wi,w in enumerate(W):
 ix=ixs[w];h=pd.read_csv(R/'data/train'/f'{w}__horizontal_well.csv');k=int(h.TVT_input.notna().sum());n=len(h);q=h.iloc[k:];p=h.iloc[:k];f=[n,k,n-k,k/n,wi/len(W),int(w[:4],16)/65535]
 # Boundary, gap, rounding, finite-difference and missingness signatures.
 for c in ['MD','X','Y','Z','GR']:
  f+=stats(h[c]);f+=stats(q[c]);
 for c in ['TVT_input','Z']:
  a=p[c].to_numpy();
  for win in [25,75,200,500]:f+=stats(a[-min(win,len(a)):])
 tv=p.TVT_input.to_numpy();zz=p.Z.to_numpy();md=p.MD.to_numpy();surf=tv+zz
 for win in [25,75,200,500,1500]:
  z0=surf[-min(win,len(surf)):];x0=md[-len(z0):]-md[-1];co=np.polyfit(x0,z0-z0[-1],min(2,len(z0)-1));f+=list(np.pad(co,(3-len(co),0)))
 gr=h.GR.to_numpy();f += [np.mean(~np.isfinite(gr)),np.mean(~np.isfinite(gr[:k])),np.mean(~np.isfinite(gr[k:])),np.mean(np.diff(np.flatnonzero(np.isfinite(gr))) if np.isfinite(gr).sum()>1 else [0])]
 # Exact mask boundary jumps/gaps.
 for c in ['MD','X','Y','Z','GR']:
  a=h[c].to_numpy();f += [a[k]-a[k-1] if np.isfinite(a[k]) and np.isfinite(a[k-1]) else 0]
 features.append(f)
 t=np.linspace(0,1,len(ix));A=np.c_[np.ones(len(ix)),t];T.append(np.linalg.lstsq(A,err[ix],rcond=None)[0]);lens.append(len(ix))
 mm=[]
 for c in M:
  a=q[c].to_numpy();co=np.polyfit(t,a,1);mm += [a[0],co[0],np.std(a-np.polyval(co,t))]
 markers.append(mm)
X=np.nan_to_num(np.asarray(features));T=np.asarray(T);Ymark=np.asarray(markers);lens=np.asarray(lens);folds=list(KFold(5,shuffle=True,random_state=1001).split(W));direct=np.zeros_like(T);proxy=np.zeros_like(T);oracle=np.zeros_like(T);mpred=np.zeros_like(Ymark)
for fo,(tr,va) in enumerate(folds):
 sc=StandardScaler().fit(X[tr]);a=sc.transform(X[tr]);b=sc.transform(X[va]);direct[va]=ExtraTreesRegressor(n_estimators=700,min_samples_leaf=12,max_features=.7,n_jobs=4,random_state=1002+fo).fit(a,T[tr]).predict(b)
 # Inner-crossfit legal prediction of train-only marker labels.
 inner=np.zeros_like(Ymark[tr]);
 for aa,bb in KFold(4,shuffle=True,random_state=1100+fo).split(tr):inner[bb]=ExtraTreesRegressor(n_estimators=350,min_samples_leaf=10,max_features=.7,n_jobs=4,random_state=1200+fo).fit(a[aa],Ymark[tr[aa]]).predict(a[bb])
 mm=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=10,max_features=.7,n_jobs=4,random_state=1300+fo).fit(a,Ymark[tr]);mv=mm.predict(b);mpred[va]=mv
 ms=StandardScaler().fit(inner);proxy[va]=Ridge(100).fit(ms.transform(inner),T[tr]).predict(ms.transform(mv))
 oracle[va]=Ridge(100).fit(Ymark[tr],T[tr]).predict(Ymark[va]);print('fold',fo,flush=True)
def row(P):
 o=np.zeros(len(y))
 for w,c in zip(W,P):
  ix=ixs[w];o[ix]=c[0]+c[1]*np.linspace(0,1,len(ix))
 return o
def rm(p):return float(np.sqrt(np.mean((y-p)**2)))
def met(P):
 return {'rmse':rm(center+row(P)),'datum_r2':float(1-np.average((T[:,0]-P[:,0])**2,weights=lens)/np.average((T[:,0]-np.average(T[:,0],weights=lens))**2,weights=lens)),'slope_r2':float(1-np.average((T[:,1]-P[:,1])**2,weights=lens)/np.average((T[:,1]-np.average(T[:,1],weights=lens))**2,weights=lens))}
mc=np.corrcoef(np.c_[T,Ymark].T)[:2,2:];mark_r2=[float(1-np.average((Ymark[:,j]-mpred[:,j])**2,weights=lens)/np.average((Ymark[:,j]-np.average(Ymark[:,j],weights=lens))**2,weights=lens)) for j in range(Ymark.shape[1])]
summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'baseline':rm(center),'direct_legal':met(direct),'predicted_marker_proxy':met(proxy),'marker_oracle_invalid_diagnostic':met(oracle),'max_abs_target_marker_corr':float(np.max(abs(mc))),'marker_legal_r2_median':float(np.median(mark_r2)),'marker_legal_r2_max':float(np.max(mark_r2)),'n_legal_features':X.shape[1],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/'oof.npz',wells=W,target=T,direct=direct,proxy=proxy,marker_oracle=oracle,marker_pred=mpred);print(json.dumps(summary,indent=2))
