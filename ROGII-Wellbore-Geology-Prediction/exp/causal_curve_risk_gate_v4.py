"""Nested cross-fitted risk shrinkage for immutable causal-prefix v1."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold,KFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_risk_v4';OUT.mkdir(parents=True,exist_ok=True)
z=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=z['wells'];C=z['true'];Q=z['query'];lens=z['length'].astype(float);L=C.shape[1]
outer=list(GroupKFold(5).split(Q,groups=W)); names=['full','ridge','et25','et50'];P={n:np.zeros_like(C) for n in names}; A={n:np.ones(len(C)) for n in names}
def fitcorr(ix,X):
 cp=PCA(24,random_state=44).fit(C[ix]); lat=cp.transform(C[ix]);m=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=12,max_features=.6,n_jobs=-1,random_state=45).fit(X,lat);return cp,m
def riskfeat(qpc,p,var,dist,n):
 return np.r_[qpc[:16],n,np.mean(p),np.std(p),p[0],p[-1],p[-1]-p[0],np.sqrt(np.mean(p*p)),np.mean(var),np.max(var),dist]
foldrows=[]
for fo,(tr,va) in enumerate(outer):
 sc=StandardScaler().fit(Q[tr]);aa=sc.transform(Q[tr]);bb=sc.transform(Q[va]);qp=PCA(48,whiten=True,random_state=43).fit(aa);a=qp.transform(aa);b=qp.transform(bb)
 inner_pred=np.zeros((len(tr),L));inner_var=np.zeros_like(inner_pred);inner_dist=np.zeros(len(tr));inn=KFold(4,shuffle=True,random_state=440+fo)
 for it,iv in inn.split(tr):
  cp,m=fitcorr(tr[it],a[it]); trees=np.stack([cp.inverse_transform(t.predict(a[iv])) for t in m.estimators_]);inner_pred[iv]=trees.mean(0);inner_var[iv]=trees.var(0)
  inner_dist[iv]=np.sqrt(((a[iv,None]-a[it][None])**2).mean(2).min(1))
 rf=np.stack([riskfeat(a[j],inner_pred[j],inner_var[j],inner_dist[j],lens[tr[j]]) for j in range(len(tr))])
 den=np.sum(inner_pred**2,1)+1e-6; alpha=np.clip(np.sum(C[tr]*inner_pred,1)/den,0,1.5)
 # Outer correction and ensemble epistemic variance.
 cp,m=fitcorr(tr,a);trees=np.stack([cp.inverse_transform(t.predict(b)) for t in m.estimators_]);po=trees.mean(0);vv=trees.var(0);dd=np.sqrt(((b[:,None]-a[None])**2).mean(2).min(1));rv=np.stack([riskfeat(b[j],po[j],vv[j],dd[j],lens[va[j]]) for j in range(len(va))])
 gates={'ridge':Ridge(alpha=100).fit(rf,alpha).predict(rv),'et25':ExtraTreesRegressor(n_estimators=500,min_samples_leaf=25,max_features=.7,n_jobs=-1,random_state=81).fit(rf,alpha).predict(rv),'et50':ExtraTreesRegressor(n_estimators=500,min_samples_leaf=50,max_features=.7,n_jobs=-1,random_state=82).fit(rf,alpha).predict(rv)}
 P['full'][va]=po
 for n,g in gates.items(): A[n][va]=np.clip(g,0,1.25);P[n][va]=po*A[n][va,None]
 foldrows.append({'fold':fo,'alpha_inner_mean':float(alpha.mean()),'alpha_outer':{n:float(A[n][va].mean()) for n in names}});print(foldrows[-1],flush=True)
def score(p,ix):return float(np.sqrt(np.average(np.mean((C[ix]-p[ix])**2,1),weights=lens[ix])))
grid=[]
for n in names:
 fs=[score(np.zeros_like(C),va)-score(P[n],va) for _,va in outer];grid.append({'model':n,'score':score(P[n],np.arange(len(C))),'fold_gains':fs,'wins':sum(x>0 for x in fs),'alpha_mean':float(A[n].mean())})
best=min(grid,key=lambda x:x['score']);np.savez_compressed(OUT/'oof.npz',wells=W,pred=P[best['model']],alpha=A[best['model']],true=C,length=lens,model=best['model'])
out={'baseline':score(np.zeros_like(C),np.arange(len(C))),'grid':grid,'best':best,'nested_folds':foldrows};(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['oof_sha256']=hashlib.sha256((OUT/'oof.npz').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
