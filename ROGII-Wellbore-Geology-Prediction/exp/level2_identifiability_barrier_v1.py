"""Repeated OOF identifiability audit for remaining L2 datum/slope residual."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

R=Path(__file__).resolve().parents[1];SRC=R/'exp/results/legal_level2_all_oof_v1/oof.npz';OUT=R/'exp/results/level2_identifiability_barrier_v1';OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];g=S['groups'].astype(str);student=.1*S['accepted']+.9*S['replacement'];l2=np.load(SRC,allow_pickle=True)['anchored_difference'];center=student+l2;err=y-center
V=np.load(R/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=V['wells'].astype(str);Q=V['query'];B=np.load(R/'exp/results/prefix_backtest_moe/backtest_features.npz',allow_pickle=True);bm={w:i for i,w in enumerate(B['wells'].astype(str))};X=np.nan_to_num(np.c_[Q,B['features'][[bm[w] for w in W]]]);
cuts=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1,len(g)];ixs={g[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])};coef=[];lens=[];oracle=center.copy()
for w in W:
 ix=ixs[w];t=np.linspace(0,1,len(ix));A=np.c_[np.ones(len(ix)),t];co=np.linalg.lstsq(A,err[ix],rcond=None)[0];coef.append(co);lens.append(len(ix));oracle[ix]+=A@co
T=np.asarray(coef);lens=np.asarray(lens);models=['et','ridge100','ridge1000','null'];reps={m:[] for m in models};perm=[]
for seed in [91,193,307]:
 folds=list(KFold(5,shuffle=True,random_state=seed).split(W));P={m:np.zeros_like(T) for m in models};PP=np.zeros_like(T)
 for fi,(tr,va) in enumerate(folds):
  sc=StandardScaler().fit(X[tr]);a=sc.transform(X[tr]);b=sc.transform(X[va]);pc=PCA(64,whiten=True,random_state=seed+fi).fit(a);a=pc.transform(a);b=pc.transform(b)
  P['et'][va]=ExtraTreesRegressor(n_estimators=450,min_samples_leaf=12,max_features=.65,n_jobs=4,random_state=seed+fi).fit(a,T[tr]).predict(b)
  P['ridge100'][va]=Ridge(100).fit(a,T[tr]).predict(b);P['ridge1000'][va]=Ridge(1000).fit(a,T[tr]).predict(b);P['null'][va]=np.average(T[tr],axis=0,weights=lens[tr])
  rng=np.random.default_rng(seed+fi);PP[va]=ExtraTreesRegressor(n_estimators=180,min_samples_leaf=12,max_features=.65,n_jobs=4,random_state=seed+900+fi).fit(a,T[rng.permutation(tr)]).predict(b)
 for m in models:reps[m].append(P[m]);perm.append(PP);print('repeat',seed,flush=True)
def rowcorr(P,mask=None):
 o=np.zeros(len(y));
 for i,w in enumerate(W):
  if mask is None or mask[i]:
   ix=ixs[w];o[ix]=P[i,0]+P[i,1]*np.linspace(0,1,len(ix))
 return o
def rm(p):return float(np.sqrt(np.mean((y-p)**2)))
def stats(P):
 out={}
 for j,n in enumerate(['datum','slope']):
  mu=np.average(T[:,j],weights=lens);out[n+'_r2']=float(1-np.average((T[:,j]-P[:,j])**2,weights=lens)/np.average((T[:,j]-mu)**2,weights=lens));out[n+'_corr']=float(np.corrcoef(T[:,j],P[:,j])[0,1])
 out['row_rmse']=rm(center+rowcorr(P));return out
summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'baseline':rm(center),'oracle_affine':rm(oracle),'models':{},'permutation':[]}
for m in models:
 summary['models'][m]={'repeats':[stats(p) for p in reps[m]],'mean_prediction':stats(np.mean(reps[m],0))}
summary['permutation']=[stats(p) for p in perm]
# Legal subgroup gates from repeated ET magnitude and disagreement only.
pm=np.mean(reps['et'],0);unc=np.mean(np.std(reps['et'],axis=0),axis=1);amp=np.sqrt(pm[:,0]**2+pm[:,1]**2);subs=[]
for mass in [.1,.25,.5]:
 n=max(1,int(len(W)*mass));
 for gate,order in [('high_amplitude',np.argsort(-amp)),('low_disagreement',np.argsort(unc)),('amplitude_per_uncertainty',np.argsort(-(amp/(unc+1e-3))))]:
  mask=np.zeros(len(W),bool);mask[order[:n]]=1;rows=sum(lens[mask]);pr=center+rowcorr(pm,mask);subs.append({'gate':gate,'well_mass':mass,'row_mass':float(rows/lens.sum()),'rmse':rm(pr),'gain':rm(center)-rm(pr)})
summary['subgroups']=subs;(OUT/'summary.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(subs).to_csv(OUT/'subgroups.csv',index=False);np.savez_compressed(OUT/'oof.npz',wells=W,target=T,et=np.asarray(reps['et']),ridge100=np.asarray(reps['ridge100']),permutation=np.asarray(perm));print(json.dumps(summary,indent=2))
