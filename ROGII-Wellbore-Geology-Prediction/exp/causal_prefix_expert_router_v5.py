"""Causal-prefix-error router over accepted legal curve family + Student PF."""
from pathlib import Path
import contextlib,io,runpy,json,hashlib
import numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/causal_prefix_expert_router_v5';OUT.mkdir(parents=True,exist_ok=True)
c=np.load(ROOT/'exp/results/complete_well_moe/well_table.npz',allow_pickle=True);old=np.load(ROOT/'exp/results/continuous_well_moe/cpu_search_cache.npz',allow_pickle=True)
v=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');wn=c['wn'].astype(str);vw=v['wells'].astype(str);assert set(wn)==set(vw);vm={w:i for i,w in enumerate(vw)};X=v['query'][[vm[w] for w in wn]];nrow=c['nrow'].astype(float)
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(ROOT/'exp/meta_all_honest_oof.py'))
common,y,wells,L=s['common'],s['y'],s['wells'],s['legs'];fm=np.load(ROOT/'exp/results/heel_calibrated_gr_datum/full_meta/oof.npz');student=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True)
vf=pd.read_pickle(ROOT/'r_v4b/train_feats.pkl').iloc[common].reset_index(drop=True);hq=np.load(ROOT/'exp/results/heel_calibrated_gr_datum/oof.npz')
def project(p,deg):
 out=p.copy()
 for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False):
  ix=ii.to_numpy();x=vf.d_md.to_numpy()[ix];x=2*(x-x.min())/max(np.ptp(x),1e-6)-1;out[ix]=np.polyval(np.polyfit(x,p[ix]+vf.d_z.to_numpy()[ix],deg),x)-vf.d_z.to_numpy()[ix]
 return out
heel=L['v4_lgb7']+hq['correction'][common]
P=np.column_stack([fm['add'],fm['base'],heel,L['v4_lgb7'],L['har_physics'],L['har_lgb'],L['har_xgb'],L['pil_blend_oof_postprocessed'],project(L['v4_lgb7'],2),project(L['v4_lgb7'],3),student['replacement']])
names=['accepted','base5','heel','v4','har_phys','har_lgb','har_xgb','pil','poly2','poly3','student_replacement'];wix=[ii.to_numpy() for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False)];K=P.shape[1]
G=np.empty((len(wn),K,K))
for i,ix in enumerate(wix):e=P[ix]-y[ix,None];G[i]=e.T@e/len(ix)
e0=np.zeros(K);e0[0]=.1;e0[-1]=.9
target=np.zeros((len(wn),K))
for i in range(len(wn)):
 fun=lambda w:w@G[i]@w+.5*np.sum((w-e0)**2);target[i]=minimize(fun,e0,method='SLSQP',bounds=[(0,1)]*K,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-9}).x
def norm(a):a=np.clip(a,0,None);return a/(a.sum(1,keepdims=True)+1e-9)
def score(W):return float(np.sqrt(np.sum(nrow*np.einsum('ni,nij,nj->n',W,G,W))/nrow.sum()))
folds=list(GroupKFold(5).split(X,groups=wn));pred=np.zeros_like(target);fid=np.zeros(len(wn),int)
for k,(tr,va) in enumerate(folds):
 m=ExtraTreesRegressor(n_estimators=1500,min_samples_leaf=10,max_features=.6,n_jobs=-1,random_state=510+k).fit(X[tr],target[tr],sample_weight=np.sqrt(nrow[tr]));pred[va]=norm(m.predict(X[va]));fid[va]=k;print('fold',k,flush=True)
base=np.tile(e0,(len(wn),1));grid=[]
for a in [.1,.2,.35,.5,.75,1]:
 q=(1-a)*base+a*pred;fg=[]
 for k in range(5):
  ix=fid==k;f=lambda W:float(np.sqrt(np.sum(nrow[ix]*np.einsum('ni,nij,nj->n',W[ix],G[ix],W[ix]))/nrow[ix].sum()));fg.append(f(base)-f(q))
 grid.append({'blend':a,'rmse':score(q),'fold_gains':fg,'wins':sum(x>0 for x in fg),'entropy':float(np.mean(-np.sum(q*np.log(q+1e-12),1)))})
best=min(grid,key=lambda z:z['rmse']);np.savez_compressed(OUT/'oof_weights.npz',wells=wn,weights=pred,target=target,fold=fid,nrow=nrow,names=names,G=G,X=X)
out={'baseline':score(base),'oracle':score(target),'grid':grid,'best':best,'experts':names};(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['sha256']=hashlib.sha256((OUT/'oof_weights.npz').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
