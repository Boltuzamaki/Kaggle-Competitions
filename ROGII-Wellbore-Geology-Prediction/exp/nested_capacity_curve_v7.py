"""Strict nested capacity selection/ensembling for v6 causal features."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_nested_v7';OUT.mkdir(parents=True,exist_ok=True)
v=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=v['wells'].astype(str);C=v['true'];lens=v['length'].astype(float);Q=v['query']
b=np.load(ROOT/'exp/results/prefix_backtest_moe/backtest_features.npz',allow_pickle=True);bw=b['wells'].astype(str);bm={w:i for i,w in enumerate(bw)};X=np.c_[Q,b['features'][[bm[w] for w in W]]]
cfg=[('et_base','et',24,12,.6),('et_leaf6','et',24,6,.6),('et_leaf20','et',24,20,.6),('et_lowdim','et',16,12,.4),('et_highdim','et',32,12,.8),('rf','rf',24,8,.6),('ridge','ridge',24,0,100.)]
def fitpred(tr,va,c,seed):
 name,kind,npc,leaf,mf=c;sc=StandardScaler().fit(X[tr]);aa=sc.transform(X[tr]);bb=sc.transform(X[va]);xp=PCA(48,whiten=True,random_state=43).fit(aa);aa=xp.transform(aa);bb=xp.transform(bb);cp=PCA(npc,random_state=44).fit(C[tr]);lat=cp.transform(C[tr])
 if kind=='et':m=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=leaf,max_features=mf,n_jobs=-1,random_state=seed)
 elif kind=='rf':m=RandomForestRegressor(n_estimators=400,min_samples_leaf=leaf,max_features=mf,n_jobs=-1,random_state=seed)
 else:m=Ridge(alpha=mf)
 m.fit(aa,lat);return cp.inverse_transform(m.predict(bb))
def mse(p,ix):return float(np.average(np.mean((C[ix]-p)**2,1),weights=lens[ix]))
outer=list(GroupKFold(5).split(X,groups=W));pred=np.zeros_like(C);choices=[]
for fo,(tr,va) in enumerate(outer):
 inner=list(GroupKFold(4).split(X[tr],groups=W[tr]));oof={c[0]:np.zeros((len(tr),C.shape[1])) for c in cfg}
 for ii,(it,iv) in enumerate(inner):
  for c in cfg:oof[c[0]][iv]=fitpred(tr[it],tr[iv],c,7000+fo*100+ii)
 scores={n:mse(p,tr) for n,p in oof.items()};rank=sorted(scores,key=scores.get);top=rank[:3];ens=np.mean([oof[n] for n in top],0);ens_score=mse(ens,tr)
 if ens_score< scores[rank[0]]: selected=top
 else:selected=[rank[0]]
 outerp=[fitpred(tr,va,next(c for c in cfg if c[0]==n),8000+fo) for n in selected];pred[va]=np.mean(outerp,0)
 choices.append({'fold':fo,'inner_rmse':{n:float(np.sqrt(x)) for n,x in scores.items()},'top3':top,'ensemble_rmse':float(np.sqrt(ens_score)),'selected':selected});print(choices[-1],flush=True)
def score(p,ix):return float(np.sqrt(np.average(np.mean((C[ix]-p[ix])**2,1),weights=lens[ix])))
fg=[score(np.zeros_like(C),va)-score(pred,va) for _,va in outer];out={'grid_baseline':score(np.zeros_like(C),np.arange(len(C))),'score':score(pred,np.arange(len(C))),'fold_gains':fg,'wins':sum(x>0 for x in fg),'choices':choices,'protocol':'outer GKF5; inner GKF4 capacity selection; outer labels untouched'}
np.savez_compressed(OUT/'oof.npz',wells=W,pred=pred,true=C,length=lens);(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['sha256']=hashlib.sha256((OUT/'oof.npz').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
