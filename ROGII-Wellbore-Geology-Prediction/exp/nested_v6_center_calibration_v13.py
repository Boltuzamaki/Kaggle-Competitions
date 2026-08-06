"""Nested datum/slope calibration from causal multi-cut self-error features."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_center_v13';OUT.mkdir(parents=True,exist_ok=True)
v=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=v['wells'].astype(str);C=v['true'];lens=v['length'];Q=v['query'];b=np.load(ROOT/'exp/results/prefix_backtest_moe/backtest_features.npz',allow_pickle=True);bm={w:i for i,w in enumerate(b['wells'].astype(str))};BT=b['features'][[bm[w] for w in W]];X=np.c_[Q,BT];outer=list(GroupKFold(5).split(X,groups=W));L=C.shape[1];u=np.linspace(-.5,.5,L)
def fitpred(tr,va):
 sc=StandardScaler().fit(X[tr]);aa=sc.transform(X[tr]);bb=sc.transform(X[va]);xp=PCA(48,whiten=True,random_state=43).fit(aa);cp=PCA(24,random_state=44).fit(C[tr]);m=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=12,max_features=.6,n_jobs=-1,random_state=45).fit(xp.transform(aa),cp.transform(C[tr]));return cp.inverse_transform(m.predict(xp.transform(bb)))
base=np.zeros_like(C);cal={a:np.zeros_like(C) for a in [.25,.5,1.]};oracle0=np.zeros_like(C);oracle1=np.zeros_like(C);meta=[]
for fo,(tr,va) in enumerate(outer):
 ip=np.zeros((len(tr),L));inner=list(GroupKFold(4).split(X[tr],groups=W[tr]))
 for it,iv in inner:ip[iv]=fitpred(tr[it],tr[iv])
 # Honest inner v6 error -> optimal datum and line targets.
 err=C[tr]-ip;coef=np.asarray([np.polyfit(u,e,1) for e in err]);off=coef[:,1];slope=coef[:,0]
 # Causal self-backtest summaries dominate; predicted correction morphology
 # adds only legal current-query context.
 Z=np.c_[BT[tr],ip.mean(1),ip.std(1),ip[:,-1]-ip[:,0],lens[tr]];Zv=np.c_[BT[va],np.zeros((len(va),3)),lens[va]]
 po=fitpred(tr,va);Zv[:,-4]=po.mean(1);Zv[:,-3]=po.std(1);Zv[:,-2]=po[:,-1]-po[:,0];base[va]=po
 sc=StandardScaler().fit(Z);zz=sc.transform(Z);zv=sc.transform(Zv);mo=Ridge(alpha=100).fit(zz,off,sample_weight=np.sqrt(lens[tr]));ms=Ridge(alpha=100).fit(zz,slope,sample_weight=np.sqrt(lens[tr]));oo=mo.predict(zv);ss=ms.predict(zv)
 for a in cal:cal[a][va]=po+a*(oo[:,None]+ss[:,None]*u)
 # Diagnostic oracles around the frozen outer center.
 ev=C[va]-po;oracle0[va]=po+ev.mean(1)[:,None];cc=np.asarray([np.polyfit(u,e,1) for e in ev]);oracle1[va]=po+cc[:,1,None]+cc[:,0,None]*u
 meta.append({'fold':fo,'inner_offset_std':float(off.std()),'pred_offset_std':float(oo.std()),'pred_slope_std':float(ss.std())});print(meta[-1],flush=True)
def score(p,ix=None):
 if ix is None:ix=np.arange(len(C))
 return float(np.sqrt(np.average(np.mean((C[ix]-p[ix])**2,1),weights=lens[ix])))
grid=[]
for n,p in [('base',base)]+[(f'cal{a}',cal[a]) for a in cal]+[('datum_oracle',oracle0),('line_oracle',oracle1)]:grid.append({'model':n,'score':score(p),'folds':[score(p,va) for _,va in outer]})
best=min(grid[:4],key=lambda x:x['score']);bp=base if best['model']=='base' else cal[float(best['model'][3:])];np.savez_compressed(OUT/'oof.npz',wells=W,pred=bp,base=base,datum_oracle=oracle0,line_oracle=oracle1,true=C,length=lens);out={'grid':grid,'best_deployable':best,'meta':meta,'protocol':'outer GKF5; inner GKF4 honest v6 errors; fixed Ridge100 and shrink grid'};(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['sha256']=hashlib.sha256((OUT/'oof.npz').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
