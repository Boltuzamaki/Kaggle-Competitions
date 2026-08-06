"""Whole-well models predicting continuous V4 correction from cross-fit GR cost curves."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from tabicl import TabICLRegressor
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/crossfit_horizontal_gr_reference';MODEL=ROOT/'exp/public_artifacts/tabicl/tabicl-regressor-v2-20260212.ckpt'
r=np.load(OUT/'oof.npz',allow_pickle=True);c=np.load(ROOT/'exp/results/complete_well_moe/well_table.npz',allow_pickle=True)
mp={w:i for i,w in enumerate(r['wells'].astype(str))};ri=np.asarray([mp[w] for w in c['wn'].astype(str)])
C=r['costs'][ri].astype(np.float32);C-=C.min(1,keepdims=True);X=np.c_[c['X'].astype(np.float32),C,np.sort(C,axis=1)[:,:8],C.std(1),C.mean(1)]
target=np.clip(r['oracle_well_shift'][ri],-40,40).astype(np.float32);fold=r['fold'][ri];pred={k:np.zeros(len(X),np.float32) for k in ('extra','cat','hist','tabicl')}
for k in range(5):
 tr=np.flatnonzero(fold!=k);va=np.flatnonzero(fold==k)
 models={
  'extra':ExtraTreesRegressor(n_estimators=500,min_samples_leaf=8,max_features=.65,n_jobs=12,random_state=8100+k),
  'cat':CatBoostRegressor(iterations=900,depth=6,learning_rate=.035,l2_leaf_reg=25,loss_function='RMSE',verbose=False,random_seed=8200+k),
  'hist':HistGradientBoostingRegressor(max_iter=350,max_leaf_nodes=15,l2_regularization=30,learning_rate=.04,random_state=8300+k),
  'tabicl':TabICLRegressor(n_estimators=4,batch_size=4,kv_cache=False,model_path=MODEL,allow_auto_download=False,device='cuda',use_amp=True,random_state=8400+k,verbose=False)}
 for name,m in models.items():m.fit(X[tr],target[tr]);pred[name][va]=m.predict(X[va])
 print('fold',k,'done',flush=True)
# Exact row scoring aligned to common 765-well meta state.
import contextlib,io,runpy
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(ROOT/'exp/meta_all_honest_oof.py'))
y=s['y'];w=s['wells'];base=np.load(ROOT/'exp/results/heel_calibrated_gr_datum/full_meta/oof.npz')['add'];uw=pd.Series(w).drop_duplicates().astype(str).to_numpy()
if not np.array_equal(uw,c['wn'].astype(str)):raise RuntimeError('well identity')
codes=pd.Categorical(np.asarray(w).astype(str),categories=uw).codes
if np.any(codes<0):raise RuntimeError('unknown well code')
rows=[]
for name,q in pred.items():
 corr=q[codes]
 for a in (.05,.1,.2,.3,.5,.7,1):
  p=base+a*corr;fg=[]
  for k in range(5):
   ix=fold[codes]==k;fg.append(float(np.sqrt(np.mean((base[ix]-y[ix])**2))-np.sqrt(np.mean((p[ix]-y[ix])**2))))
  rows.append({'model':name,'blend':a,'rmse':float(np.sqrt(np.mean((p-y)**2))),'fold_gains':fg,'fold_wins':sum(x>0 for x in fg)})
summary={'protocol':'outer reference folds reused for whole-well correction models; target is clipped per-well mean V4 error; exact accepted-stack row scoring','accepted':float(np.sqrt(np.mean((base-y)**2))),
         'best':min(rows,key=lambda x:x['rmse']),'best_5of5':min((x for x in rows if x['fold_wins']==5),key=lambda x:x['rmse'],default=None)}
(OUT/'shift_models_summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/'shift_models_oof.npz',**pred,fold=fold,wells=uw);print(json.dumps(summary,indent=2))
