"""Analytical leave-one-out exact-value TE LightGBM; official S6E8 data only."""
from pathlib import Path
import gc,json,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
TARGET,ID,SEED="addicted_label","id",20260803
SMOOTHS=(10.,30.)
def locate():
 for root in (Path('/kaggle/input'),Path('.')):
  for p in root.rglob('train.csv'):
   try: cols=pd.read_csv(p,nrows=2).columns
   except: continue
   if TARGET in cols and (p.parent/'test.csv').exists(): return p,p.parent/'test.csv'
 raise FileNotFoundError('official competition data not found')
def key(s): return s.astype('string').fillna('__NA__')
def base(df):
 x=df.drop(columns=[ID,TARGET],errors='ignore').copy(); raw=list(x); miss=x.isna()
 x['missing_count']=miss.sum(1).astype('int8'); x['missing_pattern']=miss.astype('uint8').astype(str).agg(''.join,axis=1)
 for c in raw: x[c+'__missing']=miss[c].astype('int8')
 x['leisure_hours']=x.social_media_hours+x.gaming_hours
 x['accounted_hours']=x.leisure_hours+x.work_study_hours
 x['unaccounted_screen']=x.daily_screen_time_hours-x.accounted_hours
 x['weekend_gap']=x.weekend_screen_time-x.daily_screen_time_hours
 x['notifications_per_open']=x.notifications_per_day/(x.app_opens_per_day+3.)
 x['opens_per_screen_hour']=x.app_opens_per_day/(x.daily_screen_time_hours+.5)
 return x.replace([np.inf,-np.inf],np.nan),raw
def loo_encode(fit,yfit,valid,test,cols,smooth,verify=False):
 a,b,c=fit.reset_index(drop=True).copy(),valid.reset_index(drop=True).copy(),test.reset_index(drop=True).copy(); yy=np.asarray(yfit,dtype='float64'); total=yy.sum(); n=len(yy)
 for col in cols:
  kk=key(a[col]); z=pd.DataFrame({'k':kk.to_numpy(),'y':yy}); st=z.groupby('k',observed=True).y.agg(['sum','count'])
  sums=kk.map(st['sum']).to_numpy('float64'); counts=kk.map(st['count']).to_numpy('float64')
  # Both the group evidence and global prior exclude the row's own label.
  prior_loo=(total-yy)/(n-1); te=(sums-yy+smooth*prior_loo)/(counts-1+smooth)
  prior=float(yy.mean()); full=(st['sum']+smooth*prior)/(st['count']+smooth)
  a[col+'__loo_te']=te.astype('float32'); b[col+'__loo_te']=key(b[col]).map(full).fillna(prior).to_numpy('float32'); c[col+'__loo_te']=key(c[col]).map(full).fillna(prior).to_numpy('float32')
  a[col+'__loo_logcount']=np.log1p(counts-1).astype('float32'); b[col+'__loo_logcount']=np.log1p(key(b[col]).map(st['count']).fillna(0)).to_numpy('float32'); c[col+'__loo_logcount']=np.log1p(key(c[col]).map(st['count']).fillna(0)).to_numpy('float32')
  if verify:
   singles=np.flatnonzero(counts==1)
   if len(singles): assert np.allclose(te[singles[:100]],prior_loo[singles[:100]],rtol=0,atol=1e-12)
   # Brute-force check proves the analytical value equals fitting after row removal.
   for i in np.linspace(0,n-1,min(25,n),dtype=int):
    mask=np.ones(n,dtype=bool); mask[i]=False; other_global=yy[mask].mean(); same=mask & (kk.to_numpy()==kk.iloc[i]); brute=(yy[same].sum()+smooth*other_global)/(same.sum()+smooth)
    assert abs(te[i]-brute)<1e-12,(col,i,te[i],brute)
 for col in a.select_dtypes(exclude='number'):
  levels=pd.Index(pd.concat([key(a[col]),key(b[col]),key(c[col])],ignore_index=True).unique()); dtype=pd.CategoricalDtype(levels)
  for frame in (a,b,c): frame[col]=key(frame[col]).astype(dtype)
 return a,b,c,list(a.select_dtypes(exclude='number'))
trp,tep=locate(); tr,te=pd.read_csv(trp),pd.read_csv(tep); y=tr[TARGET].to_numpy('int8'); xb,raw=base(tr); xt,_=base(te)
cols=raw+['missing_pattern']; outer=StratifiedKFold(5,shuffle=True,random_state=SEED); folds=np.zeros(len(tr),dtype='int8'); oofs={str(int(s)):np.zeros(len(tr)) for s in SMOOTHS}; tests={str(int(s)):[] for s in SMOOTHS}; rows=[]
for fold,(fi,vi) in enumerate(outer.split(xb,y),1):
 for smooth in SMOOTHS:
  name=str(int(smooth)); a,b,c,cats=loo_encode(xb.iloc[fi],y[fi],xb.iloc[vi],xt,cols,smooth,verify=(fold==1))
  m=lgb.LGBMClassifier(objective='binary',n_estimators=3000,learning_rate=.027,num_leaves=31,max_depth=7,min_child_samples=220,subsample=.85,subsample_freq=1,colsample_bytree=.85,reg_alpha=.2,reg_lambda=7.,random_state=SEED+fold,n_jobs=-1,verbosity=-1)
  m.fit(a,y[fi],eval_set=[(b,y[vi])],eval_metric='auc',categorical_feature=cats,callbacks=[lgb.early_stopping(180,verbose=False)])
  pv=m.predict_proba(b)[:,1]; oofs[name][vi]=pv; tests[name].append(m.predict_proba(c)[:,1]); rows.append({'fold':fold,'smooth':smooth,'auc':float(roc_auc_score(y[vi],pv)),'best_iteration':int(m.best_iteration_)}); print(rows[-1],flush=True); del a,b,c,m; gc.collect()
 folds[vi]=fold
scores={k:float(roc_auc_score(y,v)) for k,v in oofs.items()}; stds={k:float(np.std([r['auc'] for r in rows if str(int(r['smooth']))==k],ddof=1)) for k in oofs}; selected=max(scores,key=scores.get)
pd.DataFrame({ID:tr[ID],'fold':folds,'y':y,**{'pred_s'+k:v for k,v in oofs.items()}}).to_csv('/kaggle/working/oof_loo_te_lgb.csv',index=False)
pd.DataFrame({ID:te[ID],**{'pred_s'+k:np.mean(v,axis=0) for k,v in tests.items()}}).to_csv('/kaggle/working/test_loo_te_lgb.csv',index=False)
pd.DataFrame({ID:tr[ID],'fold':folds}).to_csv('/kaggle/working/folds.csv',index=False)
metrics={'provenance':'official competition data only; no public predictions','seed':SEED,'folds':5,'smoothing':SMOOTHS,'loo_definition':'own y removed from group sum and global prior; count decremented','verification':'singleton equals leave-one-out global prior; 25 brute-force removals per encoded column passed on fold 1','fold_metrics':rows,'oof_auc':scores,'fold_auc_std':stds,'selected_by_oof':selected}
Path('/kaggle/working/metrics_loo_te_lgb.json').write_text(json.dumps(metrics,indent=2)+'\n'); print(json.dumps(metrics,indent=2),flush=True)
