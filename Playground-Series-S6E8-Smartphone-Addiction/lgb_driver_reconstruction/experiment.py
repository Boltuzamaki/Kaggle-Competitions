"""Fold-local missing-driver reconstruction, inspired by public generator forensics."""
from pathlib import Path
import gc,json,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import roc_auc_score,mean_squared_error
from sklearn.model_selection import StratifiedKFold
TARGET,ID,SEED="addicted_label","id",20260803
DRIVERS=('daily_screen_time_hours','social_media_hours')
def locate():
 for root in (Path('/kaggle/input'),Path('.')):
  for p in root.rglob('train.csv'):
   try: cols=pd.read_csv(p,nrows=2).columns
   except: continue
   if TARGET in cols and (p.parent/'test.csv').exists(): return p,p.parent/'test.csv'
 raise FileNotFoundError
def raw(df): return df.drop(columns=[ID,TARGET],errors='ignore').copy()
def numeric_code(a,b,c):
 a,b,c=a.copy(),b.copy(),c.copy()
 for col in a.select_dtypes(exclude='number'):
  joint=pd.concat([a[col].astype('string'),b[col].astype('string'),c[col].astype('string')],ignore_index=True).fillna('__NA__'); codes,_=pd.factorize(joint,sort=True)
  a[col]=codes[:len(a)]; b[col]=codes[len(a):len(a)+len(b)]; c[col]=codes[len(a)+len(b):]
 return a.astype('float32'),b.astype('float32'),c.astype('float32')
def reconstruct(fit,valid,test,fold):
 a,b,c=numeric_code(fit,valid,test); report={}
 for driver in DRIVERS:
  predictors=[x for x in a if x!=driver]; observed=a[driver].notna()
  reg=lgb.LGBMRegressor(n_estimators=700,learning_rate=.04,num_leaves=31,max_depth=7,min_child_samples=180,reg_lambda=5.,verbosity=-1,n_jobs=-1,random_state=SEED+fold)
  reg.fit(a.loc[observed,predictors],a.loc[observed,driver])
  for frame in (a,b,c):
   miss=frame[driver].isna(); frame[driver+'__reconstructed_flag']=miss.astype('int8'); pred=reg.predict(frame.loc[miss,predictors]) if miss.any() else np.array([]); frame[driver+'__completed']=frame[driver]; frame.loc[miss,driver+'__completed']=pred
  # Diagnostic only on outer validation's observed driver values.
  vm=b[driver].notna(); report[driver]={'validation_observed_rmse':float(mean_squared_error(b.loc[vm,driver],reg.predict(b.loc[vm,predictors]))**.5),'fit_observed_rows':int(observed.sum())}
  del reg
 for frame in (a,b,c):
  frame['component_sum']=frame.social_media_hours+frame.gaming_hours+frame.work_study_hours
  frame['completed_component_sum']=frame.social_media_hours__completed+frame.gaming_hours+frame.work_study_hours
  frame['completed_other_screen']=frame.daily_screen_time_hours__completed-frame.completed_component_sum
  frame['completed_social_share']=frame.social_media_hours__completed/(frame.daily_screen_time_hours__completed+.5)
  miss=frame.isna(); frame['missing_count']=miss.sum(1).astype('int8')
  for col in fit.columns: frame[col+'__missing']=frame[col].isna().astype('int8')
 return a,b,c,report
trp,tep=locate(); tr,te=pd.read_csv(trp),pd.read_csv(tep); y=tr[TARGET].to_numpy('int8'); x,xt=raw(tr),raw(te)
outer=StratifiedKFold(5,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); folds=np.zeros(len(tr),dtype='int8'); tests=[]; rows=[]
for fold,(fi,vi) in enumerate(outer.split(x,y),1):
 a,b,c,recon=reconstruct(x.iloc[fi],x.iloc[vi],xt,fold)
 m=lgb.LGBMClassifier(objective='binary',n_estimators=3000,learning_rate=.027,num_leaves=31,max_depth=7,min_child_samples=220,subsample=.85,subsample_freq=1,colsample_bytree=.86,reg_alpha=.2,reg_lambda=7.,verbosity=-1,n_jobs=-1,random_state=SEED+fold)
 m.fit(a,y[fi],eval_set=[(b,y[vi])],eval_metric='auc',callbacks=[lgb.early_stopping(180,verbose=False)])
 pv=m.predict_proba(b)[:,1]; oof[vi]=pv; tests.append(m.predict_proba(c)[:,1]); folds[vi]=fold; rows.append({'fold':fold,'auc':float(roc_auc_score(y[vi],pv)),'best_iteration':int(m.best_iteration_),'reconstruction':recon}); print(rows[-1],flush=True); del a,b,c,m; gc.collect()
pred=np.mean(tests,axis=0); auc=float(roc_auc_score(y,oof))
pd.DataFrame({ID:tr[ID],'fold':folds,'y':y,'pred':oof}).to_csv('/kaggle/working/oof_driver_reconstruction_lgb.csv',index=False)
pd.DataFrame({ID:te[ID],'pred':pred}).to_csv('/kaggle/working/test_driver_reconstruction_lgb.csv',index=False)
pd.DataFrame({ID:tr[ID],'fold':folds}).to_csv('/kaggle/working/folds.csv',index=False)
metrics={'provenance':'official competition data only; no source prediction outputs','source_credit':['Georgy Mamarin: missing strong drivers explain part of auxiliary-feature gain','ryota517: time-allocation generator constraint','Dariush Afshar: other_screen residual'],'our_change':'outer-fold-local regressors reconstruct missing daily/social drivers; classifier sees completed drivers and completed constraint residual','seed':SEED,'folds':5,'fold_metrics':rows,'oof_auc':auc}
Path('/kaggle/working/metrics.json').write_text(json.dumps(metrics,indent=2)+'\n'); print(json.dumps(metrics,indent=2),flush=True)
