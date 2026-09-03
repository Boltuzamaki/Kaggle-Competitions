from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
TARGET,ID,SEED='addicted_label','id',20260803
def locate():
 for p in Path('/kaggle/input').rglob('train.csv'):
  try:
   q=pd.read_csv(p,nrows=2)
   if TARGET in q and p.with_name('test.csv').exists(): return p
  except: pass
 raise FileNotFoundError
def feat(d):
 x=d.drop(columns=[ID,TARGET],errors='ignore').copy(); raw=list(x.columns); miss=x.isna()
 x['missing_count']=miss.sum(1); x['missing_pattern']=miss.astype('int8').astype(str).agg(''.join,axis=1)
 for c in raw: x[c+'__missing']=miss[c].astype('int8')
 x['leisure']=x.social_media_hours+x.gaming_hours; x['accounted']=x.leisure+x.work_study_hours
 x['screen_residual']=x.daily_screen_time_hours-x.accounted; x['weekend_gap']=x.weekend_screen_time-x.daily_screen_time_hours
 x['weekend_ratio']=x.weekend_screen_time/(x.daily_screen_time_hours+.25); x['leisure_share']=x.leisure/(x.daily_screen_time_hours+.25)
 x['notif_per_open']=x.notifications_per_day/(x.app_opens_per_day+2); x['sleep_balance']=x.sleep_hours-x.daily_screen_time_hours
 return x.replace([np.inf,-np.inf],np.nan)
p=locate(); tr=pd.read_csv(p); te=pd.read_csv(p.with_name('test.csv')); y=tr[TARGET].astype('int8'); x,z=feat(tr),feat(te)
cc=list(x.select_dtypes(exclude='number').columns); enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1)
enc.fit(pd.concat([x[cc],z[cc]]).fillna('Missing').astype(str)); x[cc]=enc.transform(x[cc].fillna('Missing').astype(str)); z[cc]=enc.transform(z[cc].fillna('Missing').astype(str))
x=x.astype('float32'); z=z.astype('float32'); sk=StratifiedKFold(5,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); pred=np.zeros(len(te)); folds=np.full(len(tr),-1); scores=[]; iterations=[]
for k,(a,b) in enumerate(sk.split(x,y)):
 m=HistGradientBoostingClassifier(loss='log_loss',learning_rate=.065,max_iter=800,max_leaf_nodes=31,min_samples_leaf=80,l2_regularization=1.5,max_bins=255,early_stopping=True,validation_fraction=.1,n_iter_no_change=60,random_state=SEED+k)
 m.fit(x.iloc[a],y.iloc[a]); oof[b]=m.predict_proba(x.iloc[b])[:,1]; pred+=m.predict_proba(z)[:,1]/5; folds[b]=k; scores.append(roc_auc_score(y.iloc[b],oof[b])); iterations.append(m.n_iter_)
pd.DataFrame({ID:tr[ID],TARGET:y,'fold':folds,'prediction':oof}).to_csv('/kaggle/working/oof.csv',index=False); pd.DataFrame({ID:te[ID],TARGET:pred}).to_csv('/kaggle/working/test_predictions.csv',index=False); pd.DataFrame({ID:tr[ID],'fold':folds}).to_csv('/kaggle/working/folds.csv',index=False)
r={'model':'sklearn HistGradientBoosting diverse baseline','seed':SEED,'fold_auc':scores,'oof_auc':roc_auc_score(y,oof),'iterations':iterations,'provenance':'competition train only'}; Path('/kaggle/working/metrics.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
