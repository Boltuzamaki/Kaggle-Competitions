from pathlib import Path
import json, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
TARGET,ID,SEED='addicted_label','id',20260803
def locate():
 for p in Path('/kaggle/input').rglob('train.csv'):
  try:
   h=pd.read_csv(p,nrows=2)
   if TARGET in h and p.with_name('test.csv').exists(): return p
  except: pass
 raise FileNotFoundError
def feat(d):
 x=d.drop(columns=[ID,TARGET],errors='ignore').copy(); cols=list(x.columns); miss=x[cols].isna()
 x['missing_count']=miss.sum(1); x['missing_pattern']=miss.astype('int8').astype(str).agg(''.join,axis=1)
 for c in cols: x[c+'__missing']=miss[c].astype('int8')
 x['leisure']=x.social_media_hours+x.gaming_hours; x['accounted']=x.leisure+x.work_study_hours
 x['screen_residual']=x.daily_screen_time_hours-x.accounted; x['weekend_gap']=x.weekend_screen_time-x.daily_screen_time_hours
 x['leisure_share']=x.leisure/(x.daily_screen_time_hours+.25); x['work_share']=x.work_study_hours/(x.daily_screen_time_hours+.25)
 x['notif_per_open']=x.notifications_per_day/(x.app_opens_per_day+2); x['sleep_balance']=x.sleep_hours-x.daily_screen_time_hours
 for c in x.select_dtypes(exclude='number'): x[c]=x[c].fillna('Missing').astype(str)
 return x.replace([np.inf,-np.inf],np.nan)
p=locate(); tr=pd.read_csv(p); te=pd.read_csv(p.with_name('test.csv')); y=tr[TARGET].astype('int8'); x,z=feat(tr),feat(te); cat=list(x.select_dtypes(exclude='number').columns)
sk=StratifiedKFold(5,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); pred=np.zeros(len(te)); folds=np.full(len(tr),-1); scores=[]; it=[]
for k,(a,b) in enumerate(sk.split(x,y)):
 m=CatBoostClassifier(iterations=2600,depth=8,learning_rate=.055,loss_function='Logloss',eval_metric='AUC',l2_leaf_reg=5,random_seed=SEED+k,thread_count=-1,verbose=250,od_type='Iter',od_wait=180,allow_writing_files=False)
 m.fit(x.iloc[a],y.iloc[a],cat_features=cat,eval_set=(x.iloc[b],y.iloc[b]),use_best_model=True)
 oof[b]=m.predict_proba(x.iloc[b])[:,1]; pred+=m.predict_proba(z)[:,1]/5; folds[b]=k; scores.append(roc_auc_score(y.iloc[b],oof[b])); it.append(m.get_best_iteration())
pd.DataFrame({ID:tr[ID],TARGET:y,'fold':folds,'prediction':oof}).to_csv('/kaggle/working/oof.csv',index=False); pd.DataFrame({ID:te[ID],TARGET:pred}).to_csv('/kaggle/working/test_predictions.csv',index=False); pd.DataFrame({ID:tr[ID],'fold':folds}).to_csv('/kaggle/working/folds.csv',index=False)
r={'model':'CatBoost composition/missing-pattern CPU','seed':SEED,'fold_auc':scores,'oof_auc':roc_auc_score(y,oof),'iterations':it,'provenance':'competition train only'}; Path('/kaggle/working/metrics.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
