from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET, ID, SEED = "addicted_label", "id", 20260803

def find_data():
    for p in Path('/kaggle/input').rglob('train.csv'):
        try:
            h = pd.read_csv(p, nrows=2)
            if TARGET in h and ID in h and p.with_name('test.csv').exists(): return p
        except Exception: pass
    raise FileNotFoundError('competition train.csv not found')

def base(d):
    x=d.drop(columns=[ID,TARGET],errors='ignore').copy(); raw=list(x.columns)
    x['missing_count']=x[raw].isna().sum(1).astype('int8')
    x['missing_pattern']=x[raw].isna().astype(str).agg(''.join,axis=1).astype('category')
    for c in raw: x[c+'__missing']=x[c].isna().astype('int8')
    x['leisure']=x.social_media_hours+x.gaming_hours
    x['accounted']=x.social_media_hours+x.gaming_hours+x.work_study_hours
    x['screen_residual']=x.daily_screen_time_hours-x.accounted
    x['weekend_gap']=x.weekend_screen_time-x.daily_screen_time_hours
    x['leisure_share']=x.leisure/(x.daily_screen_time_hours+.25)
    x['notif_per_open']=x.notifications_per_day/(x.app_opens_per_day+2)
    return x.replace([np.inf,-np.inf],np.nan)

def encode(fit, apply, y, cols):
    z=apply.copy(); prior=float(np.mean(y))
    for c in cols:
        a=fit[c].astype('string').fillna('__NA__'); b=apply[c].astype('string').fillna('__NA__')
        s=pd.DataFrame({'k':a,'y':np.asarray(y)}).groupby('k',observed=True).y.agg(['sum','count'])
        z[c+'__te']=b.map((s['sum']+50*prior)/(s['count']+50)).fillna(prior).astype('float32')
        z[c+'__freq']=np.log1p(b.map(s['count']).fillna(0)).astype('float32')
    return z

def cats(a,b):
    cc=list(a.select_dtypes(exclude='number').columns)
    for c in cc:
        lev=pd.Index(pd.concat([a[c],b[c]]).astype('string').fillna('Missing').unique()); dt=pd.CategoricalDtype(lev)
        a[c]=a[c].astype('string').fillna('Missing').astype(dt); b[c]=b[c].astype('string').fillna('Missing').astype(dt)
    return cc

p=find_data(); tr=pd.read_csv(p); te=pd.read_csv(p.with_name('test.csv')); y=tr[TARGET].astype('int8')
raw=[c for c in te if c!=ID]; a0,b0=base(tr),base(te)
skf=StratifiedKFold(5,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); pred=np.zeros(len(te)); fold=np.full(len(tr),-1); scores=[]; iterations=[]
for k,(i,v) in enumerate(skf.split(a0,y)):
    a=encode(a0.iloc[i],a0.iloc[i],y.iloc[i],raw); b=encode(a0.iloc[i],a0.iloc[v],y.iloc[i],raw); c=cats(a,b)
    m=lgb.LGBMClassifier(objective='binary',n_estimators=3500,learning_rate=.025,num_leaves=31,min_child_samples=150,subsample=.85,subsample_freq=1,colsample_bytree=.88,reg_alpha=.15,reg_lambda=2.5,random_state=SEED+k,n_jobs=-1,verbosity=-1)
    m.fit(a,y.iloc[i],eval_set=[(b,y.iloc[v])],eval_metric='auc',categorical_feature=c,callbacks=[lgb.early_stopping(180,verbose=False)])
    oof[v]=m.predict_proba(b)[:,1]; fold[v]=k; iterations.append(m.best_iteration_); scores.append(roc_auc_score(y.iloc[v],oof[v]))
    bt=encode(a0.iloc[i],b0,y.iloc[i],raw); cats(a,bt); pred+=m.predict_proba(bt)[:,1]/5
pd.DataFrame({ID:tr[ID],TARGET:y,'fold':fold,'prediction':oof}).to_csv('/kaggle/working/oof.csv',index=False)
pd.DataFrame({ID:te[ID],TARGET:pred}).to_csv('/kaggle/working/test_predictions.csv',index=False)
pd.DataFrame({ID:tr[ID],'fold':fold}).to_csv('/kaggle/working/folds.csv',index=False)
r={'model':'LightGBM exact-value TE/frequency','seed':SEED,'fold_auc':scores,'oof_auc':roc_auc_score(y,oof),'iterations':iterations,'provenance':'competition train only'}
Path('/kaggle/working/metrics.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
