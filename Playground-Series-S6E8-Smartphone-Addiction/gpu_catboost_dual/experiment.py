"""Official-only dual numeric/exact-key CatBoost companion."""
from pathlib import Path
import gc,json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
TARGET,ID,SEED,FOLDS='addicted_label','id',20260803,5
def locate():
    for p in Path('/kaggle/input').rglob('train.csv'):
        try: cols=pd.read_csv(p,nrows=1).columns
        except Exception: continue
        if TARGET in cols and (p.parent/'test.csv').exists(): return p,p.parent/'test.csv'
    raise FileNotFoundError('official competition tables not found')
def exact_key(s):
    # StringDtype canonicalizes missingness before conversion to object strings.
    return s.astype('string').fillna('__MISSING__').astype(str)
def features(df):
    raw=df.drop(columns=[ID,TARGET],errors='ignore'); x=pd.DataFrame(index=df.index); cats=[]
    for c in raw:
        if pd.api.types.is_numeric_dtype(raw[c]): x['num__'+c]=raw[c].astype('float32')
        name='key__'+c; x[name]=exact_key(raw[c]); cats.append(name)
        x['missing__'+c]=raw[c].isna().astype('int8')
    miss=raw.isna(); x['missing_count']=miss.sum(axis=1).astype('int8'); x['is_complete']=(~miss.any(axis=1)).astype('int8')
    comp=['social_media_hours','gaming_hours','work_study_hours']; s=raw[comp].sum(axis=1,min_count=3)
    x['component_sum']=s; x['other_screen']=raw.daily_screen_time_hours-s; x['other_frac']=(raw.daily_screen_time_hours-s)/(raw.daily_screen_time_hours+.25); x['component_frac']=s/(raw.daily_screen_time_hours+.25); x['weekend_gap']=raw.weekend_screen_time-raw.daily_screen_time_hours; x['weekend_other']=raw.weekend_screen_time-s; x['screen_sleep']=raw.daily_screen_time_hours/(raw.sleep_hours+.25); x['notif_open']=raw.notifications_per_day/(raw.app_opens_per_day+1)
    return x.replace([np.inf,-np.inf],np.nan),cats
tp,sp=locate(); tr=pd.read_csv(tp); te=pd.read_csv(sp); y=tr[TARGET].to_numpy('int8'); X,cats=features(tr); XT,_=features(te)
outer=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); tests=[]; fold_id=np.zeros(len(tr),'int8'); rows=[]
for fold,(a,b) in enumerate(outer.split(X,y),1):
    fold_id[b]=fold; pa=Pool(X.iloc[a],y[a],cat_features=cats); pv=Pool(X.iloc[b],y[b],cat_features=cats); pt=Pool(XT,cat_features=cats)
    model=CatBoostClassifier(iterations=3600,depth=8,learning_rate=.035,l2_leaf_reg=6.,random_strength=.45,loss_function='Logloss',eval_metric='AUC',random_seed=SEED+fold,task_type='GPU',devices='0',bootstrap_type='Bayesian',bagging_temperature=.5,one_hot_max_size=4,max_ctr_complexity=2,od_type='Iter',od_wait=180,allow_writing_files=False,verbose=250)
    model.fit(pa,eval_set=pv,use_best_model=True); pred=model.predict_proba(pv)[:,1]; oof[b]=pred; tests.append(model.predict_proba(pt)[:,1]); score=roc_auc_score(y[b],pred); rows.append({'fold':fold,'auc':score,'best_iteration':model.get_best_iteration()}); print(rows[-1],flush=True); del model,pa,pv,pt; gc.collect()
test_pred=np.mean(tests,axis=0); pd.DataFrame({ID:tr[ID],'fold':fold_id,'y':y,'pred':oof}).to_csv('/kaggle/working/oof_dual_view_catboost.csv',index=False); pd.DataFrame({ID:te[ID],TARGET:test_pred}).to_csv('/kaggle/working/test_dual_view_catboost.csv',index=False); pd.DataFrame({ID:tr[ID],'fold':fold_id}).to_csv('/kaggle/working/folds.csv',index=False)
s=[r['auc'] for r in rows]; metrics={'concept_credit':'Tamerlan Omralinov dual-view CatBoost companion concept','implementation':'original five-fold official-only adaptation','official_data_only':True,'seed':SEED,'folds':FOLDS,'fold_metrics':rows,'fold_mean':float(np.mean(s)),'fold_std':float(np.std(s)),'oof_auc':roc_auc_score(y,oof),'submission_created':False}; Path('/kaggle/working/metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
