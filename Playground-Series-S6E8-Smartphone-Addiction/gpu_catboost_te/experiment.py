"""GPU CatBoost on nested exact-value evidence; official data only."""
from pathlib import Path
import gc,json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
TARGET,ID,SEED,FOLDS='addicted_label','id',20260803,5
def locate():
    for p in Path('/kaggle/input').rglob('train.csv'):
        try: cols=pd.read_csv(p,nrows=1).columns
        except Exception: continue
        if TARGET in cols and (p.parent/'test.csv').exists(): return p,p.parent/'test.csv'
    raise FileNotFoundError('official competition tables not found')
def key(s): return s.astype('string').fillna('__NA__')
def base(df):
    r=df.drop(columns=[ID,TARGET],errors='ignore'); x=pd.DataFrame(index=df.index)
    nums=list(r.select_dtypes(include=np.number).columns)
    for c in nums: x['raw_'+c]=r[c].astype('float32'); x['na_'+c]=r[c].isna().astype('int8')
    x['missing_count']=r.isna().sum(axis=1).astype('int8'); x['is_complete']=(~r.isna().any(axis=1)).astype('int8')
    x['missing_pattern']=sum(r[c].isna().astype('int32')*(2**i) for i,c in enumerate(r.columns))
    if all(c in r for c in ['daily_screen_time_hours','social_media_hours','gaming_hours','work_study_hours']):
        x['accounted']=r.social_media_hours+r.gaming_hours+r.work_study_hours
        x['screen_residual']=r.daily_screen_time_hours-x.accounted
        x['leisure_share']=(r.social_media_hours+r.gaming_hours)/(r.daily_screen_time_hours+.25)
        x['work_share']=r.work_study_hours/(r.daily_screen_time_hours+.25)
    for n,a,b,e in [('weekend_ratio','weekend_screen_time','daily_screen_time_hours',.25),('screen_sleep','daily_screen_time_hours','sleep_hours',.25),('notif_open','notifications_per_day','app_opens_per_day',1.)]:
        if a in r and b in r: x[n]=r[a]/(r[b]+e)
    return x.replace([np.inf,-np.inf],np.nan),list(r.columns)
def stats(frame,y,c,smooth=30.):
    z=pd.DataFrame({'k':key(frame[c]).to_numpy(),'y':y}); s=z.groupby('k').y.agg(['sum','count']); prior=float(np.mean(y)); return (s['sum']+smooth*prior)/(s['count']+smooth),s['count'],prior
def inner_fit(frame,y,cols,seed):
    out=pd.DataFrame(index=np.arange(len(frame))); inner=StratifiedKFold(5,shuffle=True,random_state=seed)
    for c in cols:
        te=np.full(len(frame),np.mean(y),dtype='float32'); fq=np.zeros(len(frame),dtype='float32')
        for a,b in inner.split(frame,y):
            mp,ct,pr=stats(frame.iloc[a],y[a],c); k=key(frame.iloc[b][c]); te[b]=k.map(mp).fillna(pr); fq[b]=np.log1p(k.map(ct).fillna(0))
        out[c+'__te']=te; out[c+'__logfreq']=fq
    return out
def apply(fit,y,other,cols):
    out=pd.DataFrame(index=np.arange(len(other)))
    for c in cols:
        mp,ct,pr=stats(fit,y,c); k=key(other[c]); out[c+'__te']=k.map(mp).fillna(pr).to_numpy('float32'); out[c+'__logfreq']=np.log1p(k.map(ct).fillna(0)).to_numpy('float32')
    return out
tp,sp=locate(); tr=pd.read_csv(tp); te=pd.read_csv(sp); y=tr[TARGET].to_numpy('int8'); btr,cols=base(tr); bte,_=base(te)
outer=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); tests=[]; fid=np.zeros(len(tr),'int8'); rows=[]
for fold,(a,b) in enumerate(outer.split(tr,y),1):
    fid[b]=fold; ein=inner_fit(tr.iloc[a].reset_index(drop=True),y[a],cols,SEED+fold); ev=apply(tr.iloc[a],y[a],tr.iloc[b],cols); et=apply(tr.iloc[a],y[a],te,cols)
    med=btr.iloc[a].median(); xa=pd.concat([btr.iloc[a].reset_index(drop=True).fillna(med),ein],axis=1); xv=pd.concat([btr.iloc[b].reset_index(drop=True).fillna(med),ev],axis=1); xt=pd.concat([bte.reset_index(drop=True).fillna(med),et],axis=1)
    model=CatBoostClassifier(iterations=2600,depth=7,learning_rate=.035,loss_function='Logloss',eval_metric='AUC',l2_leaf_reg=5.,random_strength=.35,random_seed=SEED+fold,task_type='GPU',devices='0',bootstrap_type='Bayesian',bagging_temperature=.4,verbose=200,allow_writing_files=False)
    model.fit(xa,y[a],eval_set=(xv,y[b]),early_stopping_rounds=180,verbose=200); vp=model.predict_proba(xv)[:,1]; oof[b]=vp; tests.append(model.predict_proba(xt)[:,1]); score=roc_auc_score(y[b],vp); rows.append({'fold':fold,'auc':score,'best_iteration':model.get_best_iteration()}); print(rows[-1],flush=True); del model,xa,xv,xt,ein,ev,et; gc.collect()
pred=np.mean(tests,axis=0); pd.DataFrame({ID:tr[ID],'fold':fid,'y':y,'pred':oof}).to_csv('/kaggle/working/oof_nested_te_catboost.csv',index=False); pd.DataFrame({ID:te[ID],TARGET:pred}).to_csv('/kaggle/working/test_nested_te_catboost.csv',index=False)
scores=[r['auc'] for r in rows]; metrics={'model':'CatBoost GPU on numeric nested exact-value TE/frequency','official_data_only':True,'seed':SEED,'folds':FOLDS,'depth':7,'learning_rate':.035,'fold_metrics':rows,'fold_mean':float(np.mean(scores)),'fold_std':float(np.std(scores)),'oof_auc':roc_auc_score(y,oof),'submission_created':False}
Path('/kaggle/working/metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
