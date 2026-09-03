"""Cross-fitted empirical-Bayes weight-of-evidence additive model."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts'/'local_eb_woe'; OUT.mkdir(parents=True,exist_ok=True)
TARGET,ID,SEED,FOLDS='addicted_label','id',20260803,5; SMOOTH=(10.,30.,80.)
def key(s): return s.astype('string').fillna('__NA__')
def stat(frame,y,c):
    z=pd.DataFrame({'k':key(frame[c]).to_numpy(),'y':y}); return z.groupby('k').y.agg(['sum','count'])
def evidence(st,k,prior,smooth):
    p=(st['sum']+smooth*prior)/(st['count']+smooth); q=k.map(p).fillna(prior).clip(1e-5,1-1e-5); return np.log(q/(1-q)).to_numpy('float32')
def base(df):
    r=df.drop(columns=[ID,TARGET],errors='ignore'); x=pd.DataFrame(index=df.index); miss=r.isna()
    for c in r: x['na_'+c]=miss[c].astype('float32')
    x['missing_count']=miss.sum(axis=1).astype('float32'); x['is_complete']=(~miss.any(axis=1)).astype('float32')
    s=r.social_media_hours+r.gaming_hours+r.work_study_hours
    x['screen_residual']=r.daily_screen_time_hours-s; x['component_sum']=s; x['leisure_share']=(r.social_media_hours+r.gaming_hours)/(r.daily_screen_time_hours+.25); x['work_share']=r.work_study_hours/(r.daily_screen_time_hours+.25); x['weekend_gap']=r.weekend_screen_time-r.daily_screen_time_hours; x['weekend_ratio']=r.weekend_screen_time/(r.daily_screen_time_hours+.25); x['screen_sleep']=r.daily_screen_time_hours/(r.sleep_hours+.25); x['notif_open']=r.notifications_per_day/(r.app_opens_per_day+1)
    for c in list(x.columns):
        if not c.startswith('na_') and c not in ('is_complete',): x[c+'__sq']=x[c]**2
    return x.replace([np.inf,-np.inf],np.nan)
def encode_fold(fit,y,valid,test,cols,smooth,seed):
    prior=float(y.mean()); inner=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(fit,y)); a=pd.DataFrame(index=np.arange(len(fit))); b=pd.DataFrame(index=np.arange(len(valid))); c=pd.DataFrame(index=np.arange(len(test)))
    for col in cols:
        oo=np.zeros(len(fit),'float32'); of=np.zeros(len(fit),'float32')
        for ii,jj in inner:
            st=stat(fit.iloc[ii],y[ii],col); kk=key(fit.iloc[jj][col]); oo[jj]=evidence(st,kk,float(y[ii].mean()),smooth); of[jj]=np.log1p(kk.map(st['count']).fillna(0)).to_numpy('float32')
        st=stat(fit,y,col); kb=key(valid[col]); kc=key(test[col]); a['woe_'+col]=oo; a['freq_'+col]=of; b['woe_'+col]=evidence(st,kb,prior,smooth); c['woe_'+col]=evidence(st,kc,prior,smooth); b['freq_'+col]=np.log1p(kb.map(st['count']).fillna(0)).to_numpy('float32'); c['freq_'+col]=np.log1p(kc.map(st['count']).fillna(0)).to_numpy('float32')
    return a,b,c
tr=pd.read_csv(ROOT/'train.csv'); te=pd.read_csv(ROOT/'test.csv'); y=tr[TARGET].to_numpy('int8'); cols=[c for c in te if c!=ID]; bt,be=base(tr),base(te); outer=list(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(tr,y)); fold_id=np.zeros(len(tr),'int8')
oofs={s:np.zeros(len(tr)) for s in SMOOTH}; tests={s:[] for s in SMOOTH}; rows=[]; started=time.time()
for fold,(fi,vi) in enumerate(outer,1):
    fold_id[vi]=fold
    for smooth in SMOOTH:
        ea,ev,et=encode_fold(tr.iloc[fi].reset_index(drop=True),y[fi],tr.iloc[vi].reset_index(drop=True),te,cols,smooth,SEED+fold)
        xa=pd.concat([bt.iloc[fi].reset_index(drop=True),ea],axis=1); xv=pd.concat([bt.iloc[vi].reset_index(drop=True),ev],axis=1); xt=pd.concat([be.reset_index(drop=True),et],axis=1)
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.20,max_iter=250,solver='lbfgs',n_jobs=-1))
        model.fit(xa,y[fi]); pv=model.predict_proba(xv)[:,1]; oofs[smooth][vi]=pv; tests[smooth].append(model.predict_proba(xt)[:,1]); score=roc_auc_score(y[vi],pv); rows.append({'fold':fold,'smooth':smooth,'auc':score}); print(rows[-1],flush=True)
scores={str(s):float(roc_auc_score(y,oofs[s])) for s in SMOOTH}; best=max(SMOOTH,key=lambda s:scores[str(s)]); test_pred=np.mean(tests[best],axis=0)
out=pd.DataFrame({ID:tr[ID],'fold':fold_id,'y':y});
for s in SMOOTH: out[f'pred_smooth_{int(s)}']=oofs[s]
out.to_csv(OUT/'oof.csv',index=False); pd.DataFrame({ID:te[ID],TARGET:test_pred}).to_csv(OUT/'test_predictions.csv',index=False); pd.DataFrame({ID:tr[ID],'fold':fold_id}).to_csv(OUT/'folds.csv',index=False)
metrics={'model':'cross-fitted empirical-Bayes WOE additive logistic','official_data_only':True,'seed':SEED,'folds':FOLDS,'smoothing_grid':SMOOTH,'C':.20,'fold_metrics':rows,'oof_auc':scores,'selected_smoothing':best,'runtime_seconds':time.time()-started,'submission_created':False}; (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
