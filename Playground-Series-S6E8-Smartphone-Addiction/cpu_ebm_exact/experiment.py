"""Leakage-safe exact-value evidence + explainable boosting GAM."""
from pathlib import Path
import gc, json, subprocess, sys
subprocess.check_call([sys.executable,"-m","pip","install","-q","interpret==0.7.2"])
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET,ID,SEED,FOLDS="addicted_label","id",20260803,5
def locate():
    for p in Path('/kaggle/input').rglob('train.csv'):
        try: cols=pd.read_csv(p,nrows=1).columns
        except Exception: continue
        if TARGET in cols and (p.parent/'test.csv').exists(): return p,p.parent/'test.csv'
    raise FileNotFoundError('official tables not found')
def key(s): return s.astype('string').fillna('__NA__')
def base(df):
    raw=df.drop(columns=[ID,TARGET],errors='ignore'); out=pd.DataFrame(index=df.index)
    for c in raw:
        if pd.api.types.is_numeric_dtype(raw[c]):
            out['raw_'+c]=raw[c].astype('float32'); out['na_'+c]=raw[c].isna().astype('int8')
    out['missing_count']=raw.isna().sum(axis=1).astype('int8')
    out['is_complete']=(~raw.isna().any(axis=1)).astype('int8')
    if all(c in raw for c in ['daily_screen_time_hours','social_media_hours','gaming_hours','work_study_hours']):
        out['screen_residual']=raw.daily_screen_time_hours-raw.social_media_hours-raw.gaming_hours-raw.work_study_hours
        out['leisure_share']=(raw.social_media_hours+raw.gaming_hours)/(raw.daily_screen_time_hours+.25)
    return out.replace([np.inf,-np.inf],np.nan),list(raw.columns)
def maps(fit,y,col,smooth):
    z=pd.DataFrame({'k':key(fit[col]).to_numpy(),'y':y}); st=z.groupby('k').y.agg(['sum','count']); prior=float(np.mean(y))
    te=(st['sum']+smooth*prior)/(st['count']+smooth)
    return te,st['count'],prior
def encode(fit,y,apply,cols,smooth):
    out=pd.DataFrame(index=apply.index)
    for c in cols:
        te,ct,prior=maps(fit,y,c,smooth); k=key(apply[c])
        out['evidence_'+c]=k.map(te).fillna(prior).astype('float32')
        out['logfreq_'+c]=np.log1p(k.map(ct).fillna(0)).astype('float32')
    return out
def encode_inner(fit,y,cols,smooth,seed):
    out=pd.DataFrame(index=fit.index); inner=StratifiedKFold(4,shuffle=True,random_state=seed)
    for c in cols:
        v=np.full(len(fit),float(np.mean(y)),dtype='float32'); fq=np.zeros(len(fit),dtype='float32')
        for a,b in inner.split(fit,y):
            te,ct,prior=maps(fit.iloc[a],y[a],c,smooth); k=key(fit.iloc[b][c]); v[b]=k.map(te).fillna(prior); fq[b]=np.log1p(k.map(ct).fillna(0))
        out['evidence_'+c]=v; out['logfreq_'+c]=fq
    return out

tp,sp=locate(); tr=pd.read_csv(tp); te=pd.read_csv(sp); y=tr[TARGET].to_numpy('int8'); btr,cols=base(tr); bte,_=base(te)
outer=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); tests=[]; fid=np.zeros(len(tr),'int8'); rows=[]
# Single bounded configuration; smoothing 30 is a conservative prespecified value.
SMOOTH=30.
for fold,(a,b) in enumerate(outer.split(tr,y),1):
    fid[b]=fold; ein=encode_inner(tr.iloc[a].reset_index(drop=True),y[a],cols,SMOOTH,SEED+fold)
    ev=encode(tr.iloc[a],y[a],tr.iloc[b],cols,SMOOTH).reset_index(drop=True); et=encode(tr.iloc[a],y[a],te,cols,SMOOTH).reset_index(drop=True)
    med=btr.iloc[a].median(); xa=pd.concat([btr.iloc[a].reset_index(drop=True).fillna(med),ein],axis=1); xv=pd.concat([btr.iloc[b].reset_index(drop=True).fillna(med),ev],axis=1); xt=pd.concat([bte.reset_index(drop=True).fillna(med),et],axis=1)
    model=ExplainableBoostingClassifier(interactions=12,max_bins=256,max_interaction_bins=32,learning_rate=.035,max_rounds=1800,min_samples_leaf=80,outer_bags=8,inner_bags=0,validation_size=.12,early_stopping_rounds=100,n_jobs=-1,random_state=SEED+fold)
    model.fit(xa,y[a]); vp=model.predict_proba(xv)[:,1]; oof[b]=vp; tests.append(model.predict_proba(xt)[:,1]); score=roc_auc_score(y[b],vp); rows.append({'fold':fold,'auc':score}); print(rows[-1],flush=True); del model,xa,xv,xt,ein,ev,et; gc.collect()
pred=np.mean(tests,axis=0); pd.DataFrame({ID:tr[ID],'fold':fid,'y':y,'pred':oof}).to_csv('/kaggle/working/oof_exact_ebm.csv',index=False); pd.DataFrame({ID:te[ID],TARGET:pred}).to_csv('/kaggle/working/test_exact_ebm.csv',index=False)
metrics={'model':'exact-value cross-fitted evidence ExplainableBoosting GAM','official_data_only':True,'seed':SEED,'folds':FOLDS,'smooth':SMOOTH,'fold_auc':[r['auc'] for r in rows],'fold_mean':float(np.mean([r['auc'] for r in rows])),'fold_std':float(np.std([r['auc'] for r in rows])),'oof_auc':roc_auc_score(y,oof),'submission_created':False}
Path('/kaggle/working/metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
