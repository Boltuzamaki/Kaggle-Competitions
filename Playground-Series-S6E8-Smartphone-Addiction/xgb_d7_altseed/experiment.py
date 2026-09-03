"""Bounded nested-encoding XGBoost HPO, independently trained on official S6E8 data."""
from pathlib import Path
import gc, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

TARGET, ID, SEED = "addicted_label", "id", 20260881
CONFIGS = {
 "d7_altseed": dict(max_depth=7,min_child_weight=28,reg_alpha=.45,reg_lambda=12.,gamma=.025,learning_rate=.017,colsample_bytree=.78),
}

def locate():
    for root in (Path("/kaggle/input"),Path(".")):
        for p in root.rglob("train.csv"):
            try: cols=pd.read_csv(p,nrows=2).columns
            except Exception: continue
            if TARGET in cols and (p.parent/"test.csv").exists(): return p,p.parent/"test.csv"
    raise FileNotFoundError("official competition train/test not found")

def make_features(df):
    x=df.drop(columns=[ID,TARGET],errors="ignore").copy(); raw=list(x.columns); miss=x[raw].isna()
    x["missing_count"]=miss.sum(1).astype("int8")
    x["missing_pattern"]=miss.astype("uint8").astype(str).agg("".join,axis=1)
    for c in raw: x[c+"__missing"]=miss[c].astype("int8")
    x["leisure_hours"]=x.social_media_hours+x.gaming_hours
    x["accounted_hours"]=x.leisure_hours+x.work_study_hours
    x["unaccounted_screen"]=x.daily_screen_time_hours-x.accounted_hours
    x["weekend_gap"]=x.weekend_screen_time-x.daily_screen_time_hours
    x["notifications_per_open"]=x.notifications_per_day/(x.app_opens_per_day+3.)
    x["opens_per_screen_hour"]=x.app_opens_per_day/(x.daily_screen_time_hours+.5)
    for n,c in (("social","social_media_hours"),("gaming","gaming_hours"),("work","work_study_hours")):
        x[n+"_screen_share"]=x[c]/(x.daily_screen_time_hours+.5)
    return x.replace([np.inf,-np.inf],np.nan)

def key(s): return s.astype("string").fillna("__NA__")

def maps(s,y,smooth=35.):
    z=pd.DataFrame({"k":key(s).to_numpy(),"y":np.asarray(y)}); st=z.groupby("k",observed=True).y.agg(["sum","count"]); prior=float(np.mean(y))
    return ((st["sum"]+smooth*prior)/(st["count"]+smooth)).to_dict(),st["count"].to_dict(),prior

def encoded(fit,yfit,valid,test,inner_seed):
    """Inner-OOF TE for fit; outer-fit-only maps for valid/test."""
    a=fit.reset_index(drop=True).copy(); b=valid.reset_index(drop=True).copy(); c=test.reset_index(drop=True).copy(); yy=np.asarray(yfit)
    enc_cols=list(a.columns); inner=StratifiedKFold(4,shuffle=True,random_state=inner_seed)
    for col in enc_cols:
        oo=np.full(len(a),yy.mean(),dtype="float32")
        for ii,jj in inner.split(np.zeros(len(a)),yy):
            mp,_,pr=maps(a.iloc[ii][col],yy[ii]); oo[jj]=key(a.iloc[jj][col]).map(mp).fillna(pr).to_numpy("float32")
        mp,ct,pr=maps(a[col],yy)
        a[col+"__te"]=oo; b[col+"__te"]=key(b[col]).map(mp).fillna(pr).to_numpy("float32"); c[col+"__te"]=key(c[col]).map(mp).fillna(pr).to_numpy("float32")
        for f in (a,b,c): f[col+"__logfreq"]=np.log1p(key(f[col]).map(ct).fillna(0)).to_numpy("float32")
    for col in a.select_dtypes(exclude="number"):
        levels=pd.Index(pd.concat([key(a[col]),key(b[col]),key(c[col])],ignore_index=True).unique()); dtype=pd.CategoricalDtype(levels)
        for f in (a,b,c): f[col]=key(f[col]).astype(dtype).cat.codes.astype("float32")
    return a.astype("float32"),b.astype("float32"),c.astype("float32")

def fit(config,xf,yf,xv,yv,xt,seed):
    common=dict(n_estimators=4200,subsample=.84,objective="binary:logistic",eval_metric="auc",random_state=seed,n_jobs=-1,tree_method="hist",device="cuda",early_stopping_rounds=180,max_bin=192)
    m=XGBClassifier(**common,**config); m.fit(xf,yf,eval_set=[(xv,yv)],verbose=False)
    return m.predict_proba(xv)[:,1],m.predict_proba(xt)[:,1],int(m.best_iteration+1)

trp,tep=locate(); train,test=pd.read_csv(trp),pd.read_csv(tep); y=train[TARGET].to_numpy("int8"); x,xt=make_features(train),make_features(test)
outer=StratifiedKFold(5,shuffle=True,random_state=SEED); oofs={k:np.zeros(len(train)) for k in CONFIGS}; fold_id=np.zeros(len(train),dtype="int8"); tests={k:[] for k in CONFIGS}; rows=[]
for fold,(fi,vi) in enumerate(outer.split(x,y),1):
    a,b,c=encoded(x.iloc[fi],y[fi],x.iloc[vi],xt,SEED+500+fold)
    for name,cfg in CONFIGS.items():
        pv,pt,it=fit(cfg,a,y[fi],b,y[vi],c,SEED+1000+fold); oofs[name][vi]=pv; tests[name].append(pt)
        row={"fold":fold,"config":name,"auc":float(roc_auc_score(y[vi],pv)),"best_iteration":it}; rows.append(row); print(row,flush=True)
    fold_id[vi]=fold
    del a,b,c; gc.collect()
scores={k:float(roc_auc_score(y,v)) for k,v in oofs.items()}; stds={k:float(np.std([r['auc'] for r in rows if r['config']==k],ddof=1)) for k in CONFIGS}
pd.DataFrame({ID:train[ID],"fold":fold_id,"y":y,**{"pred_"+k:v for k,v in oofs.items()}}).to_csv("/kaggle/working/oof_d7_altseed_xgb.csv",index=False)
pd.DataFrame({ID:test[ID],**{"pred_"+k:np.mean(v,axis=0) for k,v in tests.items()}}).to_csv("/kaggle/working/test_d7_altseed_xgb.csv",index=False)
metrics={"provenance":"independently trained; official competition data only; no public predictions","split_seed":SEED,"folds":5,"inner_folds":4,"configs":CONFIGS,"fold_metrics":rows,"oof_auc":scores,"fold_auc_std":stds,"purpose":"outer-fold and nested-mapping diversity relative to seed 20260803 depth-7 stream"}
Path("/kaggle/working/metrics_d7_altseed_xgb.json").write_text(json.dumps(metrics,indent=2)+"\n"); print(json.dumps(metrics,indent=2),flush=True)
