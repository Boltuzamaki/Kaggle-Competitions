"""S6E8 exact-value representations and missingness experts.

From-scratch experiment: official competition train/test only.  All reported
training predictions use nested encodings and a common five-fold outer split.
"""
from pathlib import Path
import gc, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID, SEED, NFOLD = "addicted_label", "id", 20260803, 5

def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        for p in root.rglob("train.csv"):
            try: cols = pd.read_csv(p, nrows=2).columns
            except Exception: continue
            if TARGET in cols and (p.parent/"test.csv").exists() and (p.parent/"sample_submission.csv").exists():
                return p, p.parent/"test.csv", p.parent/"sample_submission.csv"
    raise FileNotFoundError("official S6E8 train/test/sample files not found")

def base(df):
    x=df.drop(columns=[ID,TARGET],errors="ignore").copy()
    raw=list(x.columns); nums=list(x.select_dtypes("number").columns)
    miss=x[raw].isna()
    x["missing_count"]=miss.sum(1).astype("int8")
    x["missing_pattern"]=miss.astype("uint8").astype(str).agg("".join,axis=1)
    for c in nums: x[c+"__missing"]=x[c].isna().astype("int8")
    x["leisure_hours"]=x.social_media_hours+x.gaming_hours
    x["accounted_hours"]=x.social_media_hours+x.gaming_hours+x.work_study_hours
    x["unaccounted_screen"]=x.daily_screen_time_hours-x.accounted_hours
    x["weekend_gap"]=x.weekend_screen_time-x.daily_screen_time_hours
    for name,a,b,off in [
        ("weekend_ratio","weekend_screen_time","daily_screen_time_hours",.25),
        ("social_share","social_media_hours","daily_screen_time_hours",.25),
        ("gaming_share","gaming_hours","daily_screen_time_hours",.25),
        ("work_share","work_study_hours","daily_screen_time_hours",.25),
        ("notif_open_ratio","notifications_per_day","app_opens_per_day",2),
        ("screen_sleep_ratio","daily_screen_time_hours","sleep_hours",.25)]:
        x[name]=x[a]/(x[b]+off)
    return x.replace([np.inf,-np.inf],np.nan), raw

def key(s): return s.astype("string").fillna("__NA__")

def learn_map(s,y,smooth=35.):
    z=pd.DataFrame({"k":key(s).to_numpy(),"y":np.asarray(y)})
    st=z.groupby("k",observed=True).y.agg(["sum","count"]); prior=float(np.mean(y))
    return ((st["sum"]+smooth*prior)/(st["count"]+smooth)).to_dict(),st["count"].to_dict(),prior

def nested_features(fit0, yfit, valid0, test0, enc_cols, fold):
    """Inner-OOF encodings for fit; outer-fit mappings for valid and test."""
    a=fit0.copy().reset_index(drop=True); b=valid0.copy().reset_index(drop=True); c=test0.copy().reset_index(drop=True)
    yin=np.asarray(yfit); inner=StratifiedKFold(NFOLD,shuffle=True,random_state=SEED+fold)
    for col in enc_cols:
        oo=np.full(len(a),yin.mean(),dtype="float32")
        for ii,jj in inner.split(np.zeros(len(a)),yin):
            mp,_,pr=learn_map(a.iloc[ii][col],yin[ii])
            oo[jj]=key(a.iloc[jj][col]).map(mp).fillna(pr).to_numpy("float32")
        mp,ct,pr=learn_map(a[col],yin)
        a[col+"__te"]=oo
        b[col+"__te"]=key(b[col]).map(mp).fillna(pr).astype("float32")
        c[col+"__te"]=key(c[col]).map(mp).fillna(pr).astype("float32")
        # Frequency has no labels, but is still learned from the outer fit only.
        for frame in (a,b,c): frame[col+"__logfreq"]=np.log1p(key(frame[col]).map(ct).fillna(0)).astype("float32")
    cats=list(a.select_dtypes(exclude="number").columns)
    for col in cats:
        levels=pd.Index(pd.concat([key(a[col]),key(b[col]),key(c[col])],ignore_index=True).unique())
        dtype=pd.CategoricalDtype(levels)
        for frame in (a,b,c): frame[col]=key(frame[col]).astype(dtype)
    return a,b,c,cats

def model(seed, leaves=31):
    return lgb.LGBMClassifier(objective="binary",n_estimators=2200,learning_rate=.035,
      num_leaves=leaves,min_child_samples=180,subsample=.85,subsample_freq=1,
      colsample_bytree=.88,reg_alpha=.15,reg_lambda=3.,random_state=seed,n_jobs=-1,verbosity=-1)

def fit_predict(xf,yf,xv,yv,xt,cats,seed,leaves=31):
    m=model(seed,leaves); m.fit(xf,yf,eval_set=[(xv,yv)],eval_metric="auc",categorical_feature=cats,
      callbacks=[lgb.early_stopping(140,verbose=False)])
    return m.predict_proba(xv)[:,1],m.predict_proba(xt)[:,1],int(m.best_iteration_)

trp,tep,smp=locate(); train,test,sample=map(pd.read_csv,(trp,tep,smp))
y=train[TARGET].to_numpy("int8"); tr0,raw=base(train); te0,_=base(test)
# Encode exact raw values and the missingness fingerprint. Engineered continuous
# values remain raw because most are higher-cardinality combinations.
enc_cols=raw+["missing_pattern"]
outer=StratifiedKFold(NFOLD,shuffle=True,random_state=SEED)
oof_g=np.zeros(len(train)); oof_e=np.zeros(len(train)); tests_g=[]; tests_e=[]; rows=[]
complete=tr0.missing_count.eq(0).to_numpy(); te_complete=te0.missing_count.eq(0).to_numpy()
for fold,(fi,vi) in enumerate(outer.split(tr0,y),1):
    a,b,c,cats=nested_features(tr0.iloc[fi],y[fi],tr0.iloc[vi],te0,enc_cols,fold)
    pg,tg,itg=fit_predict(a,y[fi],b,y[vi],c,cats,SEED+fold,31); oof_g[vi]=pg; tests_g.append(tg)
    pe=np.zeros(len(vi)); tepred=np.zeros(len(test)); expert_its={}
    for regime in (True,False):
        fm=complete[fi]==regime; vm=complete[vi]==regime; tm=te_complete==regime
        # Experts see only their own missingness regime.
        pv,pt,it=fit_predict(a.loc[fm],y[fi][fm],b.loc[vm],y[vi][vm],c.loc[tm],cats,SEED+100+fold+int(regime),31)
        pe[vm]=pv; tepred[tm]=pt; expert_its[str(regime)]=it
    oof_e[vi]=pe; tests_e.append(tepred)
    rows.append({"fold":fold,"global_auc":roc_auc_score(y[vi],pg),"expert_auc":roc_auc_score(y[vi],pe),"global_iteration":itg,"expert_iterations":expert_its})
    print(rows[-1],flush=True); del a,b,c; gc.collect()

# Fixed convex candidates are retained so a later stack can choose using common OOF.
out=pd.DataFrame({ID:train[ID],TARGET:y,"pred_global":oof_g,"pred_regime_experts":oof_e})
test_g=np.mean(tests_g,axis=0); test_e=np.mean(tests_e,axis=0)
scores={"global":roc_auc_score(y,oof_g),"regime_experts":roc_auc_score(y,oof_e)}
for w in (.2,.4,.6,.8):
    name=f"blend_expert_{w:.1f}"; out[name]=(1-w)*oof_g+w*oof_e; scores[name]=roc_auc_score(y,out[name])
best=max(scores,key=scores.get); w=0 if best=="global" else (1 if best=="regime_experts" else float(best.rsplit("_",1)[1]))
pred=(1-w)*test_g+w*test_e
out.to_csv("/kaggle/working/oof_representation_lgbm.csv",index=False)
pd.DataFrame({ID:test[ID],"pred_global":test_g,"pred_regime_experts":test_e}).to_csv("/kaggle/working/test_representation_lgbm.csv",index=False)
sub=sample.copy(); sub[TARGET]=pred; sub.to_csv("/kaggle/working/submission_representation_lgbm.csv",index=False)
metrics={"provenance":"official competition data only; independently trained","folds":NFOLD,"seed":SEED,"fold_metrics":rows,"oof_auc":scores,"selected":best,"selected_expert_weight":w}
Path("/kaggle/working/metrics_representation_lgbm.json").write_text(json.dumps(metrics,indent=2)+"\n")
print(json.dumps(metrics,indent=2),flush=True)
