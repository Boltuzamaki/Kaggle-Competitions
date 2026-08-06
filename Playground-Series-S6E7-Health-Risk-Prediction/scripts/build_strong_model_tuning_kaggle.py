"""Build queued Optuna tuning notebooks for the strongest tree families."""
from __future__ import annotations
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OWNER=os.environ.get("KAGGLE_OWNER","boltuzamaki")

PREP=r'''
import gc,json,time,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings("ignore")
ID="id";TARGET="health_condition";SEED=2027;FOLDS=5
C=np.array(["at-risk","fit","unhealthy"]);M={c:i for i,c in enumerate(C)}
base=Path("/kaggle/input/competitions/playground-series-s6e7");tr=pd.read_csv(base/"train.csv");te=pd.read_csv(base/"test.csv");y=tr[TARGET].map(M).to_numpy()
def fe(d):
 z=d.drop(columns=[ID,TARGET],errors="ignore").copy();s=z.sleep_duration
 z["sleep_lt6"]=np.where(s.isna(),np.nan,(s<6).astype(float));z["sleep_lt7"]=np.where(s.isna(),np.nan,(s<7).astype(float))
 z["key_missing_count"]=z[["sleep_duration","stress_level","physical_activity_level"]].isna().sum(1)
 z["key_pattern"]=z[["sleep_duration","stress_level","physical_activity_level"]].isna().astype(int).astype(str).agg("".join,axis=1)
 z["sleep_zone"]=pd.cut(s,[-np.inf,6,7,np.inf],right=False,labels=["lt6","6to7","ge7"]).astype("object")
 z["rule_unhealthy"]=((s<6)&z.stress_level.eq("high")).astype(float);z["rule_fit"]=((s>=7)&z.stress_level.eq("low")&z.physical_activity_level.eq("active")).astype(float)
 for c in z.select_dtypes(exclude=["object","category"]).columns:
  z[f"{c}_frac10"]=np.floor(z[c].abs()*10+1e-7)%10
 return z
X=fe(tr);T=fe(te);cats=list(X.select_dtypes(include=["object","category"]).columns);nums=[c for c in X if c not in cats]
e=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1);e.fit(pd.concat([X[cats],T[cats]]).fillna("NA"));X[cats]=e.transform(X[cats].fillna("NA"));T[cats]=e.transform(T[cats].fillna("NA"))
med=X[nums].median();X[nums]=X[nums].fillna(med);T[nums]=T[nums].fillna(med);X=X.astype("float32");T=T.astype("float32")
'''

XGB=r'''
!pip install -q optuna
'''+PREP+r'''
import optuna
from xgboost import XGBClassifier
tune_idx,hold_idx=train_test_split(np.arange(len(X)),test_size=.16,random_state=SEED,stratify=y)
def objective(t):
 p=dict(
  max_depth=t.suggest_int("max_depth",4,10),min_child_weight=t.suggest_float("min_child_weight",8,80,log=True),
  learning_rate=t.suggest_float("learning_rate",.008,.05,log=True),reg_alpha=t.suggest_float("reg_alpha",1e-3,5,log=True),
  reg_lambda=t.suggest_float("reg_lambda",1e-3,10,log=True),gamma=t.suggest_float("gamma",0,6),
  max_delta_step=t.suggest_float("max_delta_step",0,5),subsample=t.suggest_float("subsample",.65,1),
  colsample_bytree=t.suggest_float("colsample_bytree",.65,1),n_estimators=7000,objective="multi:softprob",
  eval_metric="mlogloss",tree_method="hist",device="cuda",random_state=SEED,early_stopping_rounds=100)
 m=XGBClassifier(**p);m.fit(X.iloc[tune_idx],y[tune_idx],sample_weight=compute_sample_weight("balanced",y[tune_idx]),eval_set=[(X.iloc[hold_idx],y[hold_idx])],verbose=False)
 return balanced_accuracy_score(y[hold_idx],m.predict_proba(X.iloc[hold_idx]).argmax(1))
study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED));study.optimize(objective,n_trials=24)
best=study.best_params;print(study.best_value,best)
oof=np.zeros((len(X),3),np.float32);pred=np.zeros((len(T),3));fs=[];t0=time.time();skf=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)
for f,(a,b) in enumerate(skf.split(X,y),1):
 m=XGBClassifier(**best,n_estimators=10000,objective="multi:softprob",eval_metric="mlogloss",tree_method="hist",device="cuda",random_state=SEED+f,early_stopping_rounds=120)
 m.fit(X.iloc[a],y[a],sample_weight=compute_sample_weight("balanced",y[a]),eval_set=[(X.iloc[b],y[b])],verbose=False);oof[b]=m.predict_proba(X.iloc[b]);pred+=m.predict_proba(T)/FOLDS
 fs.append(float(balanced_accuracy_score(y[b],oof[b].argmax(1))));del m;gc.collect()
score=float(balanced_accuracy_score(y,oof.argmax(1)))
od=pd.DataFrame({ID:tr[ID]});td=pd.DataFrame({ID:te[ID]})
for j,c in enumerate(C):od[c]=oof[:,j];td[c]=pred[:,j]
od.to_csv("oof_preds.csv",index=False);td.to_csv("test_preds.csv",index=False);pd.DataFrame({ID:te[ID],TARGET:C[pred.argmax(1)]}).to_csv("submission.csv",index=False)
Path("training_summary.json").write_text(json.dumps({"experiment":"optuna_xgb","tuning_best_score":study.best_value,"best_params":best,"fold_scores":fs,"oof_balanced_accuracy":score,"elapsed_minutes":(time.time()-t0)/60},indent=2));print(score)
'''

HGB=r'''
!pip install -q optuna
'''+PREP+r'''
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
rng=np.random.default_rng(SEED);pool=rng.choice(len(X),size=360000,replace=False);a,b=train_test_split(pool,test_size=.20,random_state=SEED,stratify=y[pool])
def objective(t):
 p=dict(learning_rate=t.suggest_float("learning_rate",.025,.14,log=True),max_iter=t.suggest_int("max_iter",250,900),
  max_leaf_nodes=t.suggest_int("max_leaf_nodes",15,127),max_depth=t.suggest_int("max_depth",4,14),
  min_samples_leaf=t.suggest_int("min_samples_leaf",15,100),l2_regularization=t.suggest_float("l2_regularization",1e-3,10,log=True),
  max_bins=t.suggest_int("max_bins",64,255),early_stopping=True,validation_fraction=.08,n_iter_no_change=35,random_state=SEED)
 m=HistGradientBoostingClassifier(**p);m.fit(X.iloc[a],y[a],sample_weight=compute_sample_weight("balanced",y[a]))
 return balanced_accuracy_score(y[b],m.predict_proba(X.iloc[b]).argmax(1))
study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED));study.optimize(objective,n_trials=36)
best=study.best_params;print(study.best_value,best)
oof=np.zeros((len(X),3),np.float32);pred=np.zeros((len(T),3));fs=[];t0=time.time();skf=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)
for f,(a,b) in enumerate(skf.split(X,y),1):
 m=HistGradientBoostingClassifier(**best,early_stopping=True,validation_fraction=.08,n_iter_no_change=45,random_state=SEED+f)
 m.fit(X.iloc[a],y[a],sample_weight=compute_sample_weight("balanced",y[a]));oof[b]=m.predict_proba(X.iloc[b]);pred+=m.predict_proba(T)/FOLDS
 fs.append(float(balanced_accuracy_score(y[b],oof[b].argmax(1))));del m;gc.collect()
score=float(balanced_accuracy_score(y,oof.argmax(1)));od=pd.DataFrame({ID:tr[ID]});td=pd.DataFrame({ID:te[ID]})
for j,c in enumerate(C):od[c]=oof[:,j];td[c]=pred[:,j]
od.to_csv("oof_preds.csv",index=False);td.to_csv("test_preds.csv",index=False);pd.DataFrame({ID:te[ID],TARGET:C[pred.argmax(1)]}).to_csv("submission.csv",index=False)
Path("training_summary.json").write_text(json.dumps({"experiment":"optuna_hgb","tuning_best_score":study.best_value,"best_params":best,"fold_scores":fs,"oof_balanced_accuracy":score,"elapsed_minutes":(time.time()-t0)/60},indent=2));print(score)
'''

def make(slug,title,src,gpu):
 out=ROOT/"kaggle_kernels"/slug;out.mkdir(parents=True,exist_ok=True);compile("\n".join(x for x in src.splitlines() if not x.startswith("!")),slug,"exec")
 nb={"cells":[{"cell_type":"code","execution_count":None,"id":slug,"metadata":{},"outputs":[],"source":src.splitlines(keepends=True)}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
 (out/f"{slug}.ipynb").write_text(json.dumps(nb,indent=1))
 meta={"id":f"{OWNER}/{slug.replace('_','-')}","title":title,"code_file":f"{slug}.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":gpu,"enable_internet":True,"dataset_sources":[],"kernel_sources":[],"competition_sources":["playground-series-s6e7"]}
 if gpu:meta["machine_shape"]="NvidiaTeslaT4"
 (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
def main():
 make("health_risk_optuna_xgb_gpu","Health Risk Optuna XGB GPU",XGB,True);make("health_risk_optuna_hgb_cpu","Health Risk Optuna HGBC CPU",HGB,False)
if __name__=="__main__":main()
