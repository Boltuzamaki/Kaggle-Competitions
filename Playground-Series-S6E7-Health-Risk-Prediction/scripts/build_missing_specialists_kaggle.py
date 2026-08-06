"""Build CPU notebooks targeting the three decisive-feature missing patterns."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook(source: str, ident: str) -> dict:
    return {
        "cells": [{
            "cell_type": "code", "execution_count": None, "id": ident,
            "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


LOOKUP = r'''
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ID="id"; TARGET="health_condition"; SEED=2027; FOLDS=5
C=np.array(["at-risk","fit","unhealthy"]); M={c:i for i,c in enumerate(C)}
base=Path("/kaggle/input/competitions/playground-series-s6e7")
tr=pd.read_csv(base/"train.csv"); te=pd.read_csv(base/"test.csv"); y=tr[TARGET].map(M).to_numpy()

def states(d):
    z=pd.DataFrame(index=d.index)
    s=d.sleep_duration
    z["sleep_zone"]=pd.cut(s,[-np.inf,6,7,np.inf],right=False,labels=["lt6","6to7","ge7"]).astype("object").fillna("NA")
    for c in ["stress_level","physical_activity_level","sleep_quality","diet_type","gender","smoking_alcohol"]:
        z[c]=d[c].fillna("NA").astype(str)
    z["pattern"]=d[["sleep_duration","stress_level","physical_activity_level"]].isna().astype(int).astype(str).agg("".join,axis=1)
    z["core"]=z.sleep_zone+"|"+z.stress_level+"|"+z.physical_activity_level
    z["core_sleepq"]=z.core+"|"+z.sleep_quality
    z["fallback"]=z.sleep_zone+"|"+z.stress_level
    return z

A=states(tr); B=states(te)
keys=[
 ["core_sleepq"],["core"],["sleep_zone","stress_level"],
 ["sleep_zone","physical_activity_level","pattern"],
 ["stress_level","physical_activity_level","pattern"],["pattern"],
]

def lookup_fit_predict(a_train,y_train,a_query,alpha=8.0):
    prior=np.bincount(y_train,minlength=3).astype(float); prior/=prior.sum()
    out=np.tile(prior,(len(a_query),1)); unresolved=np.ones(len(a_query),bool)
    for cols in keys:
        tmp=a_train[cols].copy()
        tmp["_y"]=y_train
        counts=tmp.groupby(cols,dropna=False)["_y"].value_counts().unstack(fill_value=0)
        for k in range(3):
            if k not in counts: counts[k]=0
        counts=counts[[0,1,2]]
        probs=(counts+alpha)/(counts.sum(1).to_numpy()[:,None]+3*alpha)
        q=a_query[cols].merge(probs,left_on=cols,right_index=True,how="left")[[0,1,2]].to_numpy(float)
        ok=unresolved & np.isfinite(q).all(1)
        out[ok]=q[ok]; unresolved[ok]=False
    return out

oof=np.zeros((len(tr),3)); pred=np.zeros((len(te),3)); fs=[]; t0=time.time()
skf=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)
for f,(i,j) in enumerate(skf.split(A,y),1):
    oof[j]=lookup_fit_predict(A.iloc[i],y[i],A.iloc[j])
    pred+=lookup_fit_predict(A.iloc[i],y[i],B)/FOLDS
    s=balanced_accuracy_score(y[j],oof[j].argmax(1));fs.append(float(s));print(f,s)
overall=float(balanced_accuracy_score(y,oof.argmax(1)))
od=pd.DataFrame({ID:tr[ID]});td=pd.DataFrame({ID:te[ID]})
for k,c in enumerate(C):od[c]=oof[:,k];td[c]=pred[:,k]
od.to_csv("oof_preds.csv",index=False);td.to_csv("test_preds.csv",index=False)
pd.DataFrame({ID:te[ID],TARGET:C[pred.argmax(1)]}).to_csv("submission.csv",index=False)
Path("training_summary.json").write_text(json.dumps({"experiment":"conditional_rule_lookup","fold_scores":fs,"oof_balanced_accuracy":overall,"elapsed_minutes":(time.time()-t0)/60},indent=2))
print(overall)
'''

PATTERN = r'''
import gc,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
ID="id";TARGET="health_condition";SEED=2027;FOLDS=5
C=np.array(["at-risk","fit","unhealthy"]);M={c:i for i,c in enumerate(C)}
base=Path("/kaggle/input/competitions/playground-series-s6e7")
tr=pd.read_csv(base/"train.csv");te=pd.read_csv(base/"test.csv");y=tr[TARGET].map(M).to_numpy()

def prep(d):
 z=d.drop(columns=[ID,TARGET],errors="ignore").copy();s=z.sleep_duration
 z["sleep_lt6"]=np.where(s.isna(),np.nan,(s<6).astype(float));z["sleep_lt7"]=np.where(s.isna(),np.nan,(s<7).astype(float))
 z["pattern"]=z[["sleep_duration","stress_level","physical_activity_level"]].isna().astype(int).astype(str).agg("".join,axis=1)
 z["sleep_zone"]=pd.cut(s,[-np.inf,6,7,np.inf],right=False,labels=["lt6","6to7","ge7"]).astype("object")
 return z
X=prep(tr);T=prep(te);patterns=sorted(set(X.pattern)|set(T.pattern))
cats=list(X.select_dtypes(include=["object","category"]).columns);nums=[c for c in X if c not in cats]
e=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)
e.fit(pd.concat([X[cats],T[cats]]).fillna("NA"));X[cats]=e.transform(X[cats].fillna("NA"));T[cats]=e.transform(T[cats].fillna("NA"))
med=X[nums].median();X[nums]=X[nums].fillna(med);T[nums]=T[nums].fillna(med);X=X.astype("float32");T=T.astype("float32")
pattern_col=X.columns.get_loc("pattern")
oof=np.zeros((len(X),3));pred=np.zeros((len(T),3));fs=[];t0=time.time()
skf=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)
for f,(a,b) in enumerate(skf.split(X,y),1):
 for p in np.unique(X.iloc[:,pattern_col]):
  ia=a[X.iloc[a,pattern_col].to_numpy()==p];ib=b[X.iloc[b,pattern_col].to_numpy()==p];it=np.where(T.iloc[:,pattern_col].to_numpy()==p)[0]
  if not len(ib):continue
  m=HistGradientBoostingClassifier(max_iter=450,learning_rate=.08,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=3,early_stopping=True,n_iter_no_change=30,random_state=SEED+f)
  m.fit(X.iloc[ia],y[ia],sample_weight=compute_sample_weight("balanced",y[ia]))
  oof[ib]=m.predict_proba(X.iloc[ib])
  if len(it):pred[it]+=m.predict_proba(T.iloc[it])/FOLDS
  del m;gc.collect()
 s=balanced_accuracy_score(y[b],oof[b].argmax(1));fs.append(float(s));print(f,s)
overall=float(balanced_accuracy_score(y,oof.argmax(1)))
od=pd.DataFrame({ID:tr[ID]});td=pd.DataFrame({ID:te[ID]})
for k,c in enumerate(C):od[c]=oof[:,k];td[c]=pred[:,k]
od.to_csv("oof_preds.csv",index=False);td.to_csv("test_preds.csv",index=False)
pd.DataFrame({ID:te[ID],TARGET:C[pred.argmax(1)]}).to_csv("submission.csv",index=False)
Path("training_summary.json").write_text(json.dumps({"experiment":"pattern_hgb","fold_scores":fs,"oof_balanced_accuracy":overall,"elapsed_minutes":(time.time()-t0)/60},indent=2));print(overall)
'''


def main() -> None:
    specs = [
        ("rule_lookup_cpu", "Health Risk Conditional Rule Lookup CPU", LOOKUP, "lookup"),
        ("pattern_hgb_cpu", "Health Risk Pattern HGBC CPU", PATTERN, "pattern"),
    ]
    for slug,title,src,ident in specs:
        compile(src,slug,"exec")
        out=ROOT/"kaggle_kernels"/slug;out.mkdir(parents=True,exist_ok=True)
        (out/f"{slug}.ipynb").write_text(json.dumps(notebook(src,ident),indent=1))
        meta={"id":f"boltuzamaki/health-{slug.replace('_','-')}","title":title,"code_file":f"{slug}.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":False,"enable_internet":False,"dataset_sources":[],"kernel_sources":[],"competition_sources":["playground-series-s6e7"]}
        (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
        print(out)

if __name__=="__main__":main()
