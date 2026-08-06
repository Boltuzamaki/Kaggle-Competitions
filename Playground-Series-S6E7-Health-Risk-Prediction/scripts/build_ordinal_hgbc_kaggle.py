"""Build an honest ordinal TE-HGBC Kaggle experiment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle_kernels/ordinal_hgbc_cpu"

CODE = r'''
import glob, json, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
warnings.filterwarnings("ignore")

def find_data():
    for path in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        folder = os.path.dirname(path)
        if os.path.exists(os.path.join(folder, "test.csv")):
            cols = pd.read_csv(path, nrows=2).columns
            if "health_condition" in cols:
                return folder
    raise FileNotFoundError("competition data not found")

DATA=find_data(); train=pd.read_csv(f"{DATA}/train.csv"); test=pd.read_csv(f"{DATA}/test.csv")
ID="id"; TARGET="health_condition"; ORDER=["fit","at-risk","unhealthy"]
y=train[TARGET].map({c:i for i,c in enumerate(ORDER)}).to_numpy()
RAW=[c for c in train if c not in (ID,TARGET)]
CATS=["diet_type","stress_level","sleep_quality","physical_activity_level","smoking_alcohol","gender"]
X=train[RAW].copy(); Xt=test[RAW].copy()
for c in CATS:
    X[c]=X[c].astype("category")
    Xt[c]=Xt[c].astype("category").cat.set_categories(X[c].cat.categories)
Xs=train[RAW].astype(str).fillna("na"); Xst=test[RAW].astype(str).fillna("na")
PARAMS=dict(learning_rate=.0627,max_iter=300,max_leaf_nodes=33,min_samples_leaf=298,
            l2_regularization=.0289,max_bins=237,max_features=.82,early_stopping=True,
            categorical_features="from_dtype")
skf=StratifiedKFold(5,shuffle=True,random_state=2027)
oof=np.zeros((len(X),3)); pred=np.zeros((len(Xt),3)); fold_scores=[]

def binary_weights(z):
    count=np.bincount(z,minlength=2)
    return (len(z)/(2*count))[z]

for fold,(tr,va) in enumerate(skf.split(X,y),1):
    qv=[]; qt=[]
    for boundary in (1,2):
        z=(y>=boundary).astype(int)
        enc=TargetEncoder(cv=5,smooth="auto",shuffle=True,random_state=2027+fold+boundary)
        names=[f"te_{c}_ge{boundary}" for c in RAW]
        Atr=pd.concat([X.iloc[tr].reset_index(drop=True),
            pd.DataFrame(enc.fit_transform(Xs.iloc[tr],z[tr]),columns=names)],axis=1)
        Ava=pd.concat([X.iloc[va].reset_index(drop=True),
            pd.DataFrame(enc.transform(Xs.iloc[va]),columns=names)],axis=1)
        Ate=pd.concat([Xt.reset_index(drop=True),
            pd.DataFrame(enc.transform(Xst),columns=names)],axis=1)
        model=HistGradientBoostingClassifier(**PARAMS,random_state=2027+fold+boundary)
        model.fit(Atr,z[tr],sample_weight=binary_weights(z[tr]))
        qv.append(model.predict_proba(Ava)[:,1]); qt.append(model.predict_proba(Ate)[:,1])
    # Enforce monotonic cumulative probabilities P(Y>=2) <= P(Y>=1).
    q1v=qv[0]; q2v=np.minimum(qv[1],q1v)
    q1t=qt[0]; q2t=np.minimum(qt[1],q1t)
    pv=np.column_stack([1-q1v,q1v-q2v,q2v])
    pt=np.column_stack([1-q1t,q1t-q2t,q2t])
    oof[va]=pv; pred += pt/5
    s=balanced_accuracy_score(y[va],pv.argmax(1)); fold_scores.append(float(s))
    print("fold",fold,s)

score=float(balanced_accuracy_score(y,oof.argmax(1)))
print("OOF",score)
od=pd.DataFrame({ID:train[ID]}); td=pd.DataFrame({ID:test[ID]})
for i,c in enumerate(ORDER): od[c]=oof[:,i]; td[c]=pred[:,i]
od.to_csv("oof_preds.csv",index=False); td.to_csv("test_preds.csv",index=False)
pd.DataFrame({ID:test[ID],TARGET:[ORDER[i] for i in pred.argmax(1)]}).to_csv("submission.csv",index=False)
Path("training_summary.json").write_text(json.dumps({
    "experiment":"ordinal_te_hgbc","oof_balanced_accuracy":score,
    "fold_scores":fold_scores,"order":ORDER,"competition_data_only":True},indent=2))
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Ordinal TE-HGBC\n",
                "Models fit → at-risk → unhealthy as two cumulative binary boundaries.\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": CODE.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / "ordinal_hgbc_cpu.ipynb").write_text(json.dumps(notebook, indent=1))
    metadata = {
        "id": "boltuzamaki/health-risk-ordinal-hgbc-cpu",
        "title": "Health Risk Ordinal HGBC CPU",
        "code_file": "ordinal_hgbc_cpu.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["playground-series-s6e7"],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
