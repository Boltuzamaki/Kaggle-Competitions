"""Build a leakage-safe OOF disagreement/meta-model Kaggle experiment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle_kernels/disagreement_meta_cpu"

CODE = r'''
import glob, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
ID="id"; TARGET="health_condition"; C=["at-risk","fit","unhealthy"]

def one(pattern):
    found=glob.glob("/kaggle/input/**/"+pattern,recursive=True)
    if not found: raise FileNotFoundError(pattern)
    print(pattern,found[0]); return found[0]
train=pd.read_csv(one("train.csv")); test=pd.read_csv(one("test.csv"))
y=train[TARGET].map({c:i for i,c in enumerate(C)}).to_numpy()
specs=[
 ("f27","oof_ftt.csv","testpred_ftt.csv","health-risk-ftt-balanced-seed-2027"),
 ("f42","oof_ftt.csv","testpred_ftt.csv","health-risk-ftt-balanced-seed-4242"),
 ("r31","oof_preds.csv","test_preds.csv","health-risk-realmlp-seed-31415"),
 ("r77","oof_preds.csv","test_preds.csv","health-risk-realmlp-variant-7777"),
]
def locate(filename,hint):
    paths=glob.glob("/kaggle/input/**/"+filename,recursive=True)
    paths=[p for p in paths if hint in p]
    if len(paths)!=1: raise RuntimeError((filename,hint,paths))
    return paths[0]
oofs=[]; tests=[]
for name,of,tf,hint in specs:
    a=pd.read_csv(locate(of,hint)).set_index(ID).loc[train[ID],C].to_numpy()
    b=pd.read_csv(locate(tf,hint)).set_index(ID).loc[test[ID],C].to_numpy()
    oofs.append(a); tests.append(b)

def make_features(ps,raw):
    z=np.hstack(ps)
    for p in ps:
        s=np.sort(p,axis=1)
        z=np.column_stack([z,p.max(1),s[:,-1]-s[:,-2],-(p*np.log(p+1e-12)).sum(1)])
    # Disagreement and missingness are the routing signals; do not give the
    # meta-model enough raw capacity to simply relearn the whole task.
    votes=np.column_stack([p.argmax(1) for p in ps])
    z=np.column_stack([z,(votes!=votes[:,[0]]).sum(1),raw.isna().sum(1)])
    for col in raw:
        z=np.column_stack([z,raw[col].isna().astype(float).to_numpy()])
    return z
X=make_features(oofs,train.drop(columns=[ID,TARGET]))
Xt=make_features(tests,test.drop(columns=[ID]))
base=.6*oofs[0]+.4*oofs[2]; baset=.6*tests[0]+.4*tests[2]
skf=StratifiedKFold(5,shuffle=True,random_state=31415)
meta=np.zeros_like(base); metat=np.zeros_like(baset); folds=np.full(len(y),-1)
for fold,(tr,va) in enumerate(skf.split(X,y)):
    folds[va]=fold; count=np.bincount(y[tr],minlength=3); sw=(len(tr)/(3*count))[y[tr]]
    m=HistGradientBoostingClassifier(max_iter=250,learning_rate=.04,max_leaf_nodes=15,
        min_samples_leaf=300,l2_regularization=2.0,early_stopping=True,random_state=fold)
    m.fit(X[tr],y[tr],sample_weight=sw)
    meta[va]=m.predict_proba(X[va]); metat += m.predict_proba(Xt)/5

alphas=np.arange(0,0.525,.025)
rows=[]; nested=np.zeros_like(base); chosen=[]
for fold in range(5):
    fit=folds!=fold; val=~fit
    scores=[balanced_accuracy_score(y[fit],((1-a)*base[fit]+a*meta[fit]).argmax(1)) for a in alphas]
    a=float(alphas[int(np.argmax(scores))]); chosen.append(a)
    nested[val]=(1-a)*base[val]+a*meta[val]
    rows.append({"fold":fold,"alpha":a,"heldout":float(balanced_accuracy_score(y[val],nested[val].argmax(1))),
                 "base":float(balanced_accuracy_score(y[val],base[val].argmax(1)))})
alpha=float(np.median(chosen)); final=(1-alpha)*baset+alpha*metat
summary={"experiment":"disagreement_meta","base_oof":float(balanced_accuracy_score(y,base.argmax(1))),
 "meta_oof":float(balanced_accuracy_score(y,meta.argmax(1))),
 "nested_blend_oof":float(balanced_accuracy_score(y,nested.argmax(1))),
 "folds":rows,"median_alpha":alpha,"competition_data_only":True}
print(json.dumps(summary,indent=2));Path("training_summary.json").write_text(json.dumps(summary,indent=2))
pd.DataFrame({ID:test[ID],TARGET:[C[i] for i in final.argmax(1)]}).to_csv("submission.csv",index=False)
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Disagreement meta-model\n",
                "OOF probabilities, margins and missingness feed a low-capacity nested meta-model.\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": CODE.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / "disagreement_meta_cpu.ipynb").write_text(json.dumps(notebook, indent=1))
    metadata = {
        "id": "boltuzamaki/health-risk-disagreement-meta-cpu",
        "title": "Health Risk Disagreement Meta CPU",
        "code_file": "disagreement_meta_cpu.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [
            "boltuzamaki/health-risk-ftt-balanced-seed-2027-gpu",
            "boltuzamaki/health-risk-ftt-balanced-seed-4242-gpu",
            "boltuzamaki/health-risk-realmlp-seed-31415-gpu",
            "boltuzamaki/health-risk-realmlp-variant-7777-gpu",
        ],
        "competition_sources": ["playground-series-s6e7"],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
