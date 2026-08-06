"""Build a competition-data-only five-fold TabNet Kaggle experiment."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"kaggle_kernels"/"rule_tabnet_gpu"
SRC=r'''
!pip install -q pytorch-tabnet
import gc,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
ID="id";TARGET="health_condition";SEED=2027;FOLDS=5
C=np.array(["at-risk","fit","unhealthy"]);M={c:i for i,c in enumerate(C)}
base=Path("/kaggle/input/competitions/playground-series-s6e7")
tr=pd.read_csv(base/"train.csv");te=pd.read_csv(base/"test.csv");y=tr[TARGET].map(M).to_numpy()
def fe(d):
 z=d.drop(columns=[ID,TARGET],errors="ignore").copy();s=z.sleep_duration
 z["sleep_lt6"]=np.where(s.isna(),np.nan,(s<6).astype(float));z["sleep_lt7"]=np.where(s.isna(),np.nan,(s<7).astype(float))
 z["missing_key_count"]=z[["sleep_duration","stress_level","physical_activity_level"]].isna().sum(1)
 z["rule_unhealthy"]=((s<6)&z.stress_level.eq("high")).astype(float)
 z["rule_fit"]=((s>=7)&z.stress_level.eq("low")&z.physical_activity_level.eq("active")).astype(float)
 return z
X=fe(tr);T=fe(te);cats=list(X.select_dtypes("object").columns);nums=[c for c in X if c not in cats]
e=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1);e.fit(pd.concat([X[cats],T[cats]]).fillna("NA"))
X[cats]=e.transform(X[cats].fillna("NA"))+1;T[cats]=e.transform(T[cats].fillna("NA"))+1
med=X[nums].median();X[nums]=X[nums].fillna(med);T[nums]=T[nums].fillna(med)
XA=X.to_numpy(np.float32);TA=T.to_numpy(np.float32)
cat_idx=[X.columns.get_loc(c) for c in cats];cat_dims=[int(max(X[c].max(),T[c].max()))+1 for c in cats]
oof=np.zeros((len(X),3),np.float32);pred=np.zeros((len(T),3));fs=[];t0=time.time()
skf=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)
class_weights={k:len(y)/(3*(y==k).sum()) for k in range(3)}
for f,(a,b) in enumerate(skf.split(XA,y),1):
 model=TabNetClassifier(n_d=32,n_a=32,n_steps=5,gamma=1.4,lambda_sparse=1e-5,cat_idxs=cat_idx,cat_dims=cat_dims,cat_emb_dim=4,optimizer_fn=torch.optim.AdamW,optimizer_params={"lr":2e-2,"weight_decay":1e-5},scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size":10,"gamma":.5},mask_type="entmax",seed=SEED+f,device_name="cuda",verbose=10)
 model.fit(XA[a],y[a],eval_set=[(XA[b],y[b])],eval_name=["val"],eval_metric=["balanced_accuracy"],max_epochs=60,patience=12,batch_size=8192,virtual_batch_size=1024,num_workers=2,weights=class_weights,drop_last=False)
 oof[b]=model.predict_proba(XA[b]);pred+=model.predict_proba(TA)/FOLDS
 s=balanced_accuracy_score(y[b],oof[b].argmax(1));fs.append(float(s));print(f,s)
 del model;gc.collect();torch.cuda.empty_cache()
overall=float(balanced_accuracy_score(y,oof.argmax(1)))
od=pd.DataFrame({ID:tr[ID]});td=pd.DataFrame({ID:te[ID]})
for j,c in enumerate(C):od[c]=oof[:,j];td[c]=pred[:,j]
od.to_csv("oof_preds.csv",index=False);td.to_csv("test_preds.csv",index=False);pd.DataFrame({ID:te[ID],TARGET:C[pred.argmax(1)]}).to_csv("submission.csv",index=False)
Path("training_summary.json").write_text(json.dumps({"experiment":"tabnet","fold_scores":fs,"oof_balanced_accuracy":overall,"elapsed_minutes":(time.time()-t0)/60},indent=2));print(overall)
'''
def main():
 OUT.mkdir(parents=True,exist_ok=True);compile("\n".join(x for x in SRC.splitlines() if not x.startswith("!")),"tabnet","exec")
 nb={"cells":[{"cell_type":"code","execution_count":None,"id":"tabnet-main","metadata":{},"outputs":[],"source":SRC.splitlines(keepends=True)}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
 (OUT/"rule_tabnet_gpu.ipynb").write_text(json.dumps(nb,indent=1))
 meta={"id":"boltuzamaki/health-risk-rule-tabnet-gpu","title":"Health Risk Rule TabNet GPU","code_file":"rule_tabnet_gpu.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":True,"dataset_sources":[],"kernel_sources":[],"competition_sources":["playground-series-s6e7"],"machine_shape":"NvidiaTeslaT4"}
 (OUT/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__=="__main__":main()
