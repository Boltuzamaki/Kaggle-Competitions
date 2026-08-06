"""Build AutoGluon and YDF CPU experiments."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

AG=r'''
!pip install -q autogluon.tabular
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score
from autogluon.tabular import TabularPredictor
ID="id";TARGET="health_condition";C=["at-risk","fit","unhealthy"]
base=Path("/kaggle/input/competitions/playground-series-s6e7");tr=pd.read_csv(base/"train.csv");te=pd.read_csv(base/"test.csv");t0=time.time()
def fe(d):
 z=d.drop(columns=[ID],errors="ignore").copy();s=z.sleep_duration
 z["sleep_lt6"]=np.where(s.isna(),np.nan,(s<6).astype(float));z["sleep_lt7"]=np.where(s.isna(),np.nan,(s<7).astype(float))
 z["key_missing_count"]=z[["sleep_duration","stress_level","physical_activity_level"]].isna().sum(1)
 z["rule_unhealthy"]=((s<6)&z.stress_level.eq("high")).astype(float);z["rule_fit"]=((s>=7)&z.stress_level.eq("low")&z.physical_activity_level.eq("active")).astype(float)
 return z
A=fe(tr);B=fe(te)
predictor=TabularPredictor(label=TARGET,problem_type="multiclass",eval_metric="balanced_accuracy",path="/kaggle/working/ag_model",verbosity=2)
# OOF and test inference both require the fitted fold children to remain on
# disk. Discarding them saves space but makes predict_proba() fail afterward.
predictor.fit(A,presets="medium_quality",time_limit=10800,num_bag_folds=5,num_stack_levels=1,dynamic_stacking=False,save_bag_folds=True)
o=predictor.predict_proba_oof();p=predictor.predict_proba(B)
o=o.reindex(columns=C);p=p.reindex(columns=C);score=float(balanced_accuracy_score(tr[TARGET],o.idxmax(1)))
od=pd.DataFrame({ID:tr[ID]});td=pd.DataFrame({ID:te[ID]})
for c in C:od[c]=o[c].to_numpy();td[c]=p[c].to_numpy()
od.to_csv("oof_preds.csv",index=False);td.to_csv("test_preds.csv",index=False);pd.DataFrame({ID:te[ID],TARGET:p.idxmax(1)}).to_csv("submission.csv",index=False)
Path("training_summary.json").write_text(json.dumps({"experiment":"autogluon_medium_bag5","oof_balanced_accuracy":score,"elapsed_minutes":(time.time()-t0)/60,"leaderboard":predictor.leaderboard(silent=True).to_dict("records")},indent=2,default=str));print(score)
'''

YDF=r'''
!pip install -q ydf
import gc,json,time
from pathlib import Path
import numpy as np,pandas as pd,ydf
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
ID="id";TARGET="health_condition";C=np.array(["at-risk","fit","unhealthy"]);SEED=2027;FOLDS=5
base=Path("/kaggle/input/competitions/playground-series-s6e7");tr=pd.read_csv(base/"train.csv");te=pd.read_csv(base/"test.csv");y=tr[TARGET].to_numpy();t0=time.time()
def fe(d):
 z=d.drop(columns=[ID],errors="ignore").copy();s=z.sleep_duration
 z["sleep_lt6"]=np.where(s.isna(),np.nan,(s<6).astype(float));z["sleep_lt7"]=np.where(s.isna(),np.nan,(s<7).astype(float))
 z["key_missing_count"]=z[["sleep_duration","stress_level","physical_activity_level"]].isna().sum(1)
 z["rule_unhealthy"]=((s<6)&z.stress_level.eq("high")).astype(float);z["rule_fit"]=((s>=7)&z.stress_level.eq("low")&z.physical_activity_level.eq("active")).astype(float)
 return z
A=fe(tr);B=fe(te);oof=np.zeros((len(A),3),np.float32);pred=np.zeros((len(B),3));fs=[]
weights={c:len(y)/(3*(y==c).sum()) for c in C};skf=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)
for f,(a,b) in enumerate(skf.split(A,y),1):
 learner=ydf.GradientBoostedTreesLearner(label=TARGET,task=ydf.Task.CLASSIFICATION,num_trees=1800,max_depth=8,shrinkage=.04,subsample=.8,min_examples=30,l2_regularization=2.0,class_weights=weights,random_seed=SEED+f)
 model=learner.train(A.iloc[a]);oof[b]=model.predict(A.iloc[b]);pred+=model.predict(B)/FOLDS
 s=balanced_accuracy_score(y[b],C[oof[b].argmax(1)]);fs.append(float(s));print(f,s);del model;gc.collect()
score=float(balanced_accuracy_score(y,C[oof.argmax(1)]));od=pd.DataFrame({ID:tr[ID]});td=pd.DataFrame({ID:te[ID]})
for j,c in enumerate(C):od[c]=oof[:,j];td[c]=pred[:,j]
od.to_csv("oof_preds.csv",index=False);td.to_csv("test_preds.csv",index=False);pd.DataFrame({ID:te[ID],TARGET:C[pred.argmax(1)]}).to_csv("submission.csv",index=False)
Path("training_summary.json").write_text(json.dumps({"experiment":"ydf_gbt","fold_scores":fs,"oof_balanced_accuracy":score,"elapsed_minutes":(time.time()-t0)/60},indent=2));print(score)
'''

def make(slug,title,src,ident,internet=True):
 out=ROOT/"kaggle_kernels"/slug;out.mkdir(parents=True,exist_ok=True)
 compile("\n".join(x for x in src.splitlines() if not x.startswith("!")),slug,"exec")
 nb={"cells":[{"cell_type":"code","execution_count":None,"id":ident,"metadata":{},"outputs":[],"source":src.splitlines(keepends=True)}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
 (out/f"{slug}.ipynb").write_text(json.dumps(nb,indent=1))
 meta={"id":f"boltuzamaki/{slug.replace('_','-')}","title":title,"code_file":f"{slug}.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":False,"enable_internet":internet,"dataset_sources":[],"kernel_sources":[],"competition_sources":["playground-series-s6e7"]}
 (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
def main():
 make("health_risk_autogluon_cpu","Health Risk AutoGluon CPU",AG,"ag-main")
 make("health_risk_ydf_cpu","Health Risk YDF CPU",YDF,"ydf-main")
if __name__=="__main__":main()
