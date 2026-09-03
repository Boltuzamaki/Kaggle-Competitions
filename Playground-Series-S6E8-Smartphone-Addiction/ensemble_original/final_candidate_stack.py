#!/usr/bin/env python3
"""Final guarded stack centered on the 0.96986 deployment family plus lookup V2."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"ensemble_original/reports"; TARGET="addicted_label"; SEED=20260803
spec=importlib.util.spec_from_file_location("audit",ROOT/"ensemble_original/audit_and_blend.py"); ab=importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
tr=pd.read_csv(ROOT/"train.csv"); te=pd.read_csv(ROOT/"test.csv"); y=tr[TARGET].to_numpy(int)
core_names=["xgb_te_5fold","xgb_te_4fold","catboost_unique","histgb_5fold","neural_5fold","raw_xgb_bag","lgb_te_5fold","tabm_rank1_v2"]
names=[]; oofs=[]; tests=[]
for name in core_names:
 o,t,_=ab.load_pair(name,ab.MODELS[name],tr,te); names.append(name); oofs.append(o); tests.append(t)

def add(name,op,tp,oc="pred",tc="pred"):
 o=pd.read_csv(ROOT/op).set_index("id").reindex(tr.id); t=pd.read_csv(ROOT/tp).set_index("id").reindex(te.id)
 assert len(o)==len(tr) and len(t)==len(te) and np.isfinite(o[oc]).all() and np.isfinite(t[tc]).all()
 names.append(name); oofs.append(o[oc].to_numpy(float)); tests.append(t[tc].to_numpy(float))

# The selected pre-V2 family; pair-lattice/depth variants were measured negative conditionally.
add("xgb_hpo","xgb_nested_hpo/output/oof_nested_hpo_xgb.csv","xgb_nested_hpo/output/test_nested_hpo_xgb.csv")
add("lookup_v1","gpu_lookup_original/output/oof_lookup_transformer.csv","gpu_lookup_original/output/test_lookup_transformer.csv","pred",TARGET)
add("nested_cat","gpu_catboost_te/output/oof_nested_te_catboost.csv","gpu_catboost_te/output/test_nested_te_catboost.csv","pred",TARGET)
add("dual_cat","gpu_catboost_dual/output/oof_dual_view_catboost.csv","gpu_catboost_dual/output/test_dual_view_catboost.csv","pred",TARGET)
add("driver_lgb","lgb_driver_reconstruction/output/oof_driver_reconstruction_lgb.csv","lgb_driver_reconstruction/output/test_driver_reconstruction_lgb.csv")
base_count=len(names)

v2o=pd.read_csv(ROOT/"artifacts/local_lookup_v2/oof_lookup_v2.csv").set_index("id").reindex(tr.id)
v2t=pd.read_csv(ROOT/"artifacts/local_lookup_v2/test_lookup_v2.csv").set_index("id").reindex(te.id)
raw_o=v2o.pred.to_numpy(float); raw_t=v2t[TARGET].to_numpy(float); fold_rank=np.zeros(len(v2o))
for f in sorted(v2o.fold.unique()):
 m=v2o.fold.to_numpy()==f; fold_rank[m]=pd.Series(raw_o[m]).rank(pct=True).to_numpy()
test_rank=pd.Series(raw_t).rank(pct=True).to_numpy()

X=np.column_stack(oofs); XT=np.column_stack(tests); cv=StratifiedKFold(5,shuffle=True,random_state=SEED+91)
def crossfit(x):
 p=np.zeros(len(y))
 for fit,val in cv.split(x,y):
  w=ab.fit_weights(x[fit],y[fit],"logit"); p[val]=ab.combine(x[val],w,"logit")
 return roc_auc_score(y,p)

configs={"base":(X,XT,names),"plus_v2_raw":(np.column_stack([X,raw_o]),np.column_stack([XT,raw_t]),names+["lookup_v2_raw"]),"plus_v2_foldrank":(np.column_stack([X,fold_rank]),np.column_stack([XT,test_rank]),names+["lookup_v2_foldrank"])}
rows=[]
for name,(x,xt,ns) in configs.items(): rows.append({"configuration":name,"crossfit_auc":crossfit(x)})
report=pd.DataFrame(rows).sort_values("crossfit_auc",ascending=False); report.to_csv(OUT/"final_candidate_crossfit.csv",index=False); print(report)
best=report.iloc[0].configuration; x,xt,ns=configs[best]; w=ab.fit_weights(x,y,"logit"); pred=ab.combine(xt,w,"logit")
sub=pd.DataFrame({"id":te.id,TARGET:pred}); assert sub.id.equals(te.id) and np.isfinite(pred).all() and pd.Series(pred).between(0,1).all(); sub.to_csv(OUT/"final_candidate_submission.csv",index=False)
manifest={"provenance":"independently trained official-data-only models; no public predictions","configuration":best,"crossfit":dict(zip(report.configuration,map(float,report.crossfit_auc))),"weights":dict(zip(ns,map(float,w))),"rows":len(sub),"test_id_sha256":hashlib.sha256(np.asarray(te.id,dtype=np.int64).tobytes()).hexdigest()[:16],"submitted":False}
(OUT/"final_candidate_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps(manifest,indent=2))
