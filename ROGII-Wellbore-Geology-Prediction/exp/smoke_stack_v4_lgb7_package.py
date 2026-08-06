"""Fresh placeholder inference and integrity audit for packaged Stack V4 lgb7."""
from pathlib import Path
import hashlib, importlib.util, json, os, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/"exp/results/stack_v4_lgb7_package"
os.environ["ROGII_DATA_ROOT"]=str(ROOT/"data")
spec=importlib.util.spec_from_file_location("stack_v4_feature_builder",
                                             PKG/"stack_v4_feature_builder.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
t0=time.time()
pred=mod.predict_hidden(PKG/"stack_v4_lgb7.txt",PKG/"manifest.json",ROOT/"data","test")
sample=pd.read_csv(ROOT/"data/sample_submission.csv")
manifest=json.loads((PKG/"manifest.json").read_text())
audit={"runtime_seconds":time.time()-t0,"rows":len(pred),"sample_rows":len(sample),
       "unique_ids":int(pred.id.nunique()),"ids_exact_set":set(pred.id)==set(sample.id),
       "finite":bool(np.isfinite(pred.tvt).all()),"feature_count":manifest["feature_count"],
       "min":float(pred.tvt.min()),"max":float(pred.tvt.max()),"prefix_by_well":{}}
for w,g in pred.groupby(pred.id.str.split("_").str[0]):
    h=pd.read_csv(ROOT/"data/test"/f"{w}__horizontal_well.csv")
    p=int(h.TVT_input.notna().sum())
    audit["prefix_by_well"][w]={"hidden_rows":len(h)-p,"pred_rows":len(g)}
if not(audit["ids_exact_set"] and audit["finite"] and len(pred)==len(sample)):
    raise RuntimeError(audit)
manifest["files"]["builder_sha256"]=hashlib.sha256(
    (PKG/"stack_v4_feature_builder.py").read_bytes()).hexdigest()
(PKG/"manifest.json").write_text(json.dumps(manifest,indent=2))
(PKG/"placeholder_audit.json").write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
