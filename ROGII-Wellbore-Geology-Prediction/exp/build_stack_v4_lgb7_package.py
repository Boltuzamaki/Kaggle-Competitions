"""Train and package the complementary Stack V4 lgb7 leg.

The feature-builder module is mechanically extracted from the audited public
notebook's PF, beam, and feature-factory cells, then extended only with a hidden
test entry point.  Package contents are model code/weights/manifest, never
prediction CSVs.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/"kernels/stack_v2/rogii-general-stack-v4.ipynb"
OUT=ROOT/"exp/results/stack_v4_lgb7_package"
OUT.mkdir(parents=True,exist_ok=True)

nb=json.loads(NB.read_text())
code=["".join(c.get("source",[])) for c in nb["cells"] if c["cell_type"]=="code"]
# Code cells: tiny import, PF, beam, factory+training, model training.
factory=code[1]+"\n\n"+code[2]+"\n\n"+code[3].split("t0=time.time()")[0]
factory=factory.replace(
    'for r in ["/kaggle/input/rogii-wellbore-geology-prediction",',
    'for r in ([os.environ.get("ROGII_DATA_ROOT")] if os.environ.get("ROGII_DATA_ROOT") else []) + ["/kaggle/input/rogii-wellbore-geology-prediction",')
factory += r'''

def build_inference_features(data_root=None, split="test", n_jobs=1):
    """Build exact V4 legal features for hidden query wells."""
    global DATA, PIDX, PTREE, PSCL, PXA, PYA, PFA, PWID
    if data_root is not None:
        DATA=str(data_root)
        PIDX,PTREE,PSCL=build_plane_index()
        PXA=PIDX["x"].to_numpy(); PYA=PIDX["y"].to_numpy()
        PFA=PIDX[FORMS].to_numpy(np.float64)
        PWID={w:i for i,w in enumerate(PIDX["wid"])}
    paths=sorted(glob.glob(os.path.join(DATA,split,"*__horizontal_well.csv")))
    rows=Parallel(n_jobs=n_jobs)(delayed(build_well)(p,False,False) for p in paths)
    rows=[r for r in rows if r is not None]
    if not rows:
        raise RuntimeError(f"No inference wells found under {DATA}/{split}")
    return pd.concat(rows,ignore_index=True)

def predict_hidden(model_path, manifest_path, data_root=None, split="test"):
    """Fresh model inference; returns IDs and TVT, never reads predictions."""
    import json
    import lightgbm as _lgb
    manifest=json.load(open(manifest_path))
    f=build_inference_features(data_root,split,n_jobs=1)
    missing=[c for c in manifest["features"] if c not in f]
    if missing: raise ValueError(f"missing features: {missing}")
    model=_lgb.Booster(model_file=str(model_path))
    delta=model.predict(f[manifest["features"]])
    return pd.DataFrame({"id":f["id"],"tvt":f["last_known_tvt"]+delta})
'''
(OUT/"stack_v4_feature_builder.py").write_text(factory)

train=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
DROP={"well","id","target","last_known_tvt","supertype"}
features=[c for c in train.columns if c not in DROP]
X=train[features].to_numpy(np.float32); y=train.target.to_numpy(np.float32)
params=dict(objective="regression",n_estimators=1500,learning_rate=.02,
            num_leaves=127,min_child_samples=80,subsample=.8,subsample_freq=1,
            colsample_bytree=.7,reg_lambda=5.,reg_alpha=1.,verbose=-1,
            n_jobs=max(1,os.cpu_count() or 8),random_state=7)
model=lgb.LGBMRegressor(**params).fit(X,y)
model.booster_.save_model(str(OUT/"stack_v4_lgb7.txt"))

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={
 "schema":"rogii_stack_v4_lgb7_v1",
 "source_notebook":"kernels/stack_v2/rogii-general-stack-v4.ipynb",
 "target":"TVT - last visible TVT",
 "features":features,"feature_count":len(features),
 "training_rows":len(train),"training_wells":int(train.well.nunique()),
 "model_params":params,
 "legal_inputs":["horizontal trajectory","horizontal GR","paired typewell GR",
                 "visible TVT_input prefix","training-well spatial formation surfaces"],
 "prediction_artifacts_used":False,
 "files":{}
}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
manifest["files"]={
 "model_sha256":sha(OUT/"stack_v4_lgb7.txt"),
 "builder_sha256":sha(OUT/"stack_v4_feature_builder.py"),
}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))

# Import the packaged builder itself and run fresh placeholder inference.
os.environ["ROGII_DATA_ROOT"]=str(ROOT/"data")
spec=importlib.util.spec_from_file_location("stack_v4_feature_builder",
                                             OUT/"stack_v4_feature_builder.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
pred=mod.predict_hidden(OUT/"stack_v4_lgb7.txt",OUT/"manifest.json",
                        ROOT/"data","test")
sample=pd.read_csv(ROOT/"data/sample_submission.csv")
audit={
 "rows":len(pred),"sample_rows":len(sample),
 "unique_ids":int(pred.id.nunique()),
 "ids_exact_set":set(pred.id)==set(sample.id),
 "finite":bool(np.isfinite(pred.tvt).all()),
 "min":float(pred.tvt.min()),"max":float(pred.tvt.max()),
 "prefix_by_well":{},
}
for w,g in pred.groupby(pred.id.str.split("_").str[0]):
    h=pd.read_csv(ROOT/"data/test"/f"{w}__horizontal_well.csv")
    p=int(h.TVT_input.notna().sum())
    audit["prefix_by_well"][w]={"hidden_rows":len(h)-p,"pred_rows":len(g)}
if not (audit["ids_exact_set"] and audit["finite"] and audit["rows"]==audit["sample_rows"]):
    raise RuntimeError(f"inference audit failed: {audit}")
(OUT/"placeholder_audit.json").write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2),flush=True)
