"""Fresh 14,151-row smoke test for packaged multiscale heel posterior."""
from pathlib import Path
import importlib.util,json,numpy as np,pandas as pd,lightgbm as lgb
R=Path(__file__).resolve().parents[1];V=R/"exp/results/stack_v4_lgb7_package";P=R/"exp/kaggle_heel_gr_dataset"
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v=load("v4_builder",V/"stack_v4_feature_builder.py");h=load("heel",P/"heel_gr_datum.py")
F=v.build_inference_features(R/"data","test",n_jobs=1);man=json.loads((V/"manifest.json").read_text())
delta=lgb.Booster(model_file=str(V/"stack_v4_lgb7.txt")).predict(F[man["features"]])
frame=pd.DataFrame({"id":F.id.astype(str),"last_known_tvt":F.last_known_tvt,
                    "v4_lgb7":delta})
out=h.correct_v4_delta_multiscale(R/"data",frame);sample=pd.read_csv(R/"data/sample_submission.csv",dtype={"id":str})
audit={"rows":len(out),"sample_rows":len(sample),"ids_exact_set":set(out.id)==set(sample.id),
 "finite":bool(np.isfinite(out.iloc[:,1:]).all().all()),"shift_min":float(out.gr_shift_multiscale.min()),
 "shift_max":float(out.gr_shift_multiscale.max()),"wells":int(out.id.str.split("_",n=1).str[0].nunique())}
if not(audit["rows"]==audit["sample_rows"] and audit["ids_exact_set"] and audit["finite"]):raise RuntimeError(audit)
(P/"multiscale_placeholder_audit.json").write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
