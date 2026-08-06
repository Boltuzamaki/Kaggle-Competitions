"""Build CPU fresh-inference Harshini + Pilkwang + Stack V4 kernel."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"exp/results/harshini_pilkwang_kernel_source"
OUT=ROOT/"kernels/harshini_pilkwang_v4"
OUT.mkdir(parents=True,exist_ok=True)
nb=json.loads((SRC/"rogii-harshini-pilkwang-fresh-inference.ipynb").read_text())
s="".join(nb["cells"][8]["source"])
s=s.replace("@njit(cache=True, nogil=True)","@njit(cache=False, nogil=True)")

# Preserve Harshini's legal physics/feature definitions, remove expensive PF
# training and OOF model fitting, and load our full-data model weights.
prefix=s[:s.index("_=_pf(")]
defs=s[s.index("SCAN_F="):s.index("def build_rows():")]
infer=s[s.index("# ---- inference ----"):]
trust_pos=infer.index("TRUST=")
infer="# ---- inference ----\n"+r'''
from pathlib import Path as _Path
import json as _json
import lightgbm as _lgb
import xgboost as _xgb
_har_candidates=[
 _Path("/kaggle/input/datasets/boltuzamaki/rogii-harshini-fast-models"),
 _Path("/kaggle/input/rogii-harshini-fast-models")]
_har_pkg=next((p for p in _har_candidates if p.exists()),None)
if _har_pkg is None: raise RuntimeError("Harshini fast model package missing")
_har_manifest=_json.loads((_har_pkg/"manifest.json").read_text())
if set(FEATS_R) != set(_har_manifest["features"]) or len(FEATS_R)!=92:
    raise RuntimeError("Harshini 92-feature schema differs from trained manifest")
_HLGB=_lgb.Booster(model_file=str(_har_pkg/"harshini_fast_lgb.txt"))
_HXGB=_xgb.XGBRegressor()
_HXGB.load_model(_har_pkg/"harshini_fast_xgb.json")
_HXGB.set_params(device="cpu",n_jobs=-1)
''' + infer[trust_pos:]
old='''    legs=[blend_d]+[FULL[k].predict(F) for k in base]
    delta=ridge.predict(np.c_[tuple(legs)])
    tvt=apply_proj(flat+warmup(MD-md0)*delta,Z,s,anchor)
    tvt=np.where(np.isfinite(tvt),tvt,tvt0)
    for j,i in enumerate(range(ps,len(h))): rows.append((f"{wid}_{i}",float(tvt[j])))
sub=pd.DataFrame(rows,columns=["id","tvt"])
ss=pd.read_csv(os.path.join(DATA,"sample_submission.csv"))
sub=ss[["id"]].merge(sub,on="id",how="left"); sub["tvt"]=sub["tvt"].fillna(sub["tvt"].median())
assert sub.tvt.notna().all() and np.isfinite(sub.tvt).all()
sub.to_csv("submission.csv",index=False)
print(f"\\n[CELL 21] submission.csv {sub.shape} tvt [{sub.tvt.min():.1f}, {sub.tvt.max():.1f}]")'''
new='''    _wu=warmup(MD-md0)
    _plgb=_wu*_HLGB.predict(F)
    _pxgb=_wu*_HXGB.predict(F)
    _pphy=_wu*blend_d
    for j,i in enumerate(range(ps,len(h))):
        rows.append((f"{wid}_{i}",float(tvt0),float(_pphy[j]),
                     float(_plgb[j]),float(_pxgb[j])))
har_components=pd.DataFrame(rows,columns=[
    "id","last_known_tvt","har_physics","har_lgb","har_xgb"])
if len(har_components)==0 or not np.isfinite(
        har_components.iloc[:,1:].to_numpy()).all():
    raise RuntimeError("Harshini fresh component inference failed")
print("Harshini fresh components",har_components.shape)'''
if old not in infer: raise RuntimeError("Harshini inference replacement point missing")
infer=infer.replace(old,new)
infer=infer.replace(
    'F=pd.DataFrame(row)[FEATS_R].values.astype(np.float32)',
    'F=pd.DataFrame(row)[_har_manifest["features"]].values.astype(np.float32)')
nb["cells"][8]["source"]=prefix+"\n"+defs+"\n"+infer

# Replace old two-leg integration with strict five-leg delta-space integration.
integration=r'''
"""Strict fresh five-leg inference using deployable restricted OOF weights."""
from pathlib import Path as _Path
import importlib.util as _ilu
import json as _json
import numpy as _np
import pandas as _pd
import lightgbm as _lgb

_data=_Path(DATA)
_sample=_pd.read_csv(_data/"sample_submission.csv",dtype={"id":str})
if not _sample.id.is_unique: raise RuntimeError("sample IDs are not unique")

# Pilkwang model-weight inference.
_pil_candidates=[
 _Path("/kaggle/input/datasets/pilkwang/rogii-model-package"),
 _Path("/kaggle/input/rogii-model-package")]
_pil_pkg=next((p for p in _pil_candidates if p.exists()),None)
if _pil_pkg is None: raise RuntimeError("Pilkwang package missing")
_pil_frame,_pil_delta,_pil_models=fresh_pilkwang_delta(
    _data,_pil_pkg,sample=_sample)
if not _pil_frame.id.astype(str).equals(_sample.id):
    raise RuntimeError("Pilkwang order mismatch")

# Stack V4 fresh PF/beam/NCC/spatial features and lgb7 model weights.
_v4_candidates=[
 _Path("/kaggle/input/datasets/boltuzamaki/rogii-stack-v4-lgb7-package"),
 _Path("/kaggle/input/rogii-stack-v4-lgb7-package")]
_v4_pkg=next((p for p in _v4_candidates if p.exists()),None)
if _v4_pkg is None: raise RuntimeError("Stack V4 package missing")
_spec=_ilu.spec_from_file_location("stack_v4_feature_builder",
                                   _v4_pkg/"stack_v4_feature_builder.py")
_v4mod=_ilu.module_from_spec(_spec); _spec.loader.exec_module(_v4mod)
_v4manifest=_json.loads((_v4_pkg/"manifest.json").read_text())
_v4frame=_v4mod.build_inference_features(_data,"test",n_jobs=1)
if _v4manifest["features"] != [c for c in _v4manifest["features"]
                              if c in _v4frame.columns]:
    raise RuntimeError("Stack V4 feature manifest mismatch")
_v4model=_lgb.Booster(model_file=str(_v4_pkg/"stack_v4_lgb7.txt"))
_v4raw=_pd.DataFrame({"id":_v4frame.id.astype(str),
                      "v4_lgb7":_v4model.predict(
                          _v4frame[_v4manifest["features"]])})

# Identity alignment is always sample-left, never positional except after proof.
_all=_sample[["id"]].merge(har_components.assign(
    id=har_components.id.astype(str)),on="id",how="left",validate="one_to_one")
_all=_all.merge(_v4raw,on="id",how="left",validate="one_to_one")
if len(_all)!=len(_sample) or _all.isna().any().any():
    raise RuntimeError("component ID coverage failure")

_W={"har_physics":0.3985925949,"har_lgb":0.2873064512,
    "har_xgb":0.0101013985,"pil_blend":0.2752512537,
    "v4_lgb7":0.2479481257}
_final_delta=(_W["har_physics"]*_all.har_physics.to_numpy()+
              _W["har_lgb"]*_all.har_lgb.to_numpy()+
              _W["har_xgb"]*_all.har_xgb.to_numpy()+
              _W["pil_blend"]*_pil_delta+
              _W["v4_lgb7"]*_all.v4_lgb7.to_numpy())
sub=_sample[["id"]].copy()
sub["tvt"]=_all.last_known_tvt.to_numpy()+_final_delta
if not sub.id.equals(_sample.id): raise RuntimeError("final order mismatch")
if not _np.isfinite(sub.tvt).all(): raise RuntimeError("nonfinite final TVT")
sub.to_csv("/kaggle/working/submission.csv",index=False)
_audit={
 "rows":len(sub),"unique_ids":int(sub.id.nunique()),
 "sample_order_exact":bool(sub.id.equals(_sample.id)),
 "finite":bool(_np.isfinite(sub.tvt).all()),
 "weights":_W,"weight_sum":float(sum(_W.values())),
 "weight_sum_above_one_intentional":True,
 "weight_source":"restricted honest nested GroupKFold CV 8.384425; full OOF positive Ridge alpha=100",
 "fresh_inference":{"harshini":True,"pilkwang":True,"stack_v4_lgb7":True},
 "prediction_csv_inputs_read":False,"hardcoded_test_ids":False,
 "pilkwang_models":_pil_models,
 "tvt_min":float(sub.tvt.min()),"tvt_max":float(sub.tvt.max())}
_Path("/kaggle/working/inference_audit.json").write_text(
    _json.dumps(_audit,indent=2))
print(_json.dumps(_audit,indent=2))
'''
nb["cells"][10]["source"]=integration
nb["metadata"].setdefault("kaggle",{})["isGpuEnabled"]=False

code_name="rogii-harshini-pilkwang-fresh-inference.ipynb"
(OUT/code_name).write_text(json.dumps(nb))
meta=json.loads((SRC/"kernel-metadata.json").read_text())
meta["code_file"]=code_name
meta["enable_gpu"]=False
meta["dataset_sources"]=[
 "pilkwang/rogii-model-package",
 "boltuzamaki/rogii-harshini-fast-models",
 "boltuzamaki/rogii-stack-v4-lgb7-package"]
(OUT/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
print("built",OUT)
