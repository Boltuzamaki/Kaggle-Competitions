"""Add fresh heel-GR datum correction to the fast five-leg Kaggle kernel."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"kernels/harshini_pilkwang_v4_fast"
OUT = ROOT/"kernels/harshini_pilkwang_v4_heel"
OUT.mkdir(parents=True, exist_ok=True)
nb = json.loads((SRC/"rogii-harshini-pilkwang-v4-fast.ipynb").read_text())
s = "".join(nb["cells"][10]["source"])
s = s.replace(
'''_v4raw=_pd.DataFrame({"id":_v4frame.id.astype(str),
                      "v4_lgb7":_v4model.predict(
                          _v4frame[_v4manifest["features"]])})''',
'''_v4raw=_pd.DataFrame({"id":_v4frame.id.astype(str),
                      "last_known_tvt":_v4frame.last_known_tvt.to_numpy(float),
                      "v4_lgb7":_v4model.predict(
                          _v4frame[_v4manifest["features"]])})

# Heel-calibrated GR datum: code/config only, recomputed from legal raw inputs.
_heel_candidates=[
 _Path("/kaggle/input/datasets/boltuzamaki/rogii-heel-gr-datum-code"),
 _Path("/kaggle/input/rogii-heel-gr-datum-code")]
_heel_pkg=next((p for p in _heel_candidates if p.exists()),None)
if _heel_pkg is None: raise RuntimeError("Heel GR datum package missing")
_hspec=_ilu.spec_from_file_location("heel_gr_datum",_heel_pkg/"heel_gr_datum.py")
_heelmod=_ilu.module_from_spec(_hspec); _hspec.loader.exec_module(_heelmod)
_heelmanifest=_json.loads((_heel_pkg/"manifest.json").read_text())
_v4corr=_heelmod.correct_v4_delta(
    _data,_v4raw,config=_heelmanifest["config"])
if not _v4corr.id.equals(_v4raw.id):
    raise RuntimeError("Heel GR datum order mismatch")
_v4raw["v4_gr_datum"]=_v4corr.v4_gr_datum.to_numpy(float)
_v4raw["gr_shift"]=_v4corr.gr_shift.to_numpy(float)
_v4raw=_v4raw[["id","v4_lgb7","v4_gr_datum","gr_shift"]]''')
s = s.replace(
'''_W={"har_physics":0.3985925949,"har_lgb":0.2873064512,
    "har_xgb":0.0101013985,"pil_blend":0.2752512537,
    "v4_lgb7":0.2479481257}''',
'''_W={"har_physics":0.3725964680,"har_lgb":0.2361893765,
    "har_xgb":0.0740728440,"pil_blend":0.2550928772,
    "v4_lgb7":0.0,"v4_gr_datum":0.2762936306}''')
s = s.replace(
'''              _W["pil_blend"]*_pil_delta+
              _W["v4_lgb7"]*_all.v4_lgb7.to_numpy())''',
'''              _W["pil_blend"]*_pil_delta+
              _W["v4_lgb7"]*_all.v4_lgb7.to_numpy()+
              _W["v4_gr_datum"]*_all.v4_gr_datum.to_numpy())''')
s = s.replace(
'"weight_source":"restricted honest nested GroupKFold CV 8.384425; full OOF positive Ridge alpha=100",',
'"weight_source":"six-leg restricted nested GKF 8.338857, gain .045568, 5/5 folds; full OOF positive Ridge alpha=100",')
s = s.replace(
'"fresh_inference":{"harshini":True,"pilkwang":True,"stack_v4_lgb7":True},',
'"fresh_inference":{"harshini":True,"pilkwang":True,"stack_v4_lgb7":True,"heel_gr_datum":True},\n "heel_gr_shift_by_well":_all.assign(_w=_all.id.str.split("_",n=1).str[0]).groupby("_w").gr_shift.first().to_dict(),')
if "v4_gr_datum" not in s:
    raise RuntimeError("heel integration failed")
nb["cells"][10]["source"] = s
code = "rogii-harshini-pilkwang-v4-heel.ipynb"
(OUT/code).write_text(json.dumps(nb))
meta = json.loads((SRC/"kernel-metadata.json").read_text())
meta["id"] = "boltuzamaki/rogii-harshini-pilkwang-v4-heel"
meta["title"] = "ROGII Harshini Pilkwang V4 Heel GR"
meta["code_file"] = code
if "boltuzamaki/rogii-heel-gr-datum-code" not in meta["dataset_sources"]:
    meta["dataset_sources"].append("boltuzamaki/rogii-heel-gr-datum-code")
(OUT/"kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built", OUT)
