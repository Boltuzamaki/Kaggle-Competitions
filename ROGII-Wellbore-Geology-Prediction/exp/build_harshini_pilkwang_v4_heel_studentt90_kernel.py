"""Build an unpushed fresh-inference heel stack with locked Student-t PF."""
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels/harshini_pilkwang_v4_heel"
OUT = ROOT / "kernels/harshini_pilkwang_v4_heel_studentt90"
OUT.mkdir(parents=True, exist_ok=True)
src_name = "rogii-harshini-pilkwang-v4-heel.ipynb"
code_name = "rogii-harshini-pilkwang-v4-heel-studentt90.ipynb"
nb = json.loads((SRC / src_name).read_text())
cell = "".join(nb["cells"][10]["source"])

# Reuse the exact locked OOF implementation, embedded so no new dataset or
# internet dependency is introduced.
pf_code = (ROOT / "exp/pf_decorr.py").read_text()
marker = "# Identity alignment is always sample-left, never positional except after proof."
pf_inference = pf_code + r'''

# Locked Student-t PF: exact OOF parameters, recomputed from raw test inputs.
_pf_t0 = __import__("time").time()
_pf_rows = []
_pf_wells = _sample.id.str.rsplit("_", n=1).str[0].drop_duplicates().tolist()
for _wi, _well in enumerate(_pf_wells):
    _hc = list((_data / "test").glob(f"{_well}__horizontal_well.csv"))
    if not _hc:
        _hc = [p for p in _data.rglob(f"{_well}__horizontal_well.csv")
               if "train" not in p.parts]
    _tc = list((_data / "test").glob(f"{_well}__typewell*.csv"))
    if not _tc:
        _tc = [p for p in _data.rglob(f"{_well}__typewell*.csv")
               if "train" not in p.parts]
    if len(_hc) != 1 or len(_tc) != 1:
        raise RuntimeError(f"Student-t PF file identity failure {_well}: {len(_hc)}/{len(_tc)}")
    _hw = _pd.read_csv(_hc[0])
    _tw = _pd.read_csv(_tc[0])
    _pp = pf_predict_decorr(
        _hw, _tw, n_particles=250, n_seeds=40, scale=10.0,
        decorrelate=True, seed0=41000, gs_mult=1.0, gs_floor=45.0,
        student_nu=10.0)
    _last = float(_hw.loc[_hw.TVT_input.notna(), "TVT_input"].iloc[-1])
    _wanted = _sample.loc[_sample.id.str.startswith(_well + "_"), "id"]
    for _id in _wanted:
        _row = int(_id.rsplit("_", 1)[1])
        if _row >= len(_pp) or not _pd.isna(_hw.TVT_input.iloc[_row]):
            raise RuntimeError(f"Student-t PF row identity failure: {_id}")
        _pf_rows.append((_id, float(_pp[_row] - _last)))
_student = _pd.DataFrame(_pf_rows, columns=["id", "student_t_pf"])
if len(_student) != len(_sample) or not _student.id.is_unique:
    raise RuntimeError("Student-t PF sample coverage failure")
_pf_seconds = __import__("time").time() - _pf_t0

'''
if cell.count(marker) != 1:
    raise RuntimeError("identity insertion marker changed")
cell = cell.replace(marker, pf_inference + marker)
cell = cell.replace(
    '''_all=_all.merge(_v4raw,on="id",how="left",validate="one_to_one")''',
    '''_all=_all.merge(_v4raw,on="id",how="left",validate="one_to_one")
_all=_all.merge(_student,on="id",how="left",validate="one_to_one")''')

# Full-fit positive Ridge(alpha=100, stride=8) replacement coefficients were
# fitted on all 765 legal OOF wells.  The deployed vector is exactly
# 0.10*accepted + 0.90*replacement, matching the strict 5/5 CV policy.
old_weights = '''_W={"har_physics":0.3725964680,"har_lgb":0.2361893765,
    "har_xgb":0.0740728440,"pil_blend":0.2550928772,
    "v4_lgb7":0.0,"v4_gr_datum":0.2762936306}'''
new_weights = '''_W={"har_physics":0.0677693487494662,
    "har_lgb":0.29089471129654937,
    "har_xgb":0.10876568491194674,
    "pil_blend":0.30736212828335663,
    "v4_lgb7":0.0,
    "student_t_pf":0.25627609542269963,
    "v4_gr_datum":0.1381911679318975}'''
if cell.count(old_weights) != 1:
    raise RuntimeError("accepted weights block changed")
cell = cell.replace(old_weights, new_weights)
old_sum = '''              _W["v4_lgb7"]*_all.v4_lgb7.to_numpy()+
              _W["v4_gr_datum"]*_all.v4_gr_datum.to_numpy())'''
new_sum = '''              _W["v4_lgb7"]*_all.v4_lgb7.to_numpy()+
              _W["student_t_pf"]*_all.student_t_pf.to_numpy()+
              _W["v4_gr_datum"]*_all.v4_gr_datum.to_numpy())'''
if cell.count(old_sum) != 1:
    raise RuntimeError("meta sum block changed")
cell = cell.replace(old_sum, new_sum)
cell = cell.replace(
    '"weight_source":"six-leg restricted nested GKF 8.338857, gain .045568, 5/5 folds; full OOF positive Ridge alpha=100",',
    '"weight_source":"10% accepted + 90% Student-t replacement; strict GKF 8.167141 vs 8.338857, 5/5; replacement full-fit positive Ridge alpha=100 stride=8",')
cell = cell.replace(
    '"fresh_inference":{"harshini":True,"pilkwang":True,"stack_v4_lgb7":True,"heel_gr_datum":True},',
    '"fresh_inference":{"harshini":True,"pilkwang":True,"stack_v4_lgb7":True,"heel_gr_datum":True,"student_t_pf":True},\n "student_t_pf":{"particles":250,"seeds":40,"seed0":41000,"scale":10.0,"gs_floor":45.0,"student_nu":10.0,"seconds":_pf_seconds},')
if "student_t_pf" not in cell or "_pf_seconds" not in cell:
    raise RuntimeError("Student-t integration failed")
nb["cells"][10]["source"] = cell
nb["cells"][10]["outputs"] = []
nb["cells"][10]["execution_count"] = None
(OUT / code_name).write_text(json.dumps(nb))

meta = json.loads((SRC / "kernel-metadata.json").read_text())
meta.update({
    "id": "boltuzamaki/rogii-harshini-pilkwang-v4-heel-studentt90",
    "title": "ROGII Fresh Heel Student-t PF 90",
    "code_file": code_name,
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "machine_shape": "Cpu",
})
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("built", OUT)
