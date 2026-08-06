"""Prune all train-CV/cache loops from the five-leg fresh inference kernel."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"kernels/harshini_pilkwang_v4"
OUT=ROOT/"kernels/harshini_pilkwang_v4_fast"
OUT.mkdir(parents=True,exist_ok=True)
nb=json.loads((SRC/"rogii-harshini-pilkwang-fresh-inference.ipynb").read_text())
src=["".join(c.get("source",[])) for c in nb["cells"]]
src=[s.replace("@njit(cache=True)","@njit(cache=False)")
       .replace("@njit(cache=True, nogil=True)","@njit(cache=False, nogil=True)")
     for s in src]

# Keep data/function/spatial-index construction, remove diagnostic scoring loops.
src[0]=src[0].split("dz=[]; dtvt=[]; flat_res=[]")[0]
src[1]=src[1].split("# ---- score on a SUBSET first")[0]
src[2]=src[2].split("# ---- leave-one-well-out evaluation ----")[0]
src[3]=src[3].split("# --- 3. evaluate ---")[0]
src[4]=src[4].split("random.seed(0); SUB")[0]
# Cache is unnecessary at inference; retain exact surface datum controls only.
src[5]='''# Inference-only controls; training CACHE is packaged into TRUST weights.
DATUM_WINDOWS=[("full",None),("late2",1500),("late",500),("last",150)]
PR_WIN=500
'''
src[6]='''# Per-well TRUST is loaded from the private legal model package below.
'''

# Replace the small trust fit over CACHE/R with its packaged full-data booster.
s=src[8]
old='''TRUST=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=20,
    subsample=0.8,colsample_bytree=0.8,reg_lambda=5.0,verbose=-1).fit(R[FEATS],R.w_opt)'''
new='''_tcfg=_har_manifest["trust_model"]
SH0=float(_tcfg["SH0"]); CL0=float(_tcfg["CL0"])
A_TRUST=float(_tcfg["A_TRUST"]); FEATS=list(_tcfg["features"])
TRUST=_lgb.Booster(model_file=str(_har_pkg/"harshini_trust_lgb.txt"))'''
if old not in s: raise RuntimeError("TRUST fit replacement point missing")
src[8]=s.replace(old,new)
for i,v in enumerate(src): nb["cells"][i]["source"]=v

code_name="rogii-harshini-pilkwang-v4-fast.ipynb"
(OUT/code_name).write_text(json.dumps(nb))
meta=json.loads((SRC/"kernel-metadata.json").read_text())
meta.pop("id_no",None)
meta["id"]="boltuzamaki/rogii-harshini-pilkwang-v4-meta-fast"
meta["title"]="ROGII Harshini Pilkwang V4 Meta Fast"
meta["code_file"]=code_name; meta["enable_gpu"]=False; meta["machine_shape"]="Cpu"
(OUT/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
print("built",OUT)
