"""Fast inference-only notebook: compute test-well signals + load trained LGBM."""
import json, os

PF = open("exp/pf_tracker.py", encoding="utf-8").read()
BEAM = open("exp/beam_tracker.py", encoding="utf-8").read().replace("cache=True", "cache=False")
SIG = open("exp/compute_signals.py", encoding="utf-8").read().split("def run(")[0]
for line in ["from pf_tracker import pf_predict", "from beam_tracker import beam_predict",
             "from wellbore_lib import ps_index",
             "_H = os.path.dirname(os.path.abspath(__file__))",
             "sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))"]:
    SIG = SIG.replace(line, "")
SIG = "def ps_index(h):\n    return int(h['TVT_input'].notna().sum())\n\n" + SIG

def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

MD = """# ROGII Wellbore Geology — Honest Physics-Signal Stack (inference)

Target-free, no leak, no blending. One LightGBM (trained offline on 773 wells,
5-fold GroupKFold CV row-RMSE **10.62** vs PF 10.82, constant 15.9) over
domain-physics signals: particle filter, beam search, multi-scale NCC, Q-3D
tortuosity, relative trajectory, typewell-GR residuals. Predicts
`ΔTVT = TVT − last_known_TVT` for the hidden tail; only observed test logs used."""

DRIVER = '''import glob, numpy as np, pandas as pd, lightgbm as lgb, json
from scipy.signal import savgol_filter
from joblib import Parallel, delayed

def find_root():
    for r in ["/kaggle/input/rogii-wellbore-geology-prediction","data"]+sorted(glob.glob("/kaggle/input/*")):
        if os.path.exists(f"{r}/sample_submission.csv") and glob.glob(f"{r}/test/*__horizontal_well.csv"):
            return r
    hits=glob.glob("/kaggle/input/**/*__horizontal_well.csv",recursive=True)
    if hits: return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("data")
DATA=find_root(); print("DATA:",DATA)

def mdir():
    hits=glob.glob("/kaggle/input/**/lgbm_stack.txt", recursive=True) or glob.glob("model_ds/lgbm_stack.txt")
    if not hits:
        raise FileNotFoundError("lgbm_stack.txt not found under /kaggle/input — is the model dataset attached? "+str(sorted(glob.glob("/kaggle/input/*"))))
    return os.path.dirname(hits[0])
MD_DIR=mdir(); print("MODEL:",MD_DIR)
booster=lgb.Booster(model_file=f"{MD_DIR}/lgbm_stack.txt")
meta=json.load(open(f"{MD_DIR}/meta.json")); FEATS=meta["feats"]; SH=meta["shrink"]; CL=meta["clip"]

paths=sorted(glob.glob(f"{DATA}/test/*__horizontal_well.csv"))
test=pd.concat([r for r in Parallel(n_jobs=4)(delayed(build_well)(p,False) for p in paths) if r is not None], ignore_index=True)
print("test signals", test.shape)

tp=np.clip(booster.predict(test[FEATS].to_numpy(np.float32))*SH,-CL,CL)
test=test.assign(pred_d=tp); rows=[]
for w,g in test.groupby("well",sort=False):
    v=g["pred_d"].to_numpy(float); n=len(v); wl=min(31, n if n%2==1 else n-1)
    if wl>=5: v=savgol_filter(v,wl,3)
    rows.append(pd.DataFrame({"id":g["id"].to_numpy(),"tvt":g["last_known_tvt"].to_numpy(float)+v}))
pred=pd.concat(rows,ignore_index=True)
sample=pd.read_csv(f"{DATA}/sample_submission.csv")
sub=sample[["id"]].merge(pred,on="id",how="left")
sub["tvt"]=sub["tvt"].fillna(test["last_known_tvt"].mean())
sub.to_csv("submission.csv",index=False)
print("wrote submission.csv", len(sub), "nan", sub["tvt"].isna().sum()); sub.head()'''

cells = [md(MD), code("import os"), code(PF), code(BEAM), code(SIG), code(DRIVER)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs("kernels/infer", exist_ok=True)
json.dump(nb, open("kernels/infer/wellbore-geo-physics-stack-infer.ipynb", "w"), indent=1)
meta = {"id": "boltuzamaki/rogii-wellbore-geo-physics-stack-infer",
        "title": "ROGII Wellbore Geo - Physics Stack Inference",
        "code_file": "wellbore-geo-physics-stack-infer.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": ["boltuzamaki/rogii-wellbore-lgbm-stack"],
        "competition_sources": ["rogii-wellbore-geology-prediction"], "kernel_sources": []}
json.dump(meta, open("kernels/infer/kernel-metadata.json", "w"), indent=2)
print("wrote kernels/infer/")
