"""Assemble the self-contained honest LightGBM physics-stack Kaggle notebook."""
import json, os

PF = open("exp/pf_tracker.py", encoding="utf-8").read()
BEAM = open("exp/beam_tracker.py", encoding="utf-8").read().replace("cache=True", "cache=False")
SIG = open("exp/compute_signals.py", encoding="utf-8").read()
# strip the __main__ block + local-module imports (functions are inlined instead)
SIG = SIG.split("def run(")[0]   # keep helpers + build_well only
for line in ["from pf_tracker import pf_predict",
             "from beam_tracker import beam_predict",
             "from wellbore_lib import ps_index",
             "_H = os.path.dirname(os.path.abspath(__file__))",
             "sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))"]:
    SIG = SIG.replace(line, "")
SIG = "def ps_index(h):\n    return int(h['TVT_input'].notna().sum())\n\n" + SIG

def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

MD = """# ROGII Wellbore Geology — Honest Physics-Signal Stack (target-free)

End-to-end, **no leak, no blending of independent solutions**: one LightGBM on
domain-physics signals predicting `ΔTVT = TVT − last_known_TVT`.

Signals: particle-filter path, beam-search path, multi-scale NCC vs typewell,
Q-3D tortuosity, relative trajectory geometry, typewell-GR residuals at PF
baselines. Uses only observed test logs + typewell (generalises to unseen wells).

Local 5-fold GroupKFold-by-well: row-weighted RMSE 10.62 (vs PF 10.82, constant 15.9)."""

DRIVER = '''import glob, numpy as np, pandas as pd, lightgbm as lgb
from scipy.signal import savgol_filter

def find_root():
    for r in ["/kaggle/input/rogii-wellbore-geology-prediction","data"]+sorted(glob.glob("/kaggle/input/*")):
        if os.path.exists(f"{r}/sample_submission.csv") and glob.glob(f"{r}/train/*__horizontal_well.csv"):
            return r
    hits=glob.glob("/kaggle/input/**/*__horizontal_well.csv",recursive=True)
    if hits: return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("data")
DATA=find_root(); print("DATA:",DATA)

def build_split(split, is_train):
    from joblib import Parallel, delayed
    paths=sorted(glob.glob(f"{DATA}/{split}/*__horizontal_well.csv"))
    res=Parallel(n_jobs=4)(delayed(build_well_p)(p,is_train) for p in paths)
    return pd.concat([r for r in res if r is not None], ignore_index=True)

def build_well_p(path, is_train):
    # build_well expects data/<split>/ layout; here pass the actual dir
    return build_well(path, is_train)

print("computing test signals..."); test=build_split("test", False)
print("computing train signals (takes several minutes)..."); train=build_split("train", True)
print("train", train.shape, "test", test.shape)

DROP={"well","id","target","last_known_tvt"}
FEATS=[c for c in train.columns if c not in DROP]
PARAMS=dict(objective="regression",n_estimators=1200,learning_rate=0.02,num_leaves=127,
            min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,
            reg_lambda=5.0,reg_alpha=1.0,verbose=-1,n_jobs=-1)
model=lgb.LGBMRegressor(**PARAMS).fit(train[FEATS].to_numpy(np.float32), train["target"].to_numpy(np.float32))

SH,CL=0.95,60
tp=np.clip(model.predict(test[FEATS].to_numpy(np.float32))*SH,-CL,CL)
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

# build_well in compute_signals reads f"{base}/{wid}__typewell.csv" from the path dir -> works on Kaggle.
cells = [md(MD), code("import os"), code(PF), code(BEAM), code(SIG), code(DRIVER)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs("kernels/stack", exist_ok=True)
json.dump(nb, open("kernels/stack/wellbore-geo-physics-stack.ipynb", "w"), indent=1)
meta = {"id": "boltuzamaki/rogii-wellbore-geo-physics-stack", "title": "ROGII Wellbore Geo - Physics Stack (honest)",
        "code_file": "wellbore-geo-physics-stack.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [], "competition_sources": ["rogii-wellbore-geology-prediction"], "kernel_sources": []}
json.dump(meta, open("kernels/stack/kernel-metadata.json", "w"), indent=2)
print("wrote kernels/stack/")
