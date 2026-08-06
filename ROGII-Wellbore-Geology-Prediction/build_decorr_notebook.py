"""Self-contained decorrelated-PF-ensemble notebook (runs on the 3 test wells)."""
import json, os
PFD = open("exp/pf_decorr.py", encoding="utf-8").read()

def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

MD = """# ROGII — Decorrelated Particle-Filter Ensemble (variance reduction)

Honest, target-free. Motivated by *"The Wiggle Is Free, the Trend Is the Wall"*:
the isolated PF caps at ~11 ft (holdout); the real transferable gain is **variance
reduction** via a **decorrelated** ensemble — each candidate draws its own dynamics
(process noise, GR-likelihood width, resample threshold, init spread) around the
tuned optimum, so candidate errors are more independent and their mean drops below
the seed-noise floor. Only the 3 test wells are needed → fast, self-contained."""

DRIVER = '''import os, glob, numpy as np, pandas as pd
from joblib import Parallel, delayed

N_PARTICLES, N_SEEDS, SCALE = 600, 400, 10.0   # heavy ensemble (only 3 wells)

def find_root():
    for r in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction","data"]+sorted(glob.glob("/kaggle/input/*")):
        if os.path.exists(f"{r}/sample_submission.csv") and glob.glob(f"{r}/test/*__horizontal_well.csv"):
            return r
    hits=glob.glob("/kaggle/input/**/*__horizontal_well.csv",recursive=True)
    if hits: return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("data")
DATA=find_root(); print("DATA:",DATA)

def predict_well(path):
    wid=os.path.basename(path).split("__")[0]
    h=pd.read_csv(path); tw=pd.read_csv(f"{os.path.dirname(path)}/{wid}__typewell.csv")
    ev=h["TVT_input"].isna().to_numpy(); idx=np.where(ev)[0]
    pred=pf_predict_decorr(h, tw, N_PARTICLES, N_SEEDS, scale=SCALE, decorrelate=True)
    return pd.DataFrame({"id":[f"{wid}_{i}" for i in idx], "tvt":pred[ev]})

paths=sorted(glob.glob(f"{DATA}/test/*__horizontal_well.csv"))
parts=Parallel(n_jobs=len(paths))(delayed(predict_well)(p) for p in paths)  # 1 job/well across CPUs
pred=pd.concat(parts, ignore_index=True)
sample=pd.read_csv(f"{DATA}/sample_submission.csv")
sub=sample[["id"]].merge(pred, on="id", how="left")
sub["tvt"]=sub["tvt"].ffill().fillna(0.0)
sub.to_csv("submission.csv", index=False)
print("wrote submission.csv", len(sub), "nan", sub["tvt"].isna().sum()); sub.head()'''

cells = [md(MD), code(PFD), code(DRIVER)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs("kernels/decorr", exist_ok=True)
json.dump(nb, open("kernels/decorr/rogii-decorrelated-pf-ensemble.ipynb", "w"), indent=1)
meta = {"id": "boltuzamaki/rogii-decorrelated-pf-ensemble",
        "title": "ROGII Decorrelated PF Ensemble",
        "code_file": "rogii-decorrelated-pf-ensemble.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [], "competition_sources": ["rogii-wellbore-geology-prediction"], "kernel_sources": []}
json.dump(meta, open("kernels/decorr/kernel-metadata.json", "w"), indent=2)
print("wrote kernels/decorr/")
