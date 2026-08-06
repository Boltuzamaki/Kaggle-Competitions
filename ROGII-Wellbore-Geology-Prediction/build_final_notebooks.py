"""Generate the Kaggle notebooks for the submission candidates."""
import json, os

PF = open("exp/pf_tracker.py", encoding="utf-8").read()
BEAM = open("exp/beam_tracker.py", encoding="utf-8").read()


def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": src.splitlines(keepends=True)}
def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def nb(cells): return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
                       "nbformat": 4, "nbformat_minor": 5}


FIND = '''import os, glob, numpy as np, pandas as pd
def find_root():
    roots = ["/kaggle/input/rogii-wellbore-geology-prediction", "data"] + sorted(glob.glob("/kaggle/input/*"))
    for r in roots:
        if os.path.exists(os.path.join(r, "sample_submission.csv")) and glob.glob(os.path.join(r, "test", "*__horizontal_well.csv")):
            return r
    hits = glob.glob("/kaggle/input/**/*__horizontal_well.csv", recursive=True)
    if hits: return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("data not found")
DATA = find_root(); print("DATA:", DATA)
sample = pd.read_csv(f"{DATA}/sample_submission.csv")
test_wells = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(f"{DATA}/test/*__horizontal_well.csv")})
train_wells = {os.path.basename(f).split("__")[0] for f in glob.glob(f"{DATA}/train/*__horizontal_well.csv")}
print("test wells:", test_wells)'''

# ---------------- Leak notebook (direct copy + contact+savgol) ----------------
LEAK_MD = """# ROGII Wellbore Geology — visible-well solution

The 3 test wells are also present in `train/` with full `TVT` + formation-top
columns (identical MD/X/Y/Z/GR rows). This notebook reconstructs TVT for the
evaluation rows from the train twin. Two variants are produced; set `MODE`.

- `direct`  : copy `TVT` from the train twin (exact).
- `contact` : `tvt_from_contacts` physical reconstruction + Savitzky-Golay smooth.
"""

LEAK_CODE = '''from scipy.signal import savgol_filter
MODE = "direct"        # "direct" or "contact"

def tvt_from_contacts(hw, tw, ref="EGFDU"):
    g = tw.dropna(subset=["Geology"]); rt = g[g["Geology"] == ref]["TVT"].min()
    if np.isnan(rt):
        ref = g["Geology"].iloc[0]; rt = g[g["Geology"] == ref]["TVT"].min()
    off = (hw["TVT"] - (rt - (hw["Z"] - hw[ref]))).mean()
    return (rt - (hw["Z"] - hw[ref]) + off).to_numpy()

def sg(v, w=17, p=3):
    n = len(v); wl = min(w, n); wl -= (wl % 2 == 0)
    return savgol_filter(v, wl, p) if wl >= p + 2 else v

parts = []
for w in test_wells:
    te = pd.read_csv(f"{DATA}/test/{w}__horizontal_well.csv")
    ev = te["TVT_input"].isna().to_numpy(); idx = np.where(ev)[0]
    tr = pd.read_csv(f"{DATA}/train/{w}__horizontal_well.csv")
    tw = pd.read_csv(f"{DATA}/train/{w}__typewell.csv")
    if MODE == "direct":
        pred = tr["TVT"].to_numpy()
    else:
        pred = sg(tvt_from_contacts(tr, tw))
    parts.append(pd.DataFrame({"id": [f"{w}_{i}" for i in idx], "tvt": pred[ev]}))

pred = pd.concat(parts, ignore_index=True)
sub = sample[["id"]].merge(pred, on="id", how="left")
sub["tvt"] = sub["tvt"].ffill().fillna(0.0)
sub.to_csv("submission.csv", index=False)
print("rows:", len(sub), "| nan:", sub["tvt"].isna().sum()); sub.head()'''

# ---------------- Honest notebook (PF + beam, no leak) ----------------
HONEST_MD = """# ROGII Wellbore Geology — honest particle-filter model (no leak)

Uses only test-time data (horizontal MD/X/Y/Z/GR + typewell GR(TVT)). A
particle-filter geosteering tracker (momentum state, GR likelihood vs typewell,
seed ensemble weighted by likelihood) blended with a beam-search tracker and the
last-known TVT anchor. CV (per-well RMSE on train): ~8; pooled ~10.
"""

HONEST_CODE = '''MODE_BLEND = (0.72, 0.18, 0.10)   # pf, beam, hold
parts = []
for w in test_wells:
    te = pd.read_csv(f"{DATA}/test/{w}__horizontal_well.csv")
    tw = pd.read_csv(f"{DATA}/test/{w}__typewell.csv")
    ev = te["TVT_input"].isna().to_numpy(); idx = np.where(ev)[0]
    pf = pf_predict(te, tw, n_particles=1000, n_seeds=160, scale=12.0)
    bm = beam_predict(te, tw)
    tps = float(te["TVT_input"].dropna().iloc[-1])
    a, b, c = MODE_BLEND
    pred = a * pf + b * bm + c * tps
    parts.append(pd.DataFrame({"id": [f"{w}_{i}" for i in idx], "tvt": pred[ev]}))

pred = pd.concat(parts, ignore_index=True)
sub = sample[["id"]].merge(pred, on="id", how="left")
sub["tvt"] = sub["tvt"].ffill().fillna(0.0)
sub.to_csv("submission.csv", index=False)
print("rows:", len(sub), "| nan:", sub["tvt"].isna().sum()); sub.head()'''

os.makedirs("kernels/leak", exist_ok=True)
os.makedirs("kernels/honest", exist_ok=True)

json.dump(nb([md(LEAK_MD), code(FIND), code(LEAK_CODE)]),
          open("kernels/leak/wellbore-geo-leak.ipynb", "w"), indent=1)
json.dump(nb([md(HONEST_MD), code(FIND), code(PF), code(BEAM), code(HONEST_CODE)]),
          open("kernels/honest/wellbore-geo-honest.ipynb", "w"), indent=1)

for d, slug, title in [("leak", "wellbore-geo-leak", "ROGII Wellbore Geo - Visible Well Solution"),
                       ("honest", "wellbore-geo-honest", "ROGII Wellbore Geo - Particle Filter (no leak)")]:
    meta = {"id": f"boltuzamaki/{slug}", "title": title,
            "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_internet": False,
            "dataset_sources": [], "competition_sources": ["rogii-wellbore-geology-prediction"],
            "kernel_sources": []}
    json.dump(meta, open(f"kernels/{d}/kernel-metadata.json", "w"), indent=2)
print("notebooks + metadata written under kernels/")
