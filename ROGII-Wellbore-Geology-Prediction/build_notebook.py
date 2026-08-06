"""Generate the self-contained Kaggle notebook from the library code."""
import json, os

LIB = open("wellbore_lib.py").read()

md1 = """# ROGII – Wellbore Geology Prediction · LightGBM baseline

**Goal:** for each horizontal well, predict **TVT** (the geological/stratigraphic
depth of the bit) for every point *after* the **Prediction Start (PS)** point.
TVT is known up to PS (`TVT_input`, NaN afterwards); evaluation is the **RMSE**
of `dTVT = trueTVT − predTVT`.

**Approach (strong, regularized baseline):**
1. Anchor target as `dTVT = TVT − TVT_PS` (well-agnostic, mean ≈ 0).
2. Physics-informed features: trajectory deltas from PS, GR + smoothed GR
   stats, and a **typewell GR→TVT correlation candidate** (`gr_impl_d`).
3. **LightGBM** with heavy regularization, then shrink ×0.5 and clip ±40 ft
   (geosteering keeps the well in-zone, so corrections to the anchor are small).

5-fold GroupKFold (by well): **per-well RMSE 12.10 ft vs 12.81 for hold-constant**,
beating the constant baseline in **65 %** of wells.
"""

md_eda = """## Quick EDA
Why a heavily-regularized model: TVT barely moves in the lateral (the well is
geosteered to stay in zone), so the constant-at-PS baseline is already strong and
naive extrapolations are far worse. The model makes small GR-driven corrections."""

eda_code = '''import os, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def find_data_root():
    """Locate the folder that holds train/ test/ sample_submission.csv."""
    roots = ["/kaggle/input/rogii-wellbore-geology-prediction", "data"]
    roots += sorted(glob.glob("/kaggle/input/*"))
    for r in roots:
        if os.path.exists(os.path.join(r, "sample_submission.csv")) and \\
           glob.glob(os.path.join(r, "train", "*__horizontal_well.csv")):
            return r
    # last resort: search recursively for a horizontal_well csv and walk up
    hits = glob.glob("/kaggle/input/**/*__horizontal_well.csv", recursive=True)
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("competition data not found under /kaggle/input")

DATA = find_data_root()
TRAIN, TEST = f"{DATA}/train", f"{DATA}/test"
print("DATA root:", DATA, "| train wells:", len(list_wells(TRAIN)),
      "| test wells:", len(list_wells(TEST)))

w = list_wells(TRAIN)[0]
h, tw = load_well(TRAIN, w)
ps = ps_index(h)
print("example well", w, "| PS row", ps, "| well rows", len(h))

fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(h.MD, h.TVT, "b", lw=1, label="TVT (target)")
ax[0].plot(h.MD, h.TVT_input, "g", lw=2, label="TVT_input (known until PS)")
ax[0].axvline(h.MD.iloc[ps], color="r", ls="--", label="PS")
ax[0].invert_yaxis(); ax[0].legend(); ax[0].set_ylabel("TVT (ft)")
ax[0].set_title(f"{w}: TVT is near-constant in the lateral")
ax[1].plot(h.MD, h.GR, color="purple", lw=0.5); ax[1].axvline(h.MD.iloc[ps], color="r", ls="--")
ax[1].set_ylabel("GR"); ax[1].set_xlabel("MD"); plt.tight_layout(); plt.show()'''

train_code = '''import lightgbm as lgb

print("Building train dataset (773 wells)...")
train = build_dataset(TRAIN, has_target=True)
print("train rows:", len(train))

model = lgb.LGBMRegressor(**LGB_PARAMS)
model.fit(train[FEATURES], train["d_tvt"])
print("model trained")

# feature importances
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp)'''

sub_code = '''sample = pd.read_csv(f"{DATA}/sample_submission.csv")
preds = []
for w in list_wells(TEST):
    h, tw = load_well(TEST, w)
    f = make_features(h, tw, w, has_target=False)   # anchor = TVT_input at PS
    d = postprocess(model.predict(f[FEATURES]))
    preds.append(f.assign(tvt=f["tvt_ps"] + d)[["id", "tvt"]])

pred_df = pd.concat(preds, ignore_index=True)
sub = sample[["id"]].merge(pred_df, on="id", how="left")
sub["tvt"] = sub["tvt"].ffill().fillna(0.0)
sub.to_csv("submission.csv", index=False)
print("submission.csv rows:", len(sub), "| nan:", sub["tvt"].isna().sum())
sub.head()'''

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}
def markdown(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

cells = [
    markdown(md1),
    code(LIB),                 # library: IO + features + model config
    markdown(md_eda),
    code(eda_code),
    code(train_code),
    code(sub_code),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/wellbore_baseline.ipynb", "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print("wrote notebooks/wellbore_baseline.ipynb with", len(cells), "cells")
