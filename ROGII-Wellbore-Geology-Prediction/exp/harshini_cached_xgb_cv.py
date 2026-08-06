"""Fast local replay of Harshini's final feature factory from downloaded caches.

The expensive surface imputation and particle filtering are not recomputed.
This executes the exact trust/row-feature code from the completed private
notebook, adds a GPU XGBoost OOF leg, and persists the matrix and all OOF legs.
"""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"exp/results/harshini_kernel_source/rogii-harshini-scratch-stack.ipynb"
CACHE_DIR = ROOT/"exp/results/harshini_kernel_output"
OUT = ROOT/"exp/results/harshini_cached_xgb"
OUT.mkdir(parents=True, exist_ok=True)

nb = json.loads(SRC.read_text())
cells = ["".join(c.get("source", [])) for c in nb["cells"]]

# Data harness only. Point it directly at the local official dataset.
harness = cells[0].replace(
    'CANDS = ["/kaggle/input/rogii-wellbore-geology-prediction",\n'
    '         "/kaggle/input/competitions/rogii-wellbore-geology-prediction", "."]',
    f'CANDS = [r"{ROOT/"data"}"]')
ns = {"__name__": "harshini_cached_replay"}
exec(compile(harness, str(SRC)+":cell0", "exec"), ns)

# Recreate the exact learned surface-trust variables from imp_cache.
os.chdir(CACHE_DIR)
ns["FORM"] = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
exec(compile(cells[6], str(SRC)+":cell6", "exec"), ns)

# Execute only CV portion of the final cell. Inference needs the spatial point
# cloud, but is irrelevant to this OOF experiment.
final = cells[8].split("# ---- inference ----")[0]
final = final.replace("@njit(cache=True, nogil=True)", "@njit(cache=False, nogil=True)")
final = final.replace(
    'PF_CACHE="pf_train_cache.pkl"',
    f'PF_CACHE=r"{CACHE_DIR/"pf_train_cache.pkl"}"')

# Persist the expensive-but-deterministic assembled row matrix.
needle = 'sub_i=np.arange(len(RR_))%TRAIN_STRIDE==0'
final = final.replace(
    needle,
    needle + f'\nRR_.to_pickle(r"{OUT/"rows.pkl"}")\n'
    + f'np.savez_compressed(r"{OUT/"matrix_meta.npz"}", y=y, groups=g, sub_i=sub_i)')

# Add k256's third boosted family. Harshini already has two LGB variants and
# CatBoost; XGBoost runs histogram training on the local RTX GPU.
needle = "if USE_CB:\n    def cbr(): return CatBoostRegressor(iterations=4000,depth=8,learning_rate=0.03,l2_leaf_reg=3.0,\n        loss_function=\"RMSE\",verbose=0,random_seed=7)\n    base['cb']=oof(cbr)"
replacement = needle + """
import xgboost as xgb
def xgbr():
    return xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=1600, learning_rate=0.025,
        max_depth=8, min_child_weight=25, subsample=0.80,
        colsample_bytree=0.70, reg_alpha=0.25, reg_lambda=12.0,
        tree_method="hist", device="cuda", max_bin=256,
        random_state=256, n_jobs=max(1, (os.cpu_count() or 8)//2))
base['xgb']=oof(xgbr)
"""
if needle not in final:
    raise RuntimeError("Could not locate ensemble insertion point")
final = final.replace(needle, replacement)

# Save all legs and fitted ridge for later iteration without retraining.
final += f"""
import joblib
joblib.dump({{"oofs":base, "ridge":ridge, "features":FEATS_R,
             "ridge_columns":["blend"]+list(base)}},
            r"{OUT/"oof_ensemble.joblib"}", compress=3)
pd.DataFrame({{"leg":["blend"]+list(base),
              "weight":ridge.coef_}}).to_csv(r"{OUT/"ridge_weights.csv"}",index=False)
"""
exec(compile(final, str(SRC)+":cell8_cv_xgb", "exec"), ns)
