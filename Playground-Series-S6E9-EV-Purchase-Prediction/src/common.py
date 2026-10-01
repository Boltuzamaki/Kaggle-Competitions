"""Shared data / CV / experiment-registry utilities for PS S6E9.

Every experiment uses the SAME StratifiedKFold split (SEED/N_SPLITS below) so that
out-of-fold predictions from different models are mutually comparable and can be
blended without leakage.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OOF = os.path.join(ROOT, "artifacts", "oof")
PRED = os.path.join(ROOT, "artifacts", "preds")
SUBS = os.path.join(ROOT, "submissions")
LOG = os.path.join(ROOT, "artifacts", "experiments.jsonl")
for d in (OOF, PRED, SUBS):
    os.makedirs(d, exist_ok=True)

TARGET = "Will_Buy_EV"
ID = "id"
SEED = 42
# Exploration ran at 5 folds; the final suite uses 10 (more rows per model and a
# sharper in-fold target encoding). Set FOLDS=10 in the environment for the finals.
N_SPLITS = int(os.environ.get("FOLDS", 5))

NUM_COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
            "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
            "Environmental_Concern_Level"]
CAT_COLS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
            "Subsidy_Available", "Range_Anxiety_Level"]

ORDINAL_MAPS = {
    "Range_Anxiety_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Home_Charging_Possible": {"No": 0, "Yes": 1},
    "Subsidy_Available": {"No": 0, "Yes": 1},
    "City_Type": {"Rural": 0, "Suburban": 1, "Urban": 2},
}


def load_raw():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = (tr[TARGET] == "Yes").astype(np.int8).values
    tr = tr.drop(columns=[TARGET])
    return tr, te, y


# FOLD_SEED lets a model be re-run over a DIFFERENT partition of the same rows.
# Averaging test predictions across partitions cuts split variance without
# touching the honesty of any single run's OOF estimate.
FOLD_SEED = int(os.environ.get("FOLD_SEED", SEED))


def get_folds(y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
    return list(skf.split(np.zeros(len(y)), y))


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def add_features(df, level="full"):
    """level: 'raw' (encoding only) | 'lite' | 'full'."""
    d = df.copy()

    for c, mp in ORDINAL_MAPS.items():
        d[c] = d[c].map(mp).astype(np.int8)
    d["Gender"] = d["Gender"].map({"Male": 0, "Female": 1, "Other": 2}).astype(np.int8)
    d["Current_Car_Type"] = d["Current_Car_Type"].map(
        {"Hatchback": 0, "Sedan": 1, "SUV": 2, "Truck": 3}).astype(np.int8)

    if level == "raw":
        return d

    inc = d["Annual_Income_USD"]
    env = d["Environmental_Concern_Level"]
    com = d["Daily_Commute_km"]

    # Censoring flags: Income floors at 30k (9.2% of rows), commute at 5.0km (21.6%).
    d["income_at_floor"] = (inc == 30000).astype(np.int8)
    d["commute_at_floor"] = (com == 5.0).astype(np.int8)

    d["log_income"] = np.log1p(inc)
    # The two hard gates: subsidy present AND low range anxiety.
    d["gate"] = ((d["Subsidy_Available"] == 1) & (d["Range_Anxiety_Level"] == 0)).astype(np.int8)
    d["barrier_score"] = (d["Range_Anxiety_Level"] + (1 - d["Subsidy_Available"]) * 2
                          + (1 - d["Home_Charging_Possible"])).astype(np.int8)

    # Interactions between the dominant drivers.
    d["env_x_subsidy"] = env * d["Subsidy_Available"]
    d["env_x_lowrange"] = env * (2 - d["Range_Anxiety_Level"])
    d["env_x_logincome"] = env * d["log_income"]
    d["logincome_x_subsidy"] = d["log_income"] * d["Subsidy_Available"]
    d["env_x_homecharge"] = env * d["Home_Charging_Possible"]

    if level == "lite":
        return d

    d["stations_total"] = d["Charging_Stations_Near_Home"] + d["Charging_Stations_Near_Work"]
    d["stations_diff"] = d["Charging_Stations_Near_Home"] - d["Charging_Stations_Near_Work"]
    d["access_score"] = d["stations_total"] + 3 * d["Home_Charging_Possible"]
    d["income_per_car"] = inc / d["Number_of_Cars_Owned"]
    d["commute_per_car"] = com / d["Number_of_Cars_Owned"]
    d["commute_x_range"] = com * (d["Range_Anxiety_Level"] + 1)
    d["commute_per_station"] = com / (d["stations_total"] + 1)
    d["income_x_commute"] = d["log_income"] * com
    d["age_centered_abs"] = (d["Age"] - 47).abs()
    d["income_rank"] = inc.rank(pct=True)
    d["readiness"] = (env * 2 + d["Home_Charging_Possible"] + d["Subsidy_Available"] * 2
                      - d["Range_Anxiety_Level"] * 2)
    return d


def build(level="full"):
    tr, te, y = load_raw()
    test_id = te[ID].values
    X = add_features(tr.drop(columns=[ID]), level)
    Xt = add_features(te.drop(columns=[ID]), level)
    Xt = Xt[X.columns]
    return X, Xt, y, test_id


# --------------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------------
def run_experiment(name, fit_predict, X, Xt, y, folds=None, notes="", save=True):
    """fit_predict(X_tr, y_tr, X_va, y_va, X_test, fold) -> (va_pred, test_pred)"""
    folds = folds if folds is not None else get_folds(y)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xt), dtype=np.float64)
    fold_scores = []
    t0 = time.time()

    for f, (itr, iva) in enumerate(folds):
        Xtr = X.iloc[itr] if hasattr(X, "iloc") else X[itr]
        Xva = X.iloc[iva] if hasattr(X, "iloc") else X[iva]
        pv, pt = fit_predict(Xtr, y[itr], Xva, y[iva], Xt, f)
        oof[iva] = pv
        test_pred += np.asarray(pt) / len(folds)
        s = roc_auc_score(y[iva], pv)
        fold_scores.append(s)
        print(f"  [{name}] fold {f}: AUC {s:.6f}", flush=True)

    cv = roc_auc_score(y, oof)
    rec = {
        "name": name, "cv_auc": round(float(cv), 6),
        "fold_mean": round(float(np.mean(fold_scores)), 6),
        "fold_std": round(float(np.std(fold_scores)), 6),
        "folds": [round(float(s), 6) for s in fold_scores],
        "n_features": int(X.shape[1]), "seconds": round(time.time() - t0, 1),
        "notes": notes, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[{name}] CV AUC = {cv:.6f}  (fold mean {np.mean(fold_scores):.6f} "
          f"+/- {np.std(fold_scores):.6f})  {rec['seconds']}s", flush=True)

    if save:
        np.save(os.path.join(OOF, f"{name}.npy"), oof)
        np.save(os.path.join(PRED, f"{name}.npy"), test_pred)
        with open(LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return oof, test_pred, rec


def make_submission(test_pred, test_id, filename):
    path = os.path.join(SUBS, filename)
    pd.DataFrame({ID: test_id, TARGET: test_pred}).to_csv(path, index=False)
    print("wrote", path)
    return path


# --------------------------------------------------------------------------
# Leak-free target encoding
# --------------------------------------------------------------------------
def oof_target_encode(key_tr, y_tr, key_apply_list, n_inner=5, smooth=20.0, seed=SEED):
    """Smoothed mean-target encoding.

    Returns (enc_for_key_tr, [enc for each array in key_apply_list]).
    The training-side encoding is built with an inner KFold so the model never
    sees a row's own label through its encoding; the apply-side encodings use the
    full fold-train statistics.
    """
    from sklearn.model_selection import KFold

    prior = float(np.mean(y_tr))
    s_tr = pd.Series(key_tr)
    enc_tr = np.full(len(key_tr), prior, dtype=np.float32)

    for a, b in KFold(n_inner, shuffle=True, random_state=seed).split(s_tr):
        g = pd.DataFrame({"k": s_tr.values[a], "y": y_tr[a]}).groupby("k")["y"].agg(["sum", "count"])
        m = (g["sum"] + prior * smooth) / (g["count"] + smooth)
        enc_tr[b] = s_tr.values[b].astype(object)
        enc_tr[b] = pd.Series(s_tr.values[b]).map(m).fillna(prior).values.astype(np.float32)

    g = pd.DataFrame({"k": s_tr.values, "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
    m_full = (g["sum"] + prior * smooth) / (g["count"] + smooth)
    outs = [pd.Series(k).map(m_full).fillna(prior).values.astype(np.float32)
            for k in key_apply_list]
    return enc_tr, outs


def oof_target_encode_multi(key_tr, y_tr, key_apply_list, smooths, n_inner=5, seed=SEED):
    """Same as oof_target_encode but returns every smoothing strength from ONE pass
    over the groupbys, which is what makes encoding ~30 keys per fold affordable."""
    from sklearn.model_selection import KFold

    prior = float(np.mean(y_tr))
    ktr = np.asarray(key_tr)
    s_tr = pd.Series(ktr)
    enc_tr = {sm: np.full(len(ktr), prior, dtype=np.float32) for sm in smooths}

    for a, b in KFold(n_inner, shuffle=True, random_state=seed).split(s_tr):
        g = pd.DataFrame({"k": ktr[a], "y": y_tr[a]}).groupby("k")["y"].agg(["sum", "count"])
        kb = pd.Series(ktr[b])
        for sm in smooths:
            m = (g["sum"] + prior * sm) / (g["count"] + sm)
            enc_tr[sm][b] = kb.map(m).fillna(prior).values.astype(np.float32)

    g = pd.DataFrame({"k": ktr, "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
    outs = {}
    for sm in smooths:
        m = (g["sum"] + prior * sm) / (g["count"] + sm)
        outs[sm] = [pd.Series(k).map(m).fillna(prior).values.astype(np.float32)
                    for k in key_apply_list]
    return enc_tr, outs


def add_digit_features(df):
    """Generator-artifact features: base-10 digit decomposition of the
    continuous columns. The synthetic generator wrote target signal into the
    low-order digits of Annual_Income_USD (16-25x the binomial null corridor),
    an artifact absent from the real survey it was trained on."""
    d = df.copy()
    inc = d["Annual_Income_USD"].values.astype(np.int64)
    km10 = np.rint(d["Daily_Commute_km"].values * 10).astype(np.int64)

    for k, nm in [(1, "units"), (10, "tens"), (100, "hundreds"),
                  (1000, "thousands"), (10000, "tenk")]:
        d[f"inc_d_{nm}"] = ((inc // k) % 10).astype(np.int8)
    d["inc_mod100"] = (inc % 100).astype(np.int16)
    d["inc_mod1000"] = (inc % 1000).astype(np.int16)
    d["inc_div1000"] = (inc // 1000).astype(np.int16)
    d["inc_digit_sum"] = sum(((inc // k) % 10) for k in (1, 10, 100, 1000, 10000)).astype(np.int8)

    d["km_dec"] = (km10 % 10).astype(np.int8)
    d["km_int"] = (km10 // 10).astype(np.int16)
    return d
