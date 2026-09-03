"""From-scratch, private S6E8 nested target-encoding XGBoost experiment.

Uses only the attached competition tables. Produces genuine outer-fold OOF and
test predictions; it never reads prediction files or public notebook outputs.
"""

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

TARGET = "addicted_label"
ID = "id"
SEED = 20260803
OUTER_FOLDS = 5
INNER_FOLDS = 4


def locate_data():
    root = Path("/kaggle/input")
    print("INPUT TREE (first 300 files):")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for p in files[:300]:
        print(p)
    train_candidates = [p for p in files if p.name.lower() == "train.csv"]
    for train_path in train_candidates:
        try:
            cols = pd.read_csv(train_path, nrows=2).columns
        except Exception:
            continue
        if TARGET in cols:
            folder = train_path.parent
            test_path = folder / "test.csv"
            sample_path = folder / "sample_submission.csv"
            if test_path.exists() and sample_path.exists():
                print("COMPETITION DATA:", folder)
                return train_path, test_path, sample_path
    raise FileNotFoundError(
        "Could not find train.csv containing addicted_label plus sibling test.csv "
        "and sample_submission.csv. Check the competition source attachment above."
    )


def feature_engineer(df):
    x = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    base_numeric = x.select_dtypes(include=np.number).columns.tolist()
    x["missing_count"] = x.isna().sum(axis=1).astype("int8")
    # Missing pattern is a repeated categorical signal, not an integer magnitude.
    x["missing_pattern"] = x[base_numeric].isna().astype("uint8").astype(str).agg("".join, axis=1)

    def ratio(name, a, b, offset=0.25):
        if a in x and b in x:
            x[name] = x[a] / (x[b] + offset)

    parts = ["social_media_hours", "gaming_hours", "work_study_hours"]
    if all(c in x for c in parts):
        x["component_sum"] = x[parts].sum(axis=1, min_count=1)
        x["component_known_count"] = x[parts].notna().sum(axis=1).astype("int8")
        if "daily_screen_time_hours" in x:
            x["screen_residual"] = x["daily_screen_time_hours"] - x["component_sum"]
            for c in parts:
                ratio(c + "_share", c, "daily_screen_time_hours")
    if "weekend_screen_time" in x and "daily_screen_time_hours" in x:
        x["weekend_daily_gap"] = x["weekend_screen_time"] - x["daily_screen_time_hours"]
        ratio("weekend_daily_ratio", "weekend_screen_time", "daily_screen_time_hours")
    ratio("screen_sleep_ratio", "daily_screen_time_hours", "sleep_hours")
    ratio("notifications_per_open", "notifications_per_day", "app_opens_per_day", 1.0)
    ratio("opens_per_screen_hour", "app_opens_per_day", "daily_screen_time_hours")
    ratio("notifications_per_screen_hour", "notifications_per_day", "daily_screen_time_hours")
    return x


def value_key(s):
    # Exact repeated values are intentional in this synthetic dataset. String
    # keys preserve missing as a level and avoid NaN mapping surprises.
    return s.astype("string").fillna("__NA__")


def frequency_features(train_x, test_x, cols):
    out_tr, out_te = {}, {}
    n = len(train_x) + len(test_x)
    for c in cols:
        all_keys = pd.concat([value_key(train_x[c]), value_key(test_x[c])], ignore_index=True)
        freq = all_keys.value_counts(dropna=False) / n
        out_tr[c + "__freq"] = value_key(train_x[c]).map(freq).fillna(0).astype("float32")
        out_te[c + "__freq"] = value_key(test_x[c]).map(freq).fillna(0).astype("float32")
    return pd.DataFrame(out_tr), pd.DataFrame(out_te)


def mapping(keys, y, prior, smooth=30.0):
    stats = pd.DataFrame({"k": keys.to_numpy(), "y": np.asarray(y)}).groupby("k")["y"].agg(["sum", "count"])
    return ((stats["sum"] + smooth * prior) / (stats["count"] + smooth)).to_dict()


def nested_te(train_x, y, valid_x, test_x, cols, seed):
    """Inner-OOF TE for outer-train; outer-train maps for valid and test."""
    prior = float(np.mean(y))
    te_tr = pd.DataFrame(index=train_x.index)
    te_va = pd.DataFrame(index=valid_x.index)
    te_te = pd.DataFrame(index=test_x.index)
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    y_arr = np.asarray(y)
    for c in cols:
        tr_keys = value_key(train_x[c]).reset_index(drop=True)
        oof = np.full(len(train_x), prior, dtype="float32")
        for fit_i, enc_i in inner.split(np.zeros(len(y_arr)), y_arr):
            mp = mapping(tr_keys.iloc[fit_i], y_arr[fit_i], float(y_arr[fit_i].mean()))
            oof[enc_i] = tr_keys.iloc[enc_i].map(mp).fillna(prior).to_numpy("float32")
        full_map = mapping(tr_keys, y_arr, prior)
        name = c + "__te"
        te_tr[name] = oof
        te_va[name] = value_key(valid_x[c]).map(full_map).fillna(prior).to_numpy("float32")
        te_te[name] = value_key(test_x[c]).map(full_map).fillna(prior).to_numpy("float32")
    return te_tr.reset_index(drop=True), te_va.reset_index(drop=True), te_te.reset_index(drop=True)


def raw_numeric(train_x, valid_x, test_x):
    """Raw numbers plus joint integer coding for strings; no target information."""
    tr, va, te = train_x.copy(), valid_x.copy(), test_x.copy()
    for c in tr.columns:
        if not pd.api.types.is_numeric_dtype(tr[c]):
            joint = pd.concat([value_key(tr[c]), value_key(va[c]), value_key(te[c])], ignore_index=True)
            codes, _ = pd.factorize(joint, sort=True)
            tr[c] = codes[:len(tr)]
            va[c] = codes[len(tr):len(tr)+len(va)]
            te[c] = codes[len(tr)+len(va):]
    return (tr.astype("float32").reset_index(drop=True),
            va.astype("float32").reset_index(drop=True),
            te.astype("float32").reset_index(drop=True))


train_path, test_path, sample_path = locate_data()
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)
y = train[TARGET].astype("int8").to_numpy()
train_x = feature_engineer(train)
test_x = feature_engineer(test)
encode_cols = list(train_x.columns)  # every raw and engineered repeated value
freq_tr, freq_te = frequency_features(train_x, test_x, encode_cols)

outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train), dtype="float64")
test_folds = []
fold_scores = []
best_iterations = []

for fold, (fit_i, val_i) in enumerate(outer.split(train_x, y), 1):
    print(f"\n===== OUTER FOLD {fold}/{OUTER_FOLDS} =====")
    fit_raw = train_x.iloc[fit_i]
    val_raw = train_x.iloc[val_i]
    te_fit, te_val, te_test = nested_te(
        fit_raw, y[fit_i], val_raw, test_x, encode_cols, SEED + fold
    )
    raw_fit, raw_val, raw_test = raw_numeric(fit_raw, val_raw, test_x)
    X_fit = pd.concat([raw_fit, freq_tr.iloc[fit_i].reset_index(drop=True), te_fit], axis=1)
    X_val = pd.concat([raw_val, freq_tr.iloc[val_i].reset_index(drop=True), te_val], axis=1)
    X_test = pd.concat([raw_test, freq_te.reset_index(drop=True), te_test], axis=1)

    common = dict(
        n_estimators=1800, learning_rate=0.035, max_depth=7,
        min_child_weight=10, subsample=0.82, colsample_bytree=0.82,
        reg_alpha=0.20, reg_lambda=6.0, gamma=0.01,
        objective="binary:logistic", eval_metric="auc", random_state=SEED + fold,
        n_jobs=-1, tree_method="hist", early_stopping_rounds=120,
    )
    try:
        model = XGBClassifier(device="cuda", **common)
        model.fit(X_fit, y[fit_i], eval_set=[(X_val, y[val_i])], verbose=100)
        device = "cuda"
    except Exception as exc:
        print("GPU training failed; retrying CPU:", repr(exc))
        model = XGBClassifier(device="cpu", **common)
        model.fit(X_fit, y[fit_i], eval_set=[(X_val, y[val_i])], verbose=100)
        device = "cpu"
    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    oof[val_i] = val_pred
    test_folds.append(test_pred)
    score = float(roc_auc_score(y[val_i], val_pred))
    fold_scores.append(score)
    best_iterations.append(int(model.best_iteration + 1))
    print(f"fold={fold} device={device} auc={score:.8f} best_iteration={model.best_iteration + 1}")
    del model, X_fit, X_val, X_test, te_fit, te_val, te_test
    gc.collect()

oof_auc = float(roc_auc_score(y, oof))
pred = np.mean(test_folds, axis=0)
print(f"OOF AUC={oof_auc:.8f}")
fold_id = np.empty(len(train), dtype="int8")
for fold, (_, val_i) in enumerate(outer.split(train_x, y), 1):
    fold_id[val_i] = fold
pd.DataFrame({ID: train[ID], "fold": fold_id, "y": y, "pred": oof}).to_csv("/kaggle/working/oof_nested_te_xgb.csv", index=False)
sample[TARGET] = pred
sample.to_csv("/kaggle/working/submission_nested_te_xgb.csv", index=False)
metrics = {
    "provenance": "from scratch; competition train/test only",
    "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
    "fold_auc": fold_scores, "oof_auc": oof_auc,
    "best_iterations": best_iterations,
    "encoded_columns": encode_cols,
}
Path("/kaggle/working/metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
