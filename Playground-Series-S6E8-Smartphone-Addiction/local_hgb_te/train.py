"""Original local HGB model with leakage-safe leave-one-out encodings."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local_hgb_te" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
TARGET, ID, SEED = "addicted_label", "id", 20260803


def levels(frame, cols):
    return pd.DataFrame({c: frame[c].fillna("__NA__").astype(str) for c in cols})


def base_numeric(frame, raw_cols, medians, cat_maps):
    out = {}
    for c in raw_cols:
        if pd.api.types.is_numeric_dtype(frame[c]):
            out[f"raw_{c}"] = frame[c].fillna(medians[c]).astype("float32")
            out[f"na_{c}"] = frame[c].isna().astype("float32")
        else:
            out[f"cat_{c}"] = frame[c].fillna("__NA__").astype(str).map(cat_maps[c]).fillna(-1).astype("float32")
            out[f"na_{c}"] = frame[c].isna().astype("float32")
    x = pd.DataFrame(out)
    x["missing_count"] = frame[raw_cols].isna().sum(axis=1).astype("float32")
    d = frame["daily_screen_time_hours"]
    s = frame["social_media_hours"]
    g = frame["gaming_hours"]
    w = frame["work_study_hours"]
    wk = frame["weekend_screen_time"]
    x["component_residual"] = (d - s - g - w).astype("float32")
    x["weekend_gap"] = (wk - d).astype("float32")
    x["leisure_share"] = ((s + g) / (d + 0.1)).astype("float32")
    x["work_share"] = (w / (d + 0.1)).astype("float32")
    return x


def encoded_train(lv, y, smooth=10.0):
    gm = float(y.mean())
    out = {}
    for c in lv:
        tmp = pd.DataFrame({"lv": lv[c].values, "y": y})
        count = tmp.groupby("lv")["y"].transform("count").astype("float32")
        total = tmp.groupby("lv")["y"].transform("sum").astype("float32")
        out[f"te_{c}"] = ((total - y + smooth * gm) / (count - 1 + smooth)).astype("float32")
        out[f"fq_{c}"] = np.log1p(count).astype("float32")
    return pd.DataFrame(out)


def encoded_apply(train_lv, y, apply_lv, smooth=10.0):
    gm = float(y.mean())
    out = {}
    for c in train_lv:
        stats = pd.DataFrame({"lv": train_lv[c].values, "y": y}).groupby("lv")["y"].agg(["sum", "count"])
        te = (stats["sum"] + smooth * gm) / (stats["count"] + smooth)
        out[f"te_{c}"] = apply_lv[c].map(te).fillna(gm).astype("float32")
        out[f"fq_{c}"] = np.log1p(apply_lv[c].map(stats["count"]).fillna(0)).astype("float32")
    return pd.DataFrame(out)


train = pd.read_csv(ROOT / "train.csv")
test = pd.read_csv(ROOT / "test.csv")
sample = pd.read_csv(ROOT / "sample_submission.csv")
cols = [c for c in test if c != ID]
y = train[TARGET].to_numpy(dtype=np.int8)
tr_lv, te_lv = levels(train, cols), levels(test, cols)
medians = {c: train[c].median() for c in cols if pd.api.types.is_numeric_dtype(train[c])}
cat_maps = {c: {v: i for i, v in enumerate(sorted(train[c].fillna("__NA__").astype(str).unique()))}
            for c in cols if not pd.api.types.is_numeric_dtype(train[c])}
tr_base = base_numeric(train, cols, medians, cat_maps)
te_base = base_numeric(test, cols, medians, cat_maps)

folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = np.zeros(len(train), dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float64)
fold_scores = []
fold_id = np.full(len(train), -1, dtype=np.int8)

for fold, (itr, iva) in enumerate(folds.split(train, y)):
    fold_id[iva] = fold
    e_tr = encoded_train(tr_lv.iloc[itr].reset_index(drop=True), y[itr])
    e_va = encoded_apply(tr_lv.iloc[itr], y[itr], tr_lv.iloc[iva]).reset_index(drop=True)
    e_te = encoded_apply(tr_lv.iloc[itr], y[itr], te_lv).reset_index(drop=True)
    xa = pd.concat([tr_base.iloc[itr].reset_index(drop=True), e_tr], axis=1)
    xb = pd.concat([tr_base.iloc[iva].reset_index(drop=True), e_va], axis=1)
    xt = pd.concat([te_base.reset_index(drop=True), e_te], axis=1)
    model = HistGradientBoostingClassifier(
        learning_rate=0.075, max_iter=650, max_leaf_nodes=31,
        min_samples_leaf=80, l2_regularization=2.0,
        validation_fraction=None, random_state=SEED + fold,
    )
    model.fit(xa, y[itr])
    oof[iva] = model.predict_proba(xb)[:, 1]
    test_pred += model.predict_proba(xt)[:, 1] / 5
    score = roc_auc_score(y[iva], oof[iva])
    fold_scores.append(float(score))
    print(f"fold={fold} auc={score:.8f}", flush=True)

auc = float(roc_auc_score(y, oof))
pd.DataFrame({ID: train[ID], TARGET: y, "fold": fold_id, "prediction": oof}).to_csv(OUT / "oof.csv", index=False)
pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(OUT / "test_predictions.csv", index=False)
(OUT / "metrics.json").write_text(json.dumps({"oof_auc": auc, "fold_auc": fold_scores}, indent=2))
print(f"OOF AUC={auc:.8f}")
