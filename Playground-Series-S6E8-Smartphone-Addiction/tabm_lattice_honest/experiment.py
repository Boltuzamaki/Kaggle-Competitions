"""Leakage-safe TabM with exact-value and lattice evidence; official data only."""
from pathlib import Path
import gc, json, subprocess, sys, time

if Path("/kaggle").exists():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit==1.7.3"])
    # Kaggle's current default wheel omits Pascal (sm_60), while GPU sessions
    # may receive a P100. The cu121 2.5.1 wheel includes the required kernels.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--index-url",
        "https://download.pytorch.org/whl/cu121", "torch==2.5.1",
        "torchvision==0.20.1", "torchaudio==2.5.1",
    ])

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID, SEED, FOLDS, INNER = "addicted_label", "id", 20260803, 5, 4


def locate():
    roots = [Path("/kaggle/input"), Path(".")]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("train.csv"):
            try:
                cols = pd.read_csv(path, nrows=1).columns
            except Exception:
                continue
            if TARGET in cols and (path.parent / "test.csv").exists():
                return path, path.parent / "test.csv"
    raise FileNotFoundError("competition train/test tables not found")


def value_key(s, digits=None):
    if pd.api.types.is_numeric_dtype(s):
        z = s if digits is None else s.round(digits)
        return z.astype("string").fillna("__NA__")
    return s.astype("string").fillna("__NA__")


def make_view(df):
    raw = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    out = pd.DataFrame(index=df.index)
    for col in raw:
        if pd.api.types.is_numeric_dtype(raw[col]):
            out[col] = raw[col].astype("float32")
            out[f"{col}__na"] = raw[col].isna().astype("float32")
        else:
            out[col] = raw[col].astype("string").fillna("__NA__").astype("category")

    out["missing_count"] = raw.isna().sum(axis=1).astype("float32")
    parts = ["social_media_hours", "gaming_hours", "work_study_hours"]
    present = raw[parts].notna().sum(axis=1)
    component_sum = raw[parts].sum(axis=1, min_count=1)
    out["component_present"] = present.astype("float32")
    out["component_sum"] = component_sum.astype("float32")
    out["screen_residual"] = (raw["daily_screen_time_hours"] - component_sum).astype("float32")
    out["weekend_residual"] = (raw["weekend_screen_time"] - component_sum).astype("float32")
    out["weekend_gap"] = (raw["weekend_screen_time"] - raw["daily_screen_time_hours"]).astype("float32")
    for name, a, b, eps in [
        ("social_share", "social_media_hours", "daily_screen_time_hours", .1),
        ("gaming_share", "gaming_hours", "daily_screen_time_hours", .1),
        ("work_share", "work_study_hours", "daily_screen_time_hours", .1),
        ("screen_sleep", "daily_screen_time_hours", "sleep_hours", .1),
        ("weekend_daily", "weekend_screen_time", "daily_screen_time_hours", .1),
        ("notif_open", "notifications_per_day", "app_opens_per_day", 1.),
    ]:
        out[name] = (raw[a] / (raw[b] + eps)).replace([np.inf, -np.inf], np.nan).astype("float32")
    return out


def make_keys(df):
    raw = df.drop(columns=[ID, TARGET], errors="ignore")
    keys = {}
    for col in raw:
        keys[f"single__{col}__raw"] = value_key(raw[col])
        if pd.api.types.is_numeric_dtype(raw[col]):
            keys[f"single__{col}__r1"] = value_key(raw[col], 1)
    for a, b in [
        ("daily_screen_time_hours", "weekend_screen_time"),
        ("daily_screen_time_hours", "social_media_hours"),
        ("social_media_hours", "gaming_hours"),
        ("social_media_hours", "work_study_hours"),
        ("notifications_per_day", "app_opens_per_day"),
        ("stress_level", "academic_work_impact"),
    ]:
        for tag, digits in [("raw", None), ("r1", 1)]:
            keys[f"pair__{a}__{b}__{tag}"] = value_key(raw[a], digits) + "|" + value_key(raw[b], digits)
    return pd.DataFrame(keys, index=df.index)


def map_evidence(fit_key, apply_key, y, smooth):
    tmp = pd.DataFrame({"k": fit_key.to_numpy(), "y": y})
    agg = tmp.groupby("k", sort=False, observed=True).y.agg(["sum", "count"])
    prior = float(np.mean(y))
    te = (agg["sum"] + smooth * prior) / (agg["count"] + smooth)
    return (
        apply_key.map(te).fillna(prior).to_numpy("float32"),
        np.log1p(apply_key.map(agg["count"]).fillna(0)).to_numpy("float32"),
    )


def fold_evidence(keys_train, keys_test, y, fit_idx, val_idx, fold):
    fit_out = pd.DataFrame(index=fit_idx)
    val_out = pd.DataFrame(index=val_idx)
    test_out = pd.DataFrame(index=keys_test.index)
    inner = StratifiedKFold(INNER, shuffle=True, random_state=SEED + fold)
    yf = y[fit_idx]
    for col in keys_train:
        smooth = 60.0 if col.startswith("pair__") else 30.0
        fit_te = np.zeros(len(fit_idx), dtype="float32")
        fit_fr = np.zeros(len(fit_idx), dtype="float32")
        for ia, ib in inner.split(fit_idx, yf):
            te, fr = map_evidence(keys_train.iloc[fit_idx[ia]][col], keys_train.iloc[fit_idx[ib]][col], yf[ia], smooth)
            fit_te[ib], fit_fr[ib] = te, fr
        val_te, val_fr = map_evidence(keys_train.iloc[fit_idx][col], keys_train.iloc[val_idx][col], yf, smooth)
        test_te, test_fr = map_evidence(keys_train.iloc[fit_idx][col], keys_test[col], yf, smooth)
        fit_out[f"te__{col}"], fit_out[f"fr__{col}"] = fit_te, fit_fr
        val_out[f"te__{col}"], val_out[f"fr__{col}"] = val_te, val_fr
        test_out[f"te__{col}"], test_out[f"fr__{col}"] = test_te, test_fr
    return fit_out.reset_index(drop=True), val_out.reset_index(drop=True), test_out.reset_index(drop=True)


train_path, test_path = locate()
train, test = pd.read_csv(train_path), pd.read_csv(test_path)
y = train[TARGET].to_numpy("int8")
base_train, base_test = make_view(train), make_view(test)
keys_train, keys_test = make_keys(train), make_keys(test)
splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train), dtype="float64")
test_preds, fold_rows = [], []
fold_ids = np.zeros(len(train), dtype="int8")
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (6, 0):
    assert "sm_60" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
t0 = time.time()

params = dict(
    arch_type="tabm-mini-normal", tabm_k=24, num_emb_type="pwl", d_embedding=16,
    batch_size=256, lr=7e-4, n_epochs=150, dropout=0.04, d_block=192,
    n_blocks=10, weight_decay=1e-2, verbosity=2,
)

for fold, (fit_idx, val_idx) in enumerate(splitter.split(train, y), 1):
    print(f"fold {fold}: building nested evidence", flush=True)
    ef, ev, et = fold_evidence(keys_train, keys_test, y, fit_idx, val_idx, fold)
    xf = pd.concat([base_train.iloc[fit_idx].reset_index(drop=True), ef], axis=1)
    xv = pd.concat([base_train.iloc[val_idx].reset_index(drop=True), ev], axis=1)
    xt = pd.concat([base_test.reset_index(drop=True), et], axis=1)
    # PyTabKit rejects continuous NaNs. Completion statistics are learned only
    # from this outer-fit partition; explicit missing flags retain the pattern.
    numeric = xf.select_dtypes(include=np.number).columns
    medians = xf[numeric].replace([np.inf, -np.inf], np.nan).median()
    for frame in (xf, xv, xt):
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
    model = TabM_D_Classifier(random_state=SEED + fold, device=device, patience=12, **params)
    model.fit(xf, y[fit_idx], xv, y[val_idx])
    vp = model.predict_proba(xv)[:, 1]
    tp = model.predict_proba(xt)[:, 1]
    oof[val_idx] = vp
    test_preds.append(tp)
    fold_ids[val_idx] = fold
    score = roc_auc_score(y[val_idx], vp)
    fold_rows.append({"fold": fold, "auc": float(score)})
    print(f"fold {fold} auc {score:.9f}", flush=True)
    del model, xf, xv, xt, ef, ev, et
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("tabm_lattice_honest/output")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({ID: train[ID], "fold": fold_ids, "y": y, "pred": oof}).to_csv(out / "oof_tabm_lattice.csv", index=False)
pd.DataFrame({ID: test[ID], TARGET: np.mean(test_preds, axis=0)}).to_csv(out / "test_tabm_lattice.csv", index=False)
scores = [row["auc"] for row in fold_rows]
metrics = {
    "model": "honest nested-evidence TabM-D", "official_data_only": True,
    "concept_credit": ["Omid Baghchehsaraei: TabM for S6E8", "Szymon Klapinski: constrained/lattice feature view"],
    "implementation": "independent outer/inner fold-safe adaptation", "seed": SEED,
    "folds": FOLDS, "inner_folds": INNER, "fold_metrics": fold_rows,
    "fold_mean": float(np.mean(scores)), "fold_std": float(np.std(scores)),
    "oof_auc": float(roc_auc_score(y, oof)), "runtime_seconds": time.time() - t0,
    "submission_created": False,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2), flush=True)
