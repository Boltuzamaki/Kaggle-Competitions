"""Honest nested-tuned CatBoost on single/pair empirical-Bayes evidence.

For every outer fold, hyperparameters and iteration count are selected on a
holdout contained wholly inside outer-train. The selected model is then refit
on all outer-train rows with inner-OOF evidence. Outer-validation labels are
used only for the final score. Official competition data only.
"""
from pathlib import Path
import gc, json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

TARGET, ID, SEED, FOLDS = "addicted_label", "id", 20260804, 5
SMOOTH_SINGLE, SMOOTH_PAIR = 28.0, 55.0
CONFIGS = {
    "pair_d6": dict(depth=6, learning_rate=.035, l2_leaf_reg=7., random_strength=.25,
                    bagging_temperature=.35),
    "pair_d7": dict(depth=7, learning_rate=.030, l2_leaf_reg=9., random_strength=.35,
                    bagging_temperature=.50),
    "pair_d8": dict(depth=8, learning_rate=.025, l2_leaf_reg=12., random_strength=.45,
                    bagging_temperature=.65),
}
PAIR_SPECS = [
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "notifications_per_day"),
    ("daily_screen_time_hours", "app_opens_per_day"),
    ("social_media_hours", "gaming_hours"),
    ("stress_level", "academic_work_impact"),
]

def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        for p in root.rglob("train.csv"):
            try: cols = pd.read_csv(p, nrows=1).columns
            except Exception: continue
            if TARGET in cols and (p.parent / "test.csv").exists():
                return p, p.parent / "test.csv"
    raise FileNotFoundError("official competition tables not found")

def key(s): return s.astype("string").fillna("__NA__")
def rounded(s): return pd.to_numeric(s, errors="coerce").round(1).astype("string").fillna("__NA__")
def floored(s): return np.floor(pd.to_numeric(s, errors="coerce")).astype("Int64").astype("string").fillna("__NA__")

def views(df):
    raw = df.drop(columns=[ID, TARGET], errors="ignore")
    base = pd.DataFrame(index=df.index)
    for c in raw.select_dtypes(include="number"):
        base["raw__" + c] = raw[c].astype("float32")
        base["missing__" + c] = raw[c].isna().astype("int8")
    base["missing_count"] = raw.isna().sum(axis=1).astype("int8")
    component = raw[["social_media_hours", "gaming_hours", "work_study_hours"]].sum(axis=1, min_count=3)
    base["component_sum"] = component
    base["screen_residual"] = raw.daily_screen_time_hours - component
    base["weekend_gap"] = raw.weekend_screen_time - raw.daily_screen_time_hours
    base["screen_sleep"] = raw.daily_screen_time_hours / (raw.sleep_hours + .25)
    base["notif_open"] = raw.notifications_per_day / (raw.app_opens_per_day + 1.)
    keys = {"single__" + c: key(raw[c]) for c in raw}
    for c in raw.select_dtypes(include="number"):
        keys["single_r1__" + c] = rounded(raw[c])
        keys["single_floor__" + c] = floored(raw[c])
    for a, b in PAIR_SPECS:
        for tag, fn in (("exact", key), ("r1", rounded), ("floor", floored)):
            keys[f"pair_{tag}__{a}__{b}"] = fn(raw[a]) + "|" + fn(raw[b])
    return base.replace([np.inf, -np.inf], np.nan).reset_index(drop=True), pd.DataFrame(keys).reset_index(drop=True)

def mapped(fit_key, apply_key, y, smooth):
    prior = float(np.mean(y))
    z = pd.DataFrame({"k": key(fit_key).to_numpy(), "y": np.asarray(y)})
    st = z.groupby("k", sort=False, observed=True).y.agg(["sum", "count"])
    k = key(apply_key); sums = k.map(st["sum"]).fillna(0).to_numpy(); counts = k.map(st["count"]).fillna(0).to_numpy()
    rate = (sums + smooth * prior) / (counts + smooth)
    return rate.astype("float32"), np.log1p(counts).astype("float32")

def evidence(fit_keys, fit_y, apply_keys, seed, inner_folds=4):
    """Inner-OOF evidence for fit and fit-only mapped evidence for apply."""
    fit_keys, apply_keys, fit_y = fit_keys.reset_index(drop=True), apply_keys.reset_index(drop=True), np.asarray(fit_y)
    a = np.zeros((len(fit_keys), 2 * fit_keys.shape[1]), "float32")
    b = np.zeros((len(apply_keys), a.shape[1]), "float32")
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    splits = list(inner.split(fit_keys, fit_y)); coverage = np.zeros(len(fit_keys), "int8")
    for j, col in enumerate(fit_keys):
        smooth = SMOOTH_PAIR if col.startswith("pair_") else SMOOTH_SINGLE
        for ii, iv in splits:
            a[iv, 2*j], a[iv, 2*j+1] = mapped(fit_keys.iloc[ii][col], fit_keys.iloc[iv][col], fit_y[ii], smooth)
            if j == 0: coverage[iv] += 1
        b[:, 2*j], b[:, 2*j+1] = mapped(fit_keys[col], apply_keys[col], fit_y, smooth)
    assert np.all(coverage == 1)
    names = sum(([c + "__rate", c + "__log_support"] for c in fit_keys), [])
    return pd.DataFrame(a, columns=names), pd.DataFrame(b, columns=names)

def combine(base_fit, base_apply, ev_fit, ev_apply):
    med = base_fit.median()
    return (pd.concat([base_fit.reset_index(drop=True).fillna(med), ev_fit], axis=1),
            pd.concat([base_apply.reset_index(drop=True).fillna(med), ev_apply], axis=1))

def make_model(cfg, seed, iterations, od_wait=None):
    kwargs = dict(iterations=iterations, loss_function="Logloss", eval_metric="AUC",
        random_seed=seed, task_type="GPU", devices="0", bootstrap_type="Bayesian",
        allow_writing_files=False, verbose=False, **cfg)
    if od_wait is not None: kwargs.update(od_type="Iter", od_wait=od_wait)
    return CatBoostClassifier(**kwargs)

tp, sp = locate(); tr, te = pd.read_csv(tp), pd.read_csv(sp); y = tr[TARGET].to_numpy("int8")
btr, ktr = views(tr); bte, kte = views(te)
outer = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
oof, fold_ids, test_preds, records = np.zeros(len(tr)), np.zeros(len(tr), "int8"), [], []
for fold, (fit, val) in enumerate(outer.split(btr, y), 1):
    split = StratifiedShuffleSplit(1, test_size=.20, random_state=SEED + 100 * fold)
    di_rel, dv_rel = next(split.split(fit, y[fit])); di, dv = fit[di_rel], fit[dv_rel]
    edi, edv = evidence(ktr.iloc[di], y[di], ktr.iloc[dv], SEED + 1000 + fold)
    xdi, xdv = combine(btr.iloc[di], btr.iloc[dv], edi, edv)
    candidates = []
    for name, cfg in CONFIGS.items():
        m = make_model(cfg, SEED + 2000 + fold, 2400, 140)
        m.fit(xdi, y[di], eval_set=(xdv, y[dv]), use_best_model=True, verbose=False)
        p = m.predict_proba(xdv)[:, 1]
        candidates.append({"config": name, "inner_auc": float(roc_auc_score(y[dv], p)),
                           "iterations": int(m.get_best_iteration() + 1)})
        del m; gc.collect()
    winner = max(candidates, key=lambda z: z["inner_auc"])
    apply_keys = pd.concat([ktr.iloc[val], kte], ignore_index=True)
    efit, eapply = evidence(ktr.iloc[fit], y[fit], apply_keys, SEED + 3000 + fold)
    eval_, etest = eapply.iloc[:len(val)].reset_index(drop=True), eapply.iloc[len(val):].reset_index(drop=True)
    xfit, xval = combine(btr.iloc[fit], btr.iloc[val], efit, eval_)
    _, xtest = combine(btr.iloc[fit], bte, efit, etest)
    final = make_model(CONFIGS[winner["config"]], SEED + 4000 + fold, winner["iterations"])
    final.fit(xfit, y[fit], verbose=False)
    pv = final.predict_proba(xval)[:, 1]; oof[val] = pv; fold_ids[val] = fold
    test_preds.append(final.predict_proba(xtest)[:, 1])
    records.append({"fold": fold, "outer_auc": float(roc_auc_score(y[val], pv)),
                    "selected_config": winner["config"], "selected_iterations": winner["iterations"],
                    "inner_candidates": candidates})
    print(json.dumps(records[-1]), flush=True)
    del xdi, xdv, xfit, xval, xtest, final; gc.collect()

out = Path("/kaggle/working")
pd.DataFrame({ID: tr[ID], "fold": fold_ids, "y": y, "pred": oof}).to_csv(out / "oof_nested_pair_evidence_catboost.csv", index=False)
pd.DataFrame({ID: te[ID], "pred": np.mean(test_preds, axis=0)}).to_csv(out / "test_nested_pair_evidence_catboost.csv", index=False)
metrics = {"model": "nested-tuned CatBoost single/pair evidence", "official_data_only": True,
 "public_predictions_used": False, "outer_folds": FOLDS, "inner_evidence_folds": 4,
 "tuning": "20% holdout strictly inside each outer-train", "gpu_fit_count": 20,
 "key_count": int(ktr.shape[1]), "configs": CONFIGS, "fold_records": records,
 "oof_auc": float(roc_auc_score(y, oof)), "submission_created": False}
(out / "metrics_nested_pair_evidence_catboost.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2), flush=True)
