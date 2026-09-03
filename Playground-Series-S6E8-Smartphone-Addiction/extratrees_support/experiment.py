"""Honest ExtraTrees on target-free support and missingness features.

No boosting and no target encoding are used. Every validation/test support count
is fitted on the corresponding outer training partition; training rows use
leave-one-out counts. Hyperparameter selection is confined to an inner holdout.
"""
from pathlib import Path
import gc, json, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

TARGET, ID, SEED = "addicted_label", "id", 20260804
SMOKE = os.getenv("S6E8_SMOKE", "0") == "1"
FOLDS, INNER_TREES, FINAL_TREES = (2, 35, 45) if SMOKE else (5, 180, 650)
CONFIGS = {
    "wide_leaf2": dict(max_features=.78, min_samples_leaf=2, max_depth=None),
    "regular_leaf5": dict(max_features=.62, min_samples_leaf=5, max_depth=30),
}
if SMOKE: CONFIGS = {"wide_leaf2": CONFIGS["wide_leaf2"]}
PAIRS = [
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("social_media_hours", "gaming_hours"),
    ("notifications_per_day", "app_opens_per_day"),
    ("stress_level", "academic_work_impact"),
]

def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        if not root.exists(): continue
        for p in root.rglob("train.csv"):
            try: cols = pd.read_csv(p, nrows=1).columns
            except Exception: continue
            if TARGET in cols and (p.parent / "test.csv").exists():
                return p, p.parent / "test.csv"
    raise FileNotFoundError("official S6E8 data not found")

def key(s, rounded=False):
    if rounded and pd.api.types.is_numeric_dtype(s): s = pd.to_numeric(s, errors="coerce").round(1)
    return s.astype("string").fillna("__NA__")

def keys(df):
    raw = df.drop(columns=[ID, TARGET], errors="ignore"); out = {}
    for c in raw:
        out["exact__" + c] = key(raw[c])
        if pd.api.types.is_numeric_dtype(raw[c]): out["round1__" + c] = key(raw[c], True)
    for a, b in PAIRS:
        out[f"pair_exact__{a}__{b}"] = key(raw[a]) + "|" + key(raw[b])
        if pd.api.types.is_numeric_dtype(raw[a]) and pd.api.types.is_numeric_dtype(raw[b]):
            out[f"pair_round1__{a}__{b}"] = key(raw[a], True) + "|" + key(raw[b], True)
    return pd.DataFrame(out).reset_index(drop=True)

def base(df):
    raw = df.drop(columns=[ID, TARGET], errors="ignore"); out = pd.DataFrame(index=df.index)
    for c in raw:
        if pd.api.types.is_numeric_dtype(raw[c]):
            out["raw__" + c] = raw[c].astype("float32")
        else:
            # Stable target-free hashing gives identical codes in train/test
            # without fitting a category dictionary on scored rows.
            h = pd.util.hash_array(key(raw[c]).to_numpy(dtype=object))
            out["category__" + c] = (h % np.uint64(104729)).astype("int32")
        out["missing__" + c] = raw[c].isna().astype("int8")
    out["missing_count"] = raw.isna().sum(axis=1).astype("int8")
    comp = raw[["social_media_hours", "gaming_hours", "work_study_hours"]].sum(axis=1, min_count=3)
    out["component_sum"] = comp
    out["screen_residual"] = raw.daily_screen_time_hours - comp
    out["weekend_gap"] = raw.weekend_screen_time - raw.daily_screen_time_hours
    out["leisure_share"] = (raw.social_media_hours + raw.gaming_hours) / (raw.daily_screen_time_hours + .25)
    out["screen_sleep_ratio"] = raw.daily_screen_time_hours / (raw.sleep_hours + .25)
    out["notification_open_ratio"] = raw.notifications_per_day / (raw.app_opens_per_day + 1.)
    return out.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

def support(fit, apply, loo=False):
    out = {}
    for c in fit:
        n = apply[c].map(fit[c].value_counts(dropna=False)).fillna(0).to_numpy("float32")
        if loo: n = np.maximum(n - 1., 0.)
        out[c + "__log_support"] = np.log1p(n)
        out[c + "__novel"] = (n == 0).astype("int8")
    z = pd.DataFrame(out)
    for a, b in PAIRS:
        for tag in ("exact", "round1"):
            p = f"pair_{tag}__{a}__{b}__log_support"; ka = f"{tag}__{a}__log_support"; kb = f"{tag}__{b}__log_support"
            if p in z and ka in z and kb in z:
                pn, an, bn = np.expm1(z[p]), np.expm1(z[ka]), np.expm1(z[kb])
                z[p + "__backoff"] = np.log((pn + 1.) / np.sqrt((an + 1.) * (bn + 1.))).astype("float32")
    return z

def design(bfit, kfit, bapply, kapply, loo=False):
    med = bfit.median()
    return pd.concat([bapply.reset_index(drop=True).fillna(med).fillna(0).astype("float32"),
                      support(kfit, kapply, loo)], axis=1)

def forest(cfg, trees, seed):
    return ExtraTreesClassifier(n_estimators=trees, criterion="entropy", bootstrap=False,
        class_weight=None, n_jobs=-1, random_state=seed, max_samples=None, **cfg)

tp, vp = locate(); tr, te = pd.read_csv(tp), pd.read_csv(vp)
if SMOKE:
    tr = tr.groupby(TARGET, group_keys=False).sample(n=2200, random_state=SEED).reset_index(drop=True)
    te = te.head(1000).reset_index(drop=True)
y = tr[TARGET].to_numpy("int8"); btr, bte, ktr, kte = base(tr), base(te), keys(tr), keys(te)
outer = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
oof, fold_id, test_preds, records = np.zeros(len(tr)), np.zeros(len(tr), "int8"), [], []
t0 = time.time()
for fold, (fit, val) in enumerate(outer.split(btr, y), 1):
    ss = StratifiedShuffleSplit(1, test_size=.16, random_state=SEED + fold)
    ai, av = next(ss.split(fit, y[fit])); fi, fv = fit[ai], fit[av]
    xi = design(btr.iloc[fi], ktr.iloc[fi], btr.iloc[fi], ktr.iloc[fi], True)
    xv = design(btr.iloc[fi], ktr.iloc[fi], btr.iloc[fv], ktr.iloc[fv])
    trials = []
    for name, cfg in CONFIGS.items():
        m = forest(cfg, INNER_TREES, SEED + fold)
        m.fit(xi, y[fi]); p = m.predict_proba(xv)[:, 1]
        trials.append({"config": name, "inner_auc": float(roc_auc_score(y[fv], p))})
        del m; gc.collect()
    winner = max(trials, key=lambda d: d["inner_auc"])
    xf = design(btr.iloc[fit], ktr.iloc[fit], btr.iloc[fit], ktr.iloc[fit], True)
    xv = design(btr.iloc[fit], ktr.iloc[fit], btr.iloc[val], ktr.iloc[val])
    xt = design(btr.iloc[fit], ktr.iloc[fit], bte, kte)
    m = forest(CONFIGS[winner["config"]], FINAL_TREES, SEED + 100 + fold)
    m.fit(xf, y[fit]); pv = m.predict_proba(xv)[:, 1]
    oof[val], fold_id[val] = pv, fold; test_preds.append(m.predict_proba(xt)[:, 1])
    records.append({"fold": fold, "outer_auc": float(roc_auc_score(y[val], pv)),
                    "selected": winner["config"], "inner_trials": trials})
    print(json.dumps(records[-1]), flush=True)
    del xi, xv, xf, xt, m; gc.collect()

out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("extratrees_support/output")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({ID: tr[ID], "fold": fold_id, "y": y, "pred": oof}).to_csv(out / "oof_extratrees_support.csv", index=False)
pd.DataFrame({ID: te[ID], "pred": np.mean(test_preds, axis=0)}).to_csv(out / "test_extratrees_support.csv", index=False)
metrics = {"model": "ExtraTrees target-free hierarchical support", "official_data_only": True,
 "public_predictions_used": False, "target_encoding_used": False, "boosting_used": False,
 "outer_folds": FOLDS, "inner_tuning": "bounded 16% holdout inside outer-fit only",
 "inner_trees": INNER_TREES, "final_trees": FINAL_TREES, "configs": CONFIGS,
 "fold_records": records, "oof_auc": float(roc_auc_score(y, oof)),
 "runtime_seconds": time.time()-t0, "smoke": SMOKE, "submission_created": False}
(out / "metrics_extratrees_support.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2), flush=True)
