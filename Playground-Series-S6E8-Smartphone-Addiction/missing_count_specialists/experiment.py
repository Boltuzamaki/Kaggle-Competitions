"""S6E8 fold-safe missing-count specialists; official competition data only."""
from pathlib import Path
import gc, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID, SEED, NFOLD = "addicted_label", "id", 20260803, 5
BUCKETS = ("0", "1", "2", "3", "4plus")

def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        for p in root.rglob("train.csv"):
            try: cols = pd.read_csv(p, nrows=2).columns
            except Exception: continue
            if TARGET in cols and (p.parent / "test.csv").exists():
                return p, p.parent / "test.csv"
    raise FileNotFoundError("official S6E8 train/test files not found")

def base(df):
    x = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    raw = list(x.columns); nums = list(x.select_dtypes("number").columns)
    miss = x[raw].isna()
    x["missing_count"] = miss.sum(axis=1).astype("int8")
    x["missing_pattern"] = sum(miss[c].astype("int16") * (1 << j) for j, c in enumerate(raw))
    for c in raw: x[c + "__missing"] = miss[c].astype("int8")
    x["leisure_hours"] = x.social_media_hours + x.gaming_hours
    x["accounted_hours"] = x.social_media_hours + x.gaming_hours + x.work_study_hours
    x["unaccounted_screen"] = x.daily_screen_time_hours - x.accounted_hours
    x["weekend_gap"] = x.weekend_screen_time - x.daily_screen_time_hours
    for name, a, b, off in [
        ("weekend_ratio", "weekend_screen_time", "daily_screen_time_hours", .25),
        ("social_share", "social_media_hours", "daily_screen_time_hours", .25),
        ("gaming_share", "gaming_hours", "daily_screen_time_hours", .25),
        ("work_share", "work_study_hours", "daily_screen_time_hours", .25),
        ("notif_open_ratio", "notifications_per_day", "app_opens_per_day", 2),
        ("screen_sleep_ratio", "daily_screen_time_hours", "sleep_hours", .25)]:
        x[name] = x[a] / (x[b] + off)
    return x.replace([np.inf, -np.inf], np.nan), raw

def bucket(v):
    v = np.asarray(v)
    return np.where(v >= 4, "4plus", v.astype(str))

def key(s): return s.astype("string").fillna("__NA__")

def learn_map(s, y, smooth=35.):
    z = pd.DataFrame({"k": key(s).to_numpy(), "y": np.asarray(y)})
    st = z.groupby("k", observed=True).y.agg(["sum", "count"])
    prior = float(np.mean(y))
    return ((st["sum"] + smooth * prior) / (st["count"] + smooth)).to_dict(), st["count"].to_dict(), prior

def encode(fit0, yfit, valid0, test0, cols, fold, salt):
    """Inner-OOF fit encodings and outer-fit-only validation/test mappings."""
    a, b, c = (z.copy().reset_index(drop=True) for z in (fit0, valid0, test0))
    yin = np.asarray(yfit); inner = StratifiedKFold(4, shuffle=True, random_state=SEED + 31 * fold + salt)
    for col in cols:
        oo = np.full(len(a), yin.mean(), dtype="float32")
        for ii, jj in inner.split(np.zeros(len(a)), yin):
            mp, _, pr = learn_map(a.iloc[ii][col], yin[ii])
            oo[jj] = key(a.iloc[jj][col]).map(mp).fillna(pr).to_numpy("float32")
        mp, ct, pr = learn_map(a[col], yin)
        a[col + "__te"] = oo
        b[col + "__te"] = key(b[col]).map(mp).fillna(pr).astype("float32")
        c[col + "__te"] = key(c[col]).map(mp).fillna(pr).astype("float32")
        for frame in (a, b, c):
            frame[col + "__logfreq"] = np.log1p(key(frame[col]).map(ct).fillna(0)).astype("float32")
    # Numeric coding avoids framework-dependent categorical handling and uses no labels.
    for col in a.select_dtypes(exclude="number").columns:
        levels = pd.Index(key(a[col]).unique())
        mapping = {value: j for j, value in enumerate(levels)}
        for frame in (a, b, c): frame[col] = key(frame[col]).map(mapping).fillna(-1).astype("int32")
    return a, b, c

def lgb_model(seed):
    return lgb.LGBMClassifier(objective="binary", n_estimators=2600, learning_rate=.03,
        num_leaves=31, max_depth=-1, min_child_samples=120, subsample=.88,
        subsample_freq=1, colsample_bytree=.90, reg_alpha=.15, reg_lambda=3.,
        random_state=seed, n_jobs=-1, verbosity=-1)

def xgb_model(seed):
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
        tree_method="hist", n_estimators=2600, learning_rate=.03, max_depth=6,
        min_child_weight=12, subsample=.88, colsample_bytree=.90,
        reg_alpha=.12, reg_lambda=3., random_state=seed, n_jobs=-1)

def fit_family(family, a, ya, b, yb, c, seed):
    if family == "lgb":
        model = lgb_model(seed)
        model.fit(a, ya, eval_set=[(b, yb)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(140, verbose=False)])
        iteration = int(model.best_iteration_)
    else:
        model = xgb_model(seed)
        model.fit(a, ya, eval_set=[(b, yb)], verbose=False)
        iteration = int(getattr(model, "best_iteration", 2599)) + 1
    return model.predict_proba(b)[:, 1], model.predict_proba(c)[:, 1], iteration

train_path, test_path = locate()
train, test = pd.read_csv(train_path), pd.read_csv(test_path)
y = train[TARGET].to_numpy("int8"); tr0, raw = base(train); te0, _ = base(test)
train_bucket, test_bucket = bucket(tr0.missing_count), bucket(te0.missing_count)
enc_cols = raw + ["missing_pattern"]
outer = StratifiedKFold(NFOLD, shuffle=True, random_state=SEED)
oof_global = np.zeros(len(train)); oof_lgb = np.zeros(len(train)); oof_xgb = np.zeros(len(train))
test_global, test_lgb, test_xgb, fold_rows = [], [], [], []

for fold, (fi, vi) in enumerate(outer.split(tr0, y), 1):
    # Global comparator uses the same fold-safe representation.
    a, b, c = encode(tr0.iloc[fi], y[fi], tr0.iloc[vi], te0, enc_cols, fold, 0)
    pg, tg, global_it = fit_family("lgb", a, y[fi], b, y[vi], c, SEED + fold)
    oof_global[vi] = pg; test_global.append(tg)
    fold_lgb = np.zeros(len(vi)); fold_xgb = np.zeros(len(vi))
    fold_test_lgb = np.zeros(len(test)); fold_test_xgb = np.zeros(len(test)); bucket_rows = []
    del a, b, c; gc.collect()
    for bucket_id, bucket_name in enumerate(BUCKETS):
        fm, vm, tm = train_bucket[fi] == bucket_name, train_bucket[vi] == bucket_name, test_bucket == bucket_name
        a, b, c = encode(tr0.iloc[fi].loc[fm], y[fi][fm], tr0.iloc[vi].loc[vm], te0.loc[tm], enc_cols, fold, 100 + bucket_id)
        pl, tl, il = fit_family("lgb", a, y[fi][fm], b, y[vi][vm], c, SEED + 1000 + fold * 10 + bucket_id)
        px, tx, ix = fit_family("xgb", a, y[fi][fm], b, y[vi][vm], c, SEED + 2000 + fold * 10 + bucket_id)
        fold_lgb[vm], fold_xgb[vm] = pl, px; fold_test_lgb[tm], fold_test_xgb[tm] = tl, tx
        bucket_rows.append({"bucket": bucket_name, "fit_rows": int(fm.sum()), "validation_rows": int(vm.sum()),
            "lgb_auc": roc_auc_score(y[vi][vm], pl), "xgb_auc": roc_auc_score(y[vi][vm], px),
            "lgb_iteration": il, "xgb_iteration": ix})
        del a, b, c; gc.collect()
    oof_lgb[vi], oof_xgb[vi] = fold_lgb, fold_xgb
    test_lgb.append(fold_test_lgb); test_xgb.append(fold_test_xgb)
    fold_rows.append({"fold": fold, "global_auc": roc_auc_score(y[vi], pg),
        "specialist_lgb_auc": roc_auc_score(y[vi], fold_lgb),
        "specialist_xgb_auc": roc_auc_score(y[vi], fold_xgb),
        "global_iteration": global_it, "buckets": bucket_rows})
    print(json.dumps(fold_rows[-1]), flush=True)

test_g, test_l, test_x = map(lambda z: np.mean(z, axis=0), (test_global, test_lgb, test_xgb))
oof = pd.DataFrame({ID: train[ID], "fold": np.concatenate([np.full(len(v), k) for k, (_, v) in enumerate(outer.split(tr0, y))]) if False else -1,
    "y": y, "pred_global": oof_global, "pred_specialist_lgb": oof_lgb, "pred_specialist_xgb": oof_xgb})
# Recover the exact common fold assignment in ID order.
for fold, (_, vi) in enumerate(StratifiedKFold(NFOLD, shuffle=True, random_state=SEED).split(tr0, y)):
    oof.loc[vi, "fold"] = fold
for w in (.25, .50, .75):
    oof[f"blend_lgb_xgb_{w:.2f}"] = (1 - w) * oof_lgb + w * oof_xgb
oof.to_csv("/kaggle/working/oof_missing_count_specialists.csv", index=False)
pd.DataFrame({ID: test[ID], "pred_global": test_g, "pred_specialist_lgb": test_l,
              "pred_specialist_xgb": test_x}).to_csv("/kaggle/working/test_missing_count_specialists.csv", index=False)

scores = {c: roc_auc_score(y, oof[c]) for c in oof.columns if c.startswith("pred_") or c.startswith("blend_")}
slice_scores = {}
for bucket_name in BUCKETS:
    mask = train_bucket == bucket_name
    slice_scores[bucket_name] = {c: roc_auc_score(y[mask], oof.loc[mask, c]) for c in scores}
metrics = {"provenance": "official competition data only; independently trained; fold-safe nested encodings",
    "folds": NFOLD, "seed": SEED, "buckets": list(BUCKETS), "oof_auc": scores,
    "bucket_oof_auc": slice_scores, "fold_metrics": fold_rows, "submission_created": False}
Path("/kaggle/working/metrics_missing_count_specialists.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2), flush=True)
