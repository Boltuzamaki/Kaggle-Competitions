"""Feature engineering.

Produces a small number of *materialized, fully-numeric* feature sets that
every model in the zoo picks from (see src/model_zoo.py: each ModelSpec
names a feature_set). Everything here is computed once and cached to disk
(artifacts/features/) so 300+ model-training runs never redo this work.

Feature sets produced:
  - raw_nan       : 13 base columns (7 numeric w/ NaN preserved, 6 categorical
                    ordinal-coded, NaN -> its own code). For NaN-native models
                    (LightGBM, XGBoost, CatBoost, HistGradientBoosting).
  - raw_imputed   : same 13 columns, numeric median-imputed + standardized,
                    one-hot categoricals appended. For linear/distance/NN models.
  - fe_heavy_nan  : raw_nan + ~400 engineered columns (numeric NaN preserved
                    in the numeric-derived blocks). For NaN-native models.
  - fe_heavy_imputed: same engineered blocks built on imputed numeric input,
                    everything imputed + standardized. For linear/distance/NN.

Leakage notes:
  - groupby-stat / deviation blocks aggregate only on *numeric* columns
    grouped by *categorical* columns with huge group sizes (100k+ rows each,
    since categoricals are 3-4 levels) -> a single row's contribution to its
    own group aggregate is negligible. No target used, so this is not
    target leakage in any case.
  - target-encoded columns use out-of-fold encoding keyed to the *same fixed
    folds* used for model CV (src/folds.py), so OOF predictions from models
    trained on fe_heavy_* stay leakage-free and comparable/blendable across
    the whole model zoo. Test-set target encoding is fit on the full train set.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import encode_target, load_raw
from src.folds import get_or_create_folds

NUMERIC_COLS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
CATEGORICAL_COLS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
def build_base(train: pd.DataFrame, test: pd.DataFrame):
    """Ordinal-encode categoricals (NaN -> its own code = n_categories).
    Numeric columns are returned untouched (NaN preserved)."""
    train_base = train[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    test_base = test[NUMERIC_COLS + CATEGORICAL_COLS].copy()

    cat_maps = {}
    for c in CATEGORICAL_COLS:
        cats = sorted(train_base[c].dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(cats)}
        missing_code = len(cats)
        cat_maps[c] = {"mapping": mapping, "missing_code": missing_code, "n_categories": len(cats) + 1}
        train_base[c] = train_base[c].map(mapping).fillna(missing_code).astype(np.int32)
        test_base[c] = test_base[c].map(mapping).fillna(missing_code).astype(np.int32)

    return train_base, test_base, cat_maps


# --------------------------------------------------------------------------- #
# variant-independent engineered blocks (missing indicators, freq/target enc, qbins)
# --------------------------------------------------------------------------- #
def _missing_indicators(raw_train: pd.DataFrame, raw_test: pd.DataFrame, cols):
    tr = pd.DataFrame({f"{c}_isna": raw_train[c].isnull().astype(np.int8) for c in cols}, index=raw_train.index)
    te = pd.DataFrame({f"{c}_isna": raw_test[c].isnull().astype(np.int8) for c in cols}, index=raw_test.index)
    return tr, te


def _quantile_bins(raw_train_num: pd.DataFrame, raw_test_num: pd.DataFrame, num_cols, n_bins=10):
    tr_new, te_new = {}, {}
    for c in num_cols:
        vals = raw_train_num[c].dropna().to_numpy()
        edges = np.unique(np.nanquantile(vals, np.linspace(0, 1, n_bins + 1)))[1:-1]
        sentinel = np.nanmin(vals) - 1.0
        tr_new[f"{c}_qbin"] = np.digitize(raw_train_num[c].fillna(sentinel), edges).astype(np.int16)
        te_new[f"{c}_qbin"] = np.digitize(raw_test_num[c].fillna(sentinel), edges).astype(np.int16)
    return pd.DataFrame(tr_new, index=raw_train_num.index), pd.DataFrame(te_new, index=raw_test_num.index)


def _frequency_encoding(train_col: pd.Series, test_col: pd.Series):
    freq = train_col.value_counts(normalize=True)
    tr = train_col.map(freq).fillna(0.0).to_numpy(dtype=np.float32)
    te = test_col.map(freq).fillna(0.0).to_numpy(dtype=np.float32)
    return tr, te


def _pairwise_cat_combo(df: pd.DataFrame, cat_cols):
    combos = {}
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            a, b = cat_cols[i], cat_cols[j]
            combos[f"{a}__x__{b}"] = df[a].astype(str) + "_" + df[b].astype(str)
    return pd.DataFrame(combos, index=df.index)


def _oof_target_encode(train_col: pd.Series, y: np.ndarray, folds: np.ndarray, n_classes: int, smoothing=20.0):
    n = len(train_col)
    encoded = np.zeros((n, n_classes), dtype=np.float32)
    global_rates = np.bincount(y, minlength=n_classes) / n
    for fold in np.unique(folds):
        tr_mask = folds != fold
        val_mask = folds == fold
        cat_tr = train_col.to_numpy()[tr_mask]
        y_tr = y[tr_mask]
        df_tmp = pd.DataFrame({"cat": cat_tr})
        for c in range(n_classes):
            df_tmp["is_c"] = (y_tr == c).astype(np.float64)
            grp = df_tmp.groupby("cat")["is_c"].agg(["mean", "count"])
            smoothed = (grp["mean"] * grp["count"] + global_rates[c] * smoothing) / (grp["count"] + smoothing)
            mapping = smoothed.to_dict()
            val_cats = train_col.to_numpy()[val_mask]
            encoded[val_mask, c] = pd.Series(val_cats).map(mapping).fillna(global_rates[c]).to_numpy()
    return encoded


def _target_encode_fit_transform_test(train_col: pd.Series, y: np.ndarray, test_col: pd.Series, n_classes: int, smoothing=20.0):
    n = len(train_col)
    global_rates = np.bincount(y, minlength=n_classes) / n
    df_tmp = pd.DataFrame({"cat": train_col.to_numpy()})
    out = np.zeros((len(test_col), n_classes), dtype=np.float32)
    for c in range(n_classes):
        df_tmp["is_c"] = (y == c).astype(np.float64)
        grp = df_tmp.groupby("cat")["is_c"].agg(["mean", "count"])
        smoothed = (grp["mean"] * grp["count"] + global_rates[c] * smoothing) / (grp["count"] + smoothing)
        mapping = smoothed.to_dict()
        out[:, c] = test_col.map(mapping).fillna(global_rates[c]).to_numpy()
    return out


def build_variant_independent_blocks(raw_train, raw_test, base_train, base_test, y, folds, n_classes):
    blocks_tr, blocks_te = [], []

    miss_tr, miss_te = _missing_indicators(raw_train, raw_test, NUMERIC_COLS + CATEGORICAL_COLS)
    blocks_tr.append(miss_tr)
    blocks_te.append(miss_te)

    qbin_tr, qbin_te = _quantile_bins(raw_train[NUMERIC_COLS], raw_test[NUMERIC_COLS], NUMERIC_COLS)
    blocks_tr.append(qbin_tr)
    blocks_te.append(qbin_te)

    freq_tr = {}
    freq_te = {}
    for c in CATEGORICAL_COLS:
        t, e = _frequency_encoding(base_train[c], base_test[c])
        freq_tr[f"{c}_freq"] = t
        freq_te[f"{c}_freq"] = e
    combo_train = _pairwise_cat_combo(base_train, CATEGORICAL_COLS)
    combo_test = _pairwise_cat_combo(base_test, CATEGORICAL_COLS)
    for c in combo_train.columns:
        t, e = _frequency_encoding(combo_train[c], combo_test[c])
        freq_tr[f"{c}_freq"] = t
        freq_te[f"{c}_freq"] = e
    blocks_tr.append(pd.DataFrame(freq_tr, index=raw_train.index))
    blocks_te.append(pd.DataFrame(freq_te, index=raw_test.index))

    te_tr = {}
    te_te = {}
    all_cat_sources_train = {c: base_train[c] for c in CATEGORICAL_COLS}
    all_cat_sources_train.update({c: combo_train[c] for c in combo_train.columns})
    all_cat_sources_test = {c: base_test[c] for c in CATEGORICAL_COLS}
    all_cat_sources_test.update({c: combo_test[c] for c in combo_test.columns})
    for name, col_tr in all_cat_sources_train.items():
        col_te = all_cat_sources_test[name]
        enc_tr = _oof_target_encode(col_tr, y, folds, n_classes)
        enc_te = _target_encode_fit_transform_test(col_tr, y, col_te, n_classes)
        for c in range(n_classes):
            te_tr[f"{name}_te_c{c}"] = enc_tr[:, c]
            te_te[f"{name}_te_c{c}"] = enc_te[:, c]
    blocks_tr.append(pd.DataFrame(te_tr, index=raw_train.index))
    blocks_te.append(pd.DataFrame(te_te, index=raw_test.index))

    return pd.concat(blocks_tr, axis=1), pd.concat(blocks_te, axis=1)


# --------------------------------------------------------------------------- #
# numeric-dependent engineered blocks (pair ops, log/square, groupby stats, deviation)
# --------------------------------------------------------------------------- #
def _numeric_pair_features(df: pd.DataFrame, num_cols, eps=1e-3):
    new_cols = {}
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            a, b = num_cols[i], num_cols[j]
            new_cols[f"{a}_div_{b}"] = df[a] / (df[b].abs() + eps)
            new_cols[f"{a}_minus_{b}"] = df[a] - df[b]
            new_cols[f"{a}_times_{b}"] = df[a] * df[b]
    return pd.DataFrame(new_cols, index=df.index)


def _log_and_square(df: pd.DataFrame, num_cols):
    new_cols = {}
    for c in num_cols:
        new_cols[f"{c}_log1p"] = np.log1p(df[c].clip(lower=0))
        new_cols[f"{c}_sq"] = df[c] ** 2
    return pd.DataFrame(new_cols, index=df.index)


def _groupby_stats(num_train: pd.DataFrame, num_test: pd.DataFrame, cat_train: pd.DataFrame, cat_test: pd.DataFrame,
                    cat_cols, num_cols, stats=("mean", "std", "min", "max", "median")):
    tr_new, te_new = {}, {}
    joined_train = pd.concat([cat_train[cat_cols], num_train[num_cols]], axis=1)
    for cat in cat_cols:
        agg = joined_train.groupby(cat)[num_cols].agg(list(stats))
        agg.columns = [f"{cat}__{num}__{stat}" for num, stat in agg.columns]
        mapped_train = cat_train[[cat]].join(agg, on=cat)
        mapped_test = cat_test[[cat]].join(agg, on=cat)
        for col in agg.columns:
            tr_new[col] = mapped_train[col].to_numpy(dtype=np.float32)
            te_new[col] = mapped_test[col].to_numpy(dtype=np.float32)
    return pd.DataFrame(tr_new, index=num_train.index), pd.DataFrame(te_new, index=num_test.index), \
        {cat: [f"{cat}__{num}__mean" for num in num_cols] for cat in cat_cols}


def _deviation_features(num_df: pd.DataFrame, groupby_df: pd.DataFrame, cat_cols, num_cols):
    new = {}
    for cat in cat_cols:
        for num in num_cols:
            mean_col = f"{cat}__{num}__mean"
            if mean_col in groupby_df.columns:
                new[f"{num}_dev_{cat}"] = num_df[num].to_numpy(dtype=np.float32) - groupby_df[mean_col].to_numpy(dtype=np.float32)
    return pd.DataFrame(new, index=num_df.index)


def _row_stats(num_df: pd.DataFrame, miss_df: pd.DataFrame):
    out = pd.DataFrame(index=num_df.index)
    out["row_n_missing"] = miss_df.sum(axis=1).to_numpy(dtype=np.float32)
    out["row_numeric_sum"] = num_df.sum(axis=1, skipna=True).to_numpy(dtype=np.float32)
    out["row_numeric_mean"] = num_df.mean(axis=1, skipna=True).to_numpy(dtype=np.float32)
    out["row_numeric_std"] = num_df.std(axis=1, skipna=True).to_numpy(dtype=np.float32)
    return out


def build_numeric_dependent_blocks(num_train, num_test, cat_train, cat_test, miss_train, miss_test):
    pair_tr = _numeric_pair_features(num_train, NUMERIC_COLS)
    pair_te = _numeric_pair_features(num_test, NUMERIC_COLS)

    logsq_tr = _log_and_square(num_train, NUMERIC_COLS)
    logsq_te = _log_and_square(num_test, NUMERIC_COLS)

    gb_tr, gb_te, _ = _groupby_stats(num_train, num_test, cat_train, cat_test, CATEGORICAL_COLS, NUMERIC_COLS)

    dev_tr = _deviation_features(num_train, gb_tr, CATEGORICAL_COLS, NUMERIC_COLS)
    dev_te = _deviation_features(num_test, gb_te, CATEGORICAL_COLS, NUMERIC_COLS)

    row_tr = _row_stats(num_train, miss_train)
    row_te = _row_stats(num_test, miss_test)

    train_out = pd.concat([pair_tr, logsq_tr, gb_tr, dev_tr, row_tr], axis=1)
    test_out = pd.concat([pair_te, logsq_te, gb_te, dev_te, row_te], axis=1)
    return train_out, test_out


# --------------------------------------------------------------------------- #
# assembly + caching
# --------------------------------------------------------------------------- #
def _cache_paths(cfg, name):
    d = cfg["paths"]["features_dir"]
    return (
        os.path.join(d, f"{name}_train.npy"),
        os.path.join(d, f"{name}_test.npy"),
        os.path.join(d, f"{name}_meta.json"),
    )


def _save_feature_set(cfg, name, train_df: pd.DataFrame, test_df: pd.DataFrame, cat_feature_names=None, needs_scaling=False):
    train_path, test_path, meta_path = _cache_paths(cfg, name)
    train_arr = train_df.to_numpy(dtype=np.float32)
    test_arr = test_df.to_numpy(dtype=np.float32)
    np.save(train_path, train_arr)
    np.save(test_path, test_arr)
    columns = train_df.columns.tolist()
    cat_idx = [columns.index(c) for c in (cat_feature_names or []) if c in columns]
    meta = {
        "columns": columns,
        "n_features": len(columns),
        "cat_feature_names": cat_feature_names or [],
        "cat_feature_indices": cat_idx,
        "needs_scaling": needs_scaling,
        "shape_train": list(train_arr.shape),
        "shape_test": list(test_arr.shape),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def _is_cached(cfg, name):
    train_path, test_path, meta_path = _cache_paths(cfg, name)
    return os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(meta_path)


def load_feature_set(cfg, name, mmap_mode=None):
    """mmap_mode='r' memory-maps the cached .npy instead of loading a private
    copy into RAM. Multiple processes mapping the *same* file share pages via
    the OS page cache instead of each holding their own ~1.7GB copy -- this is
    what makes running several training workers concurrently (src/train.py's
    parallel mode) not blow up total memory usage."""
    train_path, test_path, meta_path = _cache_paths(cfg, name)
    X_train = np.load(train_path, mmap_mode=mmap_mode)
    X_test = np.load(test_path, mmap_mode=mmap_mode)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return X_train, X_test, meta


FEATURE_SET_NAMES = ["raw_nan", "raw_imputed", "fe_heavy_nan", "fe_heavy_imputed"]


def build_all_feature_sets(cfg, force=False):
    """Idempotent: skips any feature set already cached on disk unless force=True."""
    if not force and all(_is_cached(cfg, n) for n in FEATURE_SET_NAMES):
        print("[features] all feature sets already cached, skipping rebuild")
        return

    print("[features] loading raw data...")
    train, test = load_raw(cfg)
    y = encode_target(train, cfg)
    folds = get_or_create_folds(cfg)
    n_classes = len(cfg["data"]["classes"])

    raw_train_num = train[NUMERIC_COLS]
    raw_test_num = test[NUMERIC_COLS]

    print("[features] building base (ordinal categoricals)...")
    base_train, base_test, cat_maps = build_base(train, test)

    # ---- raw_nan --------------------------------------------------------- #
    if force or not _is_cached(cfg, "raw_nan"):
        print("[features] building raw_nan...")
        _save_feature_set(cfg, "raw_nan", base_train, base_test, cat_feature_names=CATEGORICAL_COLS)

    # ---- raw_imputed ------------------------------------------------------ #
    if force or not _is_cached(cfg, "raw_imputed"):
        print("[features] building raw_imputed...")
        imp_train = base_train.copy()
        imp_test = base_test.copy()
        medians = imp_train[NUMERIC_COLS].median()
        imp_train[NUMERIC_COLS] = imp_train[NUMERIC_COLS].fillna(medians)
        imp_test[NUMERIC_COLS] = imp_test[NUMERIC_COLS].fillna(medians)

        ohe_train_parts, ohe_test_parts = [], []
        for c in CATEGORICAL_COLS:
            n_cat = cat_maps[c]["n_categories"]
            oh_tr = np.eye(n_cat, dtype=np.float32)[imp_train[c].to_numpy()]
            oh_te = np.eye(n_cat, dtype=np.float32)[imp_test[c].to_numpy()]
            cols = [f"{c}_ohe_{i}" for i in range(n_cat)]
            ohe_train_parts.append(pd.DataFrame(oh_tr, columns=cols, index=imp_train.index))
            ohe_test_parts.append(pd.DataFrame(oh_te, columns=cols, index=imp_test.index))

        scaler = StandardScaler()
        num_scaled_train = pd.DataFrame(
            scaler.fit_transform(imp_train[NUMERIC_COLS]), columns=NUMERIC_COLS, index=imp_train.index
        )
        num_scaled_test = pd.DataFrame(
            scaler.transform(imp_test[NUMERIC_COLS]), columns=NUMERIC_COLS, index=imp_test.index
        )

        final_train = pd.concat([num_scaled_train] + ohe_train_parts, axis=1)
        final_test = pd.concat([num_scaled_test] + ohe_test_parts, axis=1)
        _save_feature_set(cfg, "raw_imputed", final_train, final_test, needs_scaling=False)

    # ---- shared variant-independent blocks (computed once, reused below) - #
    print("[features] building variant-independent blocks (missing/qbin/freq/target-encoding)...")
    vi_train, vi_test = build_variant_independent_blocks(train, test, base_train, base_test, y, folds, n_classes)
    miss_train = vi_train[[c for c in vi_train.columns if c.endswith("_isna")]]
    miss_test = vi_test[[c for c in vi_test.columns if c.endswith("_isna")]]

    # ---- fe_heavy_nan ------------------------------------------------------ #
    if force or not _is_cached(cfg, "fe_heavy_nan"):
        print("[features] building fe_heavy_nan (numeric-dependent blocks on raw NaN numerics)...")
        nd_train, nd_test = build_numeric_dependent_blocks(
            raw_train_num, raw_test_num, base_train[CATEGORICAL_COLS], base_test[CATEGORICAL_COLS], miss_train, miss_test
        )
        full_train = pd.concat([base_train, vi_train, nd_train], axis=1)
        full_test = pd.concat([base_test, vi_test, nd_test], axis=1)
        _save_feature_set(cfg, "fe_heavy_nan", full_train, full_test, cat_feature_names=CATEGORICAL_COLS)

    # ---- fe_heavy_imputed --------------------------------------------------- #
    if force or not _is_cached(cfg, "fe_heavy_imputed"):
        print("[features] building fe_heavy_imputed (numeric-dependent blocks on imputed numerics)...")
        medians = raw_train_num.median()
        imp_num_train = raw_train_num.fillna(medians)
        imp_num_test = raw_test_num.fillna(medians)
        nd_train, nd_test = build_numeric_dependent_blocks(
            imp_num_train, imp_num_test, base_train[CATEGORICAL_COLS], base_test[CATEGORICAL_COLS], miss_train, miss_test
        )
        imp_base_train = base_train.copy()
        imp_base_test = base_test.copy()
        imp_base_train[NUMERIC_COLS] = imp_num_train
        imp_base_test[NUMERIC_COLS] = imp_num_test

        pre_scale_train = pd.concat([imp_base_train, vi_train, nd_train], axis=1)
        pre_scale_test = pd.concat([imp_base_test, vi_test, nd_test], axis=1)
        pre_scale_train = pre_scale_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pre_scale_test = pre_scale_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        scaler = StandardScaler()
        scaled_train = pd.DataFrame(
            scaler.fit_transform(pre_scale_train), columns=pre_scale_train.columns, index=pre_scale_train.index
        )
        scaled_test = pd.DataFrame(
            scaler.transform(pre_scale_test), columns=pre_scale_test.columns, index=pre_scale_test.index
        )
        _save_feature_set(cfg, "fe_heavy_imputed", scaled_train, scaled_test, needs_scaling=False)

    print("[features] done.")


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    build_all_feature_sets(cfg)
    for name in FEATURE_SET_NAMES:
        X_train, X_test, meta = load_feature_set(cfg, name)
        print(name, "train:", X_train.shape, "test:", X_test.shape, "n_features:", meta["n_features"])
