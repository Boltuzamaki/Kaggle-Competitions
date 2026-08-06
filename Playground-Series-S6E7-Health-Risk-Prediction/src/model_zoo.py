"""The model zoo: 300+ deterministically-generated model specs spanning many
algorithm families, each pinned to one of the 4 cached feature sets
(src/features.py). Every spec has a stable, content-derived `id` so the
training loop (src/train.py) can checkpoint/resume by id regardless of how
many times the process has been restarted or in what order specs are visited.

Design constraints (per user ask):
  - CPU-first: all params default to CPU-safe settings; flipping
    config.yaml -> device.mode: gpu re-derives GPU-flavored params for the
    families that support it (lightgbm, xgboost, catboost, tabnet) the next
    time the registry is generated. Nothing else needs to change.
  - 300+ *distinct* models: reached by combining ~20 algorithm families with
    hand-picked hyperparameter presets (not a blind full grid search - blind
    grids waste compute on near-duplicate configs) x feature-set choice x
    seed, mirroring how the write-up describes building diverse ensembles.
  - Runtime-aware: slow O(n^2)-ish families (kNN, GaussianNB w/ full dim,
    SVM-like linear models, MLP, GradientBoosting) get `subsample_frac` and/or
    reduced feature sets so a CPU run of the whole zoo finishes in a
    reasonable multi-hour, not multi-day, window on 690k rows.
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field

from src.config import gpu_id, is_gpu


@dataclass
class ModelSpec:
    id: str
    family: str
    feature_set: str
    params: dict = field(default_factory=dict)
    subsample_frac: float | None = None
    seed: int = 42

    def to_dict(self):
        return {
            "id": self.id,
            "family": self.family,
            "feature_set": self.feature_set,
            "params": self.params,
            "subsample_frac": self.subsample_frac,
            "seed": self.seed,
        }


def _mk_id(family: str, feature_set: str, params: dict, seed: int) -> str:
    payload = f"{family}|{feature_set}|{sorted(params.items())}|{seed}"
    h = hashlib.md5(payload.encode()).hexdigest()[:8]
    return f"{family}__{feature_set}__seed{seed}__{h}"


def _device_params(family: str, cfg: dict) -> dict:
    gpu = is_gpu(cfg)
    if family == "lightgbm":
        return {"device_type": "gpu" if gpu else "cpu"}
    if family == "xgboost":
        return {"device": "cuda" if gpu else "cpu", "tree_method": "hist"}
    if family == "catboost":
        p = {"task_type": "GPU" if gpu else "CPU"}
        if gpu:
            p["devices"] = str(gpu_id(cfg))
        return p
    if family == "tabnet":
        return {"device_name": "cuda" if gpu else "cpu"}
    if family.startswith("masamlp_"):
        return {"device": "cuda" if gpu else "cpu"}
    return {}


# --------------------------------------------------------------------------- #
# per-family hyperparameter presets
# --------------------------------------------------------------------------- #
_LGB_PRESETS = [
    {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.08, "n_estimators": 400, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "min_child_samples": 40, "reg_alpha": 0.0, "reg_lambda": 0.0},
    {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "n_estimators": 500, "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "min_child_samples": 30, "reg_alpha": 0.1, "reg_lambda": 0.1},
    {"num_leaves": 63, "max_depth": -1, "learning_rate": 0.03, "n_estimators": 700, "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 1, "min_child_samples": 20, "reg_alpha": 0.5, "reg_lambda": 0.5},
    {"num_leaves": 127, "max_depth": -1, "learning_rate": 0.02, "n_estimators": 900, "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 2, "min_child_samples": 15, "reg_alpha": 1.0, "reg_lambda": 1.0},
    {"num_leaves": 24, "max_depth": 5, "learning_rate": 0.1, "n_estimators": 300, "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "min_child_samples": 50, "reg_alpha": 0.0, "reg_lambda": 0.2},
    {"num_leaves": 90, "max_depth": 8, "learning_rate": 0.04, "n_estimators": 600, "feature_fraction": 0.65, "bagging_fraction": 0.75, "bagging_freq": 1, "min_child_samples": 25, "reg_alpha": 0.3, "reg_lambda": 0.3},
    {"num_leaves": 45, "max_depth": 7, "learning_rate": 0.06, "n_estimators": 450, "feature_fraction": 0.75, "bagging_fraction": 0.85, "bagging_freq": 1, "min_child_samples": 35, "reg_alpha": 0.05, "reg_lambda": 0.05},
    {"num_leaves": 255, "max_depth": -1, "learning_rate": 0.015, "n_estimators": 1000, "feature_fraction": 0.45, "bagging_fraction": 0.65, "bagging_freq": 3, "min_child_samples": 10, "reg_alpha": 2.0, "reg_lambda": 2.0},
    {"num_leaves": 20, "max_depth": 3, "learning_rate": 0.12, "n_estimators": 250, "feature_fraction": 0.85, "bagging_fraction": 0.9, "bagging_freq": 1, "min_child_samples": 60, "reg_alpha": 0.0, "reg_lambda": 0.0},
    {"num_leaves": 70, "max_depth": 9, "learning_rate": 0.035, "n_estimators": 650, "feature_fraction": 0.55, "bagging_fraction": 0.7, "bagging_freq": 2, "min_child_samples": 18, "reg_alpha": 0.8, "reg_lambda": 0.4},
]

_XGB_PRESETS = [
    {"max_depth": 3, "eta": 0.1, "n_estimators": 400, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_alpha": 0.0, "reg_lambda": 1.0},
    {"max_depth": 4, "eta": 0.08, "n_estimators": 500, "subsample": 0.75, "colsample_bytree": 0.7, "min_child_weight": 8, "reg_alpha": 0.1, "reg_lambda": 1.5},
    {"max_depth": 5, "eta": 0.05, "n_estimators": 600, "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 10, "reg_alpha": 0.3, "reg_lambda": 2.0},
    {"max_depth": 6, "eta": 0.03, "n_estimators": 800, "subsample": 0.65, "colsample_bytree": 0.5, "min_child_weight": 15, "reg_alpha": 0.5, "reg_lambda": 2.5},
    {"max_depth": 8, "eta": 0.02, "n_estimators": 900, "subsample": 0.6, "colsample_bytree": 0.45, "min_child_weight": 20, "reg_alpha": 1.0, "reg_lambda": 3.0},
    {"max_depth": 2, "eta": 0.15, "n_estimators": 300, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3, "reg_alpha": 0.0, "reg_lambda": 0.5},
    {"max_depth": 10, "eta": 0.015, "n_estimators": 1000, "subsample": 0.55, "colsample_bytree": 0.4, "min_child_weight": 25, "reg_alpha": 2.0, "reg_lambda": 4.0},
    {"max_depth": 4, "eta": 0.1, "n_estimators": 350, "subsample": 0.85, "colsample_bytree": 0.75, "min_child_weight": 6, "reg_alpha": 0.2, "reg_lambda": 1.0},
    {"max_depth": 6, "eta": 0.04, "n_estimators": 700, "subsample": 0.7, "colsample_bytree": 0.55, "min_child_weight": 12, "reg_alpha": 0.6, "reg_lambda": 2.2},
]

_CAT_PRESETS = [
    {"depth": 4, "learning_rate": 0.08, "iterations": 400, "l2_leaf_reg": 3.0, "bagging_temperature": 0.5},
    {"depth": 6, "learning_rate": 0.05, "iterations": 500, "l2_leaf_reg": 5.0, "bagging_temperature": 0.8},
    {"depth": 8, "learning_rate": 0.03, "iterations": 700, "l2_leaf_reg": 8.0, "bagging_temperature": 1.0},
    {"depth": 3, "learning_rate": 0.12, "iterations": 300, "l2_leaf_reg": 2.0, "bagging_temperature": 0.3},
    {"depth": 10, "learning_rate": 0.02, "iterations": 900, "l2_leaf_reg": 10.0, "bagging_temperature": 1.5},
    {"depth": 5, "learning_rate": 0.06, "iterations": 450, "l2_leaf_reg": 4.0, "bagging_temperature": 0.6},
    {"depth": 7, "learning_rate": 0.04, "iterations": 600, "l2_leaf_reg": 6.0, "bagging_temperature": 0.9},
]

_HGB_PRESETS = [
    {"max_iter": 300, "max_leaf_nodes": 31, "learning_rate": 0.08, "l2_regularization": 0.0, "min_samples_leaf": 20},
    {"max_iter": 400, "max_leaf_nodes": 63, "learning_rate": 0.05, "l2_regularization": 0.5, "min_samples_leaf": 30},
    {"max_iter": 500, "max_leaf_nodes": 127, "learning_rate": 0.03, "l2_regularization": 1.0, "min_samples_leaf": 15},
    {"max_iter": 250, "max_leaf_nodes": 15, "learning_rate": 0.1, "l2_regularization": 0.0, "min_samples_leaf": 50},
    {"max_iter": 600, "max_leaf_nodes": 255, "learning_rate": 0.02, "l2_regularization": 2.0, "min_samples_leaf": 10},
]

_RF_PRESETS = [
    {"n_estimators": 200, "max_depth": 10, "max_features": "sqrt", "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": 16, "max_features": "sqrt", "min_samples_leaf": 3},
    {"n_estimators": 250, "max_depth": None, "max_features": "log2", "min_samples_leaf": 8},
    {"n_estimators": 150, "max_depth": 8, "max_features": 0.5, "min_samples_leaf": 10},
    {"n_estimators": 350, "max_depth": 20, "max_features": "sqrt", "min_samples_leaf": 2},
]

_ET_PRESETS = [
    {"n_estimators": 200, "max_depth": 12, "max_features": "sqrt", "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": None, "max_features": "log2", "min_samples_leaf": 3},
    {"n_estimators": 250, "max_depth": 18, "max_features": 0.5, "min_samples_leaf": 6},
    {"n_estimators": 150, "max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 12},
]

_GB_PRESETS = [
    {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.7},
    {"n_estimators": 120, "max_depth": 2, "learning_rate": 0.15, "subsample": 0.9},
]

_LOGREG_PRESETS = [
    {"C": 0.01, "penalty": "l2", "solver": "lbfgs"},
    {"C": 0.1, "penalty": "l2", "solver": "lbfgs"},
    {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
    {"C": 10.0, "penalty": "l2", "solver": "lbfgs"},
    {"C": 1.0, "penalty": "l1", "solver": "saga"},
    {"C": 0.5, "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
    {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
    {"C": 0.1, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
]

_SGD_PRESETS = [
    {"loss": "log_loss", "alpha": 1e-4, "penalty": "l2"},
    {"loss": "log_loss", "alpha": 1e-3, "penalty": "l2"},
    {"loss": "log_loss", "alpha": 1e-5, "penalty": "elasticnet", "l1_ratio": 0.15},
    {"loss": "modified_huber", "alpha": 1e-4, "penalty": "l2"},
    {"loss": "modified_huber", "alpha": 1e-3, "penalty": "l1"},
]

_RIDGE_PRESETS = [
    {"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 100.0},
]

_KNN_PRESETS = [
    {"n_neighbors": 15, "weights": "uniform"},
    {"n_neighbors": 25, "weights": "distance"},
    {"n_neighbors": 50, "weights": "uniform"},
    {"n_neighbors": 75, "weights": "distance"},
]

_NB_GAUSSIAN_PRESETS = [{"var_smoothing": 1e-9}, {"var_smoothing": 1e-7}, {"var_smoothing": 1e-5}]
_NB_BERNOULLI_PRESETS = [{"alpha": 0.5}, {"alpha": 1.0}, {"alpha": 2.0}]

_LDA_PRESETS = [
    {"solver": "svd"},
    {"solver": "lsqr", "shrinkage": "auto"},
    {"solver": "eigen", "shrinkage": "auto"},
]
_QDA_PRESETS = [{"reg_param": 0.0}, {"reg_param": 0.1}, {"reg_param": 0.3}]

_MLP_PRESETS = [
    {"hidden_layer_sizes": (64,), "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 80},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 80},
    {"hidden_layer_sizes": (256, 128, 64), "alpha": 1e-3, "learning_rate_init": 5e-4, "max_iter": 60},
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-5, "learning_rate_init": 2e-3, "max_iter": 100},
]

_BAGGING_PRESETS = [
    {"n_estimators": 30, "max_samples": 0.7, "max_features": 0.8},
    {"n_estimators": 50, "max_samples": 0.5, "max_features": 0.6},
]
_ADABOOST_PRESETS = [
    {"n_estimators": 100, "learning_rate": 1.0},
    {"n_estimators": 200, "learning_rate": 0.5},
    {"n_estimators": 300, "learning_rate": 0.2},
]
_PERCEPTRON_PRESETS = [{"alpha": 1e-4}, {"alpha": 1e-3}]
_PA_PRESETS = [{"C": 0.1}, {"C": 1.0}]
_DT_PRESETS = [
    {"max_depth": 4, "criterion": "gini", "min_samples_leaf": 20},
    {"max_depth": 8, "criterion": "gini", "min_samples_leaf": 10},
    {"max_depth": 12, "criterion": "entropy", "min_samples_leaf": 5},
    {"max_depth": None, "criterion": "gini", "min_samples_leaf": 30},
]
_TABNET_PRESETS = [
    {"n_d": 8, "n_a": 8, "n_steps": 3, "gamma": 1.3, "max_epochs": 25},
    {"n_d": 16, "n_a": 16, "n_steps": 4, "gamma": 1.5, "max_epochs": 20},
    {"n_d": 24, "n_a": 24, "n_steps": 5, "gamma": 1.7, "max_epochs": 15},
]

# masamlp (pip install masamlp) -- a general tabular deep-learning library
# providing sklearn-compatible estimators for several modern architectures.
# Presets below are adapted from two public Kaggle notebooks on this exact
# competition that scored highly with these two architectures specifically
# (nawfeelrahman1124444/ps-s6-ep6-realmlp-*, masayakawamata/s6e7-ft-transformer-v2-*),
# plus a few bonus architectures the library makes essentially free to add
# (tabr, tab_transformer, gandalf) for extra ensemble diversity.
_FTT_PRESETS = [
    # close to the reference notebook's screened config
    {"model_params": {"d_block": 128, "n_blocks": 2, "attention_n_heads": 8, "n_frequencies": 24, "sigma": 0.1},
     "num_embedding": "plr-lite", "numeric_scaler": "quantile", "cat_encoding": "embedding",
     "n_epochs": 20, "batch_size": 4096, "learning_rate": 0.001, "weight_decay": 1e-5,
     "optimizer": "adamw", "lr_scheduler": "cosine", "n_ens": 2},
    # deeper/wider variant
    {"model_params": {"d_block": 192, "n_blocks": 3, "attention_n_heads": 8, "n_frequencies": 32, "sigma": 0.1},
     "num_embedding": "plr-lite", "numeric_scaler": "quantile", "cat_encoding": "embedding",
     "n_epochs": 15, "batch_size": 2048, "learning_rate": 0.0008, "weight_decay": 1e-5,
     "optimizer": "adamw", "lr_scheduler": "cosine", "n_ens": 1},
    # lighter/faster variant, single-net
    {"model_params": {"d_block": 64, "n_blocks": 2, "attention_n_heads": 4, "n_frequencies": 16, "sigma": 0.1},
     "num_embedding": "plr-lite", "numeric_scaler": "quantile", "cat_encoding": "embedding",
     "n_epochs": 25, "batch_size": 4096, "learning_rate": 0.0015, "weight_decay": 1e-5,
     "optimizer": "adamw", "lr_scheduler": "cosine", "n_ens": 1},
]

_REALMLP_PRESETS = [
    # library's own tuned "TD" (top-default) config, epochs trimmed for zoo-scale training
    {"model_params": {"num_scaling": True, "activation": "selu", "use_parametric_act": True,
     "act_lr_factor": 0.1, "dropout": 0.15, "dropout_schedule": "flat_cos", "plr_lr_factor": 0.1,
     "d_num_embedding": 4, "n_frequencies": 16, "sigma": 0.1, "cat_emb_dim": 8},
     "num_embedding": "pbld", "numeric_scaler": "rssc", "cat_encoding": "hybrid",
     "n_epochs": 25, "batch_size": 512, "optimizer": "adamw", "optimizer_betas": (0.9, 0.95),
     "lr_scheduler": "coslog4", "learning_rate": 0.04, "weight_decay": 0.02,
     "weight_decay_schedule": "flat_cos", "label_smoothing": 0.1, "n_ens": 2},
    # library's simpler non-TD default
    {"numeric_scaler": "rssc", "cat_encoding": "onehot",
     "n_epochs": 40, "batch_size": 256, "optimizer": "adam", "optimizer_betas": (0.9, 0.95),
     "lr_scheduler": "coslog4", "learning_rate": 0.04, "label_smoothing": 0.1, "n_ens": 1},
    # bigger ensemble-in-one-model, fewer epochs
    {"model_params": {"num_scaling": True, "activation": "selu", "dropout": 0.1,
     "d_num_embedding": 4, "n_frequencies": 16, "sigma": 0.1, "cat_emb_dim": 8},
     "num_embedding": "pbld", "numeric_scaler": "rssc", "cat_encoding": "hybrid",
     "n_epochs": 18, "batch_size": 512, "optimizer": "adamw", "lr_scheduler": "coslog4",
     "learning_rate": 0.03, "weight_decay": 0.02, "n_ens": 4},
]

_TABR_PRESETS = [
    {"n_epochs": 20, "batch_size": 1024, "learning_rate": 0.001, "n_ens": 1},
    {"n_epochs": 30, "batch_size": 2048, "learning_rate": 0.0015, "n_ens": 2},
]

_TAB_TRANSFORMER_PRESETS = [
    {"n_epochs": 20, "batch_size": 2048, "learning_rate": 0.001, "n_ens": 1},
    {"n_epochs": 25, "batch_size": 4096, "learning_rate": 0.0015, "n_ens": 1},
]

_GANDALF_PRESETS = [
    {"n_epochs": 25, "batch_size": 2048, "learning_rate": 0.002, "n_ens": 1},
    {"n_epochs": 20, "batch_size": 4096, "learning_rate": 0.0025, "n_ens": 2},
]

MASAMLP_MODEL_TAG = {
    "masamlp_ft_transformer": "ft_transformer",
    "masamlp_realmlp": "realmlp",
    "masamlp_tabr": "tabr",
    "masamlp_tab_transformer": "tab_transformer",
    "masamlp_gandalf": "gandalf",
}

SEEDS = [42, 202, 7777]

_NAN_TOLERANT_SETS = ["raw_nan", "fe_heavy_nan"]
_IMPUTED_SETS = ["raw_imputed", "fe_heavy_imputed"]


def _expand(family, presets, feature_sets, seeds, extra_static=None, subsample_frac=None):
    specs = []
    for preset, fset, seed in itertools.product(presets, feature_sets, seeds):
        params = dict(preset)
        if extra_static:
            params.update(extra_static)
        params["random_state"] = seed
        spec_id = _mk_id(family, fset, params, seed)
        specs.append(ModelSpec(id=spec_id, family=family, feature_set=fset, params=params, seed=seed, subsample_frac=subsample_frac))
    return specs


def build_registry(cfg: dict) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    specs += _expand("lightgbm", _LGB_PRESETS, _NAN_TOLERANT_SETS, SEEDS)
    specs += _expand("xgboost", _XGB_PRESETS, _NAN_TOLERANT_SETS, SEEDS)
    specs += _expand("catboost", _CAT_PRESETS, _NAN_TOLERANT_SETS, SEEDS)
    specs += _expand("hist_gb", _HGB_PRESETS, _NAN_TOLERANT_SETS, SEEDS)
    specs += _expand("random_forest", _RF_PRESETS, _NAN_TOLERANT_SETS[:1] + _IMPUTED_SETS[:1], SEEDS)
    specs += _expand("extra_trees", _ET_PRESETS, _NAN_TOLERANT_SETS[:1] + _IMPUTED_SETS[:1], SEEDS)
    specs += _expand("decision_tree", _DT_PRESETS, _NAN_TOLERANT_SETS, SEEDS)
    specs += _expand("gradient_boosting", _GB_PRESETS, _IMPUTED_SETS, SEEDS[:2], subsample_frac=0.35)
    specs += _expand("logistic_regression", _LOGREG_PRESETS, _IMPUTED_SETS, SEEDS[:2])
    specs += _expand("sgd", _SGD_PRESETS, _IMPUTED_SETS, SEEDS[:2])
    specs += _expand("ridge", _RIDGE_PRESETS, _IMPUTED_SETS, SEEDS[:2])
    specs += _expand("knn", _KNN_PRESETS, ["raw_imputed"], SEEDS[:2], subsample_frac=0.08)
    specs += _expand("gaussian_nb", _NB_GAUSSIAN_PRESETS, _IMPUTED_SETS, SEEDS[:1])
    specs += _expand("bernoulli_nb", _NB_BERNOULLI_PRESETS, _IMPUTED_SETS, SEEDS[:1])
    specs += _expand("lda", _LDA_PRESETS, _IMPUTED_SETS, SEEDS[:1])
    specs += _expand("qda", _QDA_PRESETS, _IMPUTED_SETS, SEEDS[:1])
    specs += _expand("mlp", _MLP_PRESETS, _IMPUTED_SETS, SEEDS[:2])
    specs += _expand("bagging", _BAGGING_PRESETS, _IMPUTED_SETS, SEEDS[:2], subsample_frac=0.5)
    specs += _expand("adaboost", _ADABOOST_PRESETS, _IMPUTED_SETS, SEEDS[:2])
    specs += _expand("perceptron", _PERCEPTRON_PRESETS, _IMPUTED_SETS, SEEDS[:2])
    specs += _expand("passive_aggressive", _PA_PRESETS, _IMPUTED_SETS, SEEDS[:2])
    specs += _expand("tabnet", _TABNET_PRESETS, _IMPUTED_SETS[:1], SEEDS[:2], subsample_frac=0.5)

    for family, presets, fsets in [
        ("masamlp_ft_transformer", _FTT_PRESETS, _NAN_TOLERANT_SETS),
        ("masamlp_realmlp", _REALMLP_PRESETS, _NAN_TOLERANT_SETS),
        ("masamlp_tabr", _TABR_PRESETS, ["raw_nan"]),
        ("masamlp_tab_transformer", _TAB_TRANSFORMER_PRESETS, ["raw_nan"]),
        ("masamlp_gandalf", _GANDALF_PRESETS, ["raw_nan"]),
    ]:
        specs += _expand(family, presets, fsets, SEEDS[:2], extra_static={"model": MASAMLP_MODEL_TAG[family]})

    seen = set()
    for s in specs:
        assert s.id not in seen, f"duplicate model id generated: {s.id}"
        seen.add(s.id)

    return specs


if __name__ == "__main__":
    from collections import Counter

    from src.config import load_config

    cfg = load_config()
    registry = build_registry(cfg)
    print("total models:", len(registry))
    fam_counts = Counter(s.family for s in registry)
    for fam, n in sorted(fam_counts.items(), key=lambda x: -x[1]):
        print(f"  {fam:22s} {n}")
    fset_counts = Counter(s.feature_set for s in registry)
    print("by feature set:", dict(fset_counts))
