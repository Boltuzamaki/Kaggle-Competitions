"""Turns a ModelSpec (src/model_zoo.py) into a fittable estimator.

Every builder returns a plain object exposing `.fit(X, y)` and
`.predict_proba(X) -> (n, n_classes)` with columns in ascending label order
(0..n_classes-1), matching src/data.py's target encoding. Estimators that
lack predict_proba natively (RidgeClassifier, Perceptron,
PassiveAggressiveClassifier) are wrapped in `DecisionFunctionSoftmax`.

Each builder pops the hyperparameter keys *it* accepts and silently ignores
irrelevant bookkeeping keys, since presets in model_zoo.py always include
`random_state` even for estimators that don't take it (KNN, NB, LDA, QDA).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class DecisionFunctionSoftmax:
    """Wraps any estimator with .decision_function into one with .predict_proba
    via a softmax over the decision scores. Used for linear models that don't
    natively output calibrated probabilities (Ridge, Perceptron, PA)."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        self.classes_ = self.base_estimator.classes_
        return self

    def predict_proba(self, X):
        scores = self.base_estimator.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)


def _pop(d, *keys):
    return {k: d[k] for k in keys if k in d}


def build_lightgbm(params, device_params, n_classes, n_jobs):
    from lightgbm import LGBMClassifier

    p = dict(params)
    p.update(device_params)
    p.update(objective="multiclass", num_class=n_classes, n_jobs=n_jobs, verbosity=-1)
    return LGBMClassifier(**p)


def build_xgboost(params, device_params, n_classes, n_jobs):
    from xgboost import XGBClassifier

    p = dict(params)
    p.update(device_params)
    p.update(objective="multi:softprob", num_class=n_classes, n_jobs=n_jobs, eval_metric="mlogloss")
    return XGBClassifier(**p)


def build_catboost(params, device_params, n_classes, n_jobs):
    from catboost import CatBoostClassifier

    p = dict(params)
    seed = p.pop("random_state", 42)
    p.update(device_params)
    p.update(loss_function="MultiClass", classes_count=n_classes, random_seed=seed,
              thread_count=n_jobs, verbose=False, allow_writing_files=False)
    return CatBoostClassifier(**p)


def build_hist_gb(params, device_params, n_classes, n_jobs):
    from sklearn.ensemble import HistGradientBoostingClassifier

    p = _pop(params, "max_iter", "max_leaf_nodes", "learning_rate", "l2_regularization",
              "min_samples_leaf", "random_state")
    return HistGradientBoostingClassifier(**p)


def build_random_forest(params, device_params, n_classes, n_jobs):
    from sklearn.ensemble import RandomForestClassifier

    p = _pop(params, "n_estimators", "max_depth", "max_features", "min_samples_leaf", "random_state")
    p["n_jobs"] = n_jobs
    return RandomForestClassifier(**p)


def build_extra_trees(params, device_params, n_classes, n_jobs):
    from sklearn.ensemble import ExtraTreesClassifier

    p = _pop(params, "n_estimators", "max_depth", "max_features", "min_samples_leaf", "random_state")
    p["n_jobs"] = n_jobs
    return ExtraTreesClassifier(**p)


def build_decision_tree(params, device_params, n_classes, n_jobs):
    from sklearn.tree import DecisionTreeClassifier

    p = _pop(params, "max_depth", "criterion", "min_samples_leaf", "random_state")
    return DecisionTreeClassifier(**p)


def build_gradient_boosting(params, device_params, n_classes, n_jobs):
    from sklearn.ensemble import GradientBoostingClassifier

    p = _pop(params, "n_estimators", "max_depth", "learning_rate", "subsample", "random_state")
    return GradientBoostingClassifier(**p)


def build_logistic_regression(params, device_params, n_classes, n_jobs):
    from sklearn.linear_model import LogisticRegression

    p = _pop(params, "C", "penalty", "solver", "l1_ratio", "class_weight", "random_state")
    p.setdefault("max_iter", 300)
    p["n_jobs"] = n_jobs
    return LogisticRegression(**p)


def build_sgd(params, device_params, n_classes, n_jobs):
    from sklearn.linear_model import SGDClassifier

    p = _pop(params, "loss", "alpha", "penalty", "l1_ratio", "random_state")
    p["n_jobs"] = n_jobs
    return SGDClassifier(**p)


def build_ridge(params, device_params, n_classes, n_jobs):
    from sklearn.linear_model import RidgeClassifier

    p = _pop(params, "alpha")
    return DecisionFunctionSoftmax(RidgeClassifier(**p))


def build_knn(params, device_params, n_classes, n_jobs):
    from sklearn.neighbors import KNeighborsClassifier

    p = _pop(params, "n_neighbors", "weights")
    p["n_jobs"] = n_jobs
    return KNeighborsClassifier(**p)


def build_gaussian_nb(params, device_params, n_classes, n_jobs):
    from sklearn.naive_bayes import GaussianNB

    p = _pop(params, "var_smoothing")
    return GaussianNB(**p)


def build_bernoulli_nb(params, device_params, n_classes, n_jobs):
    from sklearn.naive_bayes import BernoulliNB

    p = _pop(params, "alpha")
    return BernoulliNB(**p)


def build_lda(params, device_params, n_classes, n_jobs):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    p = _pop(params, "solver", "shrinkage")
    return LinearDiscriminantAnalysis(**p)


def build_qda(params, device_params, n_classes, n_jobs):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    p = _pop(params, "reg_param")
    return QuadraticDiscriminantAnalysis(**p)


def build_mlp(params, device_params, n_classes, n_jobs):
    from sklearn.neural_network import MLPClassifier

    p = _pop(params, "hidden_layer_sizes", "alpha", "learning_rate_init", "max_iter", "random_state")
    p.setdefault("early_stopping", True)
    p.setdefault("n_iter_no_change", 8)
    return MLPClassifier(**p)


def build_bagging(params, device_params, n_classes, n_jobs):
    from sklearn.ensemble import BaggingClassifier

    p = _pop(params, "n_estimators", "max_samples", "max_features", "random_state")
    p["n_jobs"] = n_jobs
    return BaggingClassifier(**p)


def build_adaboost(params, device_params, n_classes, n_jobs):
    from sklearn.ensemble import AdaBoostClassifier

    p = _pop(params, "n_estimators", "learning_rate", "random_state")
    return AdaBoostClassifier(**p)


def build_perceptron(params, device_params, n_classes, n_jobs):
    from sklearn.linear_model import Perceptron

    p = _pop(params, "alpha", "random_state")
    p["n_jobs"] = n_jobs
    return DecisionFunctionSoftmax(Perceptron(**p))


def build_passive_aggressive(params, device_params, n_classes, n_jobs):
    from sklearn.linear_model import PassiveAggressiveClassifier

    p = _pop(params, "C", "random_state")
    p["n_jobs"] = n_jobs
    return DecisionFunctionSoftmax(PassiveAggressiveClassifier(**p))


def build_tabnet(params, device_params, n_classes, n_jobs):
    from pytorch_tabnet.tab_model import TabNetClassifier

    p = _pop(params, "n_d", "n_a", "n_steps", "gamma", "random_state")
    seed = p.pop("random_state", 42)
    device_name = device_params.get("device_name", "cpu")
    max_epochs = params.get("max_epochs", 20)
    model = TabNetClassifier(n_d=p.get("n_d", 8), n_a=p.get("n_a", 8), n_steps=p.get("n_steps", 3),
                              gamma=p.get("gamma", 1.3), seed=seed, device_name=device_name, verbose=0)
    model._fit_kwargs = {"max_epochs": max_epochs, "patience": 5, "batch_size": 4096, "virtual_batch_size": 512}
    return model


def build_masamlp(params, device_params, n_classes, n_jobs, cat_feature_indices=None):
    """Generic builder for every masamlp-backed family (masamlp_ft_transformer,
    masamlp_realmlp, masamlp_tabr, masamlp_tab_transformer, masamlp_gandalf).
    Presets in model_zoo.py already contain full MasaClassifier kwargs
    (model=..., model_params={...}, n_epochs=..., etc.) -- this just wires in
    the device toggle, categorical column indices, and thread count."""
    from masamlp import MasaClassifier

    p = dict(params)
    seed = p.pop("random_state", 42)
    p.update(device_params)
    p["categorical_features"] = list(cat_feature_indices or [])
    p["random_state"] = seed
    p.setdefault("verbose", 0)
    p.setdefault("n_threads", n_jobs)
    return MasaClassifier(**p)


MASAMLP_FAMILIES = {
    "masamlp_ft_transformer", "masamlp_realmlp", "masamlp_tabr",
    "masamlp_tab_transformer", "masamlp_gandalf",
}


BUILDERS = {
    "lightgbm": build_lightgbm,
    "xgboost": build_xgboost,
    "catboost": build_catboost,
    "hist_gb": build_hist_gb,
    "random_forest": build_random_forest,
    "extra_trees": build_extra_trees,
    "decision_tree": build_decision_tree,
    "gradient_boosting": build_gradient_boosting,
    **{fam: build_masamlp for fam in MASAMLP_FAMILIES},
    "logistic_regression": build_logistic_regression,
    "sgd": build_sgd,
    "ridge": build_ridge,
    "knn": build_knn,
    "gaussian_nb": build_gaussian_nb,
    "bernoulli_nb": build_bernoulli_nb,
    "lda": build_lda,
    "qda": build_qda,
    "mlp": build_mlp,
    "bagging": build_bagging,
    "adaboost": build_adaboost,
    "perceptron": build_perceptron,
    "passive_aggressive": build_passive_aggressive,
    "tabnet": build_tabnet,
}

# families whose builder natively supports the `_device_params` toggle (see
# src/model_zoo.py::_device_params) -- everyone else is CPU-only regardless of
# config.yaml -> device.mode, and that's fine, GPU is opt-in per family.
GPU_CAPABLE_FAMILIES = {"lightgbm", "xgboost", "catboost", "tabnet"} | MASAMLP_FAMILIES

# families that natively accept NaN values (must only be paired with the
# *_nan feature sets; enforced by construction in model_zoo.py presets, this
# set exists as a second line of defense / documentation). masamlp imputes
# and scales numerics internally, so it's NaN-tolerant too.
NAN_NATIVE_FAMILIES = {
    "lightgbm", "xgboost", "catboost", "hist_gb", "random_forest", "extra_trees", "decision_tree",
} | MASAMLP_FAMILIES

# families that need categorical_feature/cat_features passed at fit() time
# rather than baked into the estimator's constructor.
NATIVE_CATEGORICAL_FIT_FAMILIES = {"lightgbm", "catboost"}

# families whose builder needs cat_feature_indices at *construction* time
# (masamlp takes categorical_features as a constructor kwarg, not a fit() arg).
NEEDS_CAT_INDICES_AT_BUILD_FAMILIES = MASAMLP_FAMILIES


def build_estimator(spec, cfg, n_jobs: int, cat_feature_indices=None):
    from src.model_zoo import _device_params as device_params_fn

    builder = BUILDERS[spec.family]
    device_params = device_params_fn(spec.family, cfg) if spec.family in GPU_CAPABLE_FAMILIES else {}
    n_classes = len(cfg["data"]["classes"])
    if spec.family in NEEDS_CAT_INDICES_AT_BUILD_FAMILIES:
        return builder(spec.params, device_params, n_classes, n_jobs, cat_feature_indices)
    return builder(spec.params, device_params, n_classes, n_jobs)


def fit_estimator(estimator, spec, X_tr, y_tr, cat_feature_indices, X_val=None, y_val=None):
    if spec.family == "lightgbm" and cat_feature_indices:
        estimator.fit(X_tr, y_tr, categorical_feature=cat_feature_indices)
    elif spec.family in MASAMLP_FAMILIES:
        fit_kwargs = {}
        if X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        estimator.fit(X_tr, y_tr, **fit_kwargs)
    elif spec.family == "catboost" and cat_feature_indices:
        # CatBoost refuses cat_features on a homogeneous-float ndarray
        # ("'data' is numpy array of floating point numerical type, it means
        # no categorical features") -- it needs per-column dtype identity to
        # tell categorical columns apart, so hand it a DataFrame with those
        # columns cast to int (they're already ordinal-encoded integers,
        # just stored as float32 in the shared cached feature array).
        df_tr = pd.DataFrame(X_tr)
        for idx in cat_feature_indices:
            df_tr[idx] = df_tr[idx].astype(int)
        estimator.fit(df_tr, y_tr, cat_features=cat_feature_indices)
    elif spec.family == "tabnet":
        fit_kwargs = dict(getattr(estimator, "_fit_kwargs", {}))
        if X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        estimator.fit(X_tr, y_tr, **fit_kwargs)
    else:
        estimator.fit(X_tr, y_tr)
    return estimator


def predict_proba_ordered(estimator, X, n_classes, family=None, cat_feature_indices=None):
    """sklearn/xgb/lgbm/catboost all expose classes_ in ascending order and,
    given every fold contains all n_classes labels (huge stratified dataset),
    predict_proba columns already line up with 0..n_classes-1. This is a
    defensive re-index in case a class is ever entirely absent from a fold."""
    if family == "catboost" and cat_feature_indices:
        # a model fit with cat_features remembers it and applies the same
        # dtype validation at predict time -- see fit_estimator's comment.
        df = pd.DataFrame(X)
        for idx in cat_feature_indices:
            df[idx] = df[idx].astype(int)
        X = df
    proba = np.asarray(estimator.predict_proba(X), dtype=np.float32)
    classes = getattr(estimator, "classes_", None)
    if classes is None or list(classes) == list(range(n_classes)):
        return proba
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    for i, c in enumerate(classes):
        out[:, int(c)] = proba[:, i]
    return out
