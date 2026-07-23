# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Step 10 - tiny combined poison-ranker gate
#
# Public-validation only. This notebook consumes the exact promoted Step 8B
# 140-sample bank. It combines four predeclared, non-duplicated signal families:
# renderer physics, PCGrad collapse, V12 survivor collapse, and V10 ensemble
# collapse. No competition test path is mounted, enumerated, or read. It cannot
# construct a test candidate or a submission.

# %%
import hashlib
import json
import math
import os
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(os.environ.get("STEP10_OUT", "/kaggle/working/step10_combined_ranker"))
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.jsonl"
SEED = 230724

EXPECTED = {
    "sample_manifest.json": "f2906451a2b6326bf9c51f4d85f1eda175a86a2e11dadd3d32fffbfb977284b7",
    "public_probe_table.csv": "36f14f3f4e667378b9056df6de8593c17e106b25404c76007d7adc99be7a21c8",
    "renderer_feature_table.csv": "d4fa6b1b062f78bfef8dc40fd62407a4c82f691c7aeafa516cc4938b353b63a8",
    "checkpoint_manifest.json": "5f8d583e1d3803326fb714163df1e804341b3dc3f537a18bb3a1b253a3dd65a2",
}

RENDERER_FEATURES = [
    "positive_fraction", "energy_density", "anisotropy", "straightness",
    "endpoint_ratio", "endpoint_asymmetry", "center_ratio", "longitudinal_cv",
    "longitudinal_total_variation", "gap_fraction", "fft_periodicity",
    "autocorrelation_peak", "gap_run_cv", "width_mean", "width_cv",
    "width_drift", "transverse_symmetry", "gaussian_psf_error",
    "inside_noise_cv", "side_positive_rate", "side_noise_std", "unique_ratio",
    "bit1_entropy", "bit2_entropy", "bit4_entropy", "bit8_entropy",
    "subpixel_phase", "endpoint_sharpness", "interpolation_fraction",
    "axis_residual", "log_length", "log_aspect",
]

# One representative per correlated checkpoint family. V9 is deliberately
# excluded because its public signal correlates 0.994 with V12. NDR229 is
# deliberately excluded because it correlates 0.913 with the V10 family.
COMBINED_FEATURES = ["renderer_physics", "pcgrad_collapse", "v12_collapse", "v10_collapse"]
CLEAN_FAMILIES = [
    "external_crop_transplant",
    "synthetic_irregular_dash",
    "synthetic_stochastic_blink",
    "synthetic_head_tail_tracklet",
]

GATES = {
    "aggregate_auc_min": 0.90,
    "each_clean_family_auc_min": 0.80,
    "median_calibrated_poison_recall_min": 0.50,
    "maximum_clean_family_fpr": 0.10,
    "calibrated_precision_min": 0.90,
    "calibrated_poison_support_min": 5,
    "coefficient_cosine_median_min": 0.50,
    "full_vs_best_single_auc_tolerance": 0.02,
    "nonrenderer_single_family_auc_min": 0.65,
    "nonrenderer_directional_families_min": 2,
    "permutation_repeats": 40,
    "permutation_auc_p95_max": 0.70,
    "clean_calibration_quantile": 0.99,
    "renderer_l2": 0.20,
    "combined_l2": 0.50,
    "fit_steps": 700,
    "inner_folds": 5,
}

ABLATIONS = {
    "full": COMBINED_FEATURES,
    "renderer_only": ["renderer_physics"],
    "pcgrad_only": ["pcgrad_collapse"],
    "v12_only": ["v12_collapse"],
    "v10_only": ["v10_collapse"],
    "without_renderer": ["pcgrad_collapse", "v12_collapse", "v10_collapse"],
    "without_pcgrad": ["renderer_physics", "v12_collapse", "v10_collapse"],
    "without_v12": ["renderer_physics", "pcgrad_collapse", "v10_collapse"],
    "without_v10": ["renderer_physics", "pcgrad_collapse", "v12_collapse"],
}

LOCK = {
    "status": "frozen_before_input_enumeration_or_table_read",
    "experiment": "STEP10_TINY_COMBINED_POISON_RANKER",
    "seed": SEED,
    "exact_step8b_artifact_hashes": EXPECTED,
    "renderer_features": RENDERER_FEATURES,
    "combined_features": COMBINED_FEATURES,
    "excluded_correlated_duplicates": {
        "v9_adversarial": "excluded in favor of V12; public correlation 0.994",
        "ndr229": "excluded in favor of V10 mean; public correlation 0.913",
    },
    "clean_families": CLEAN_FAMILIES,
    "validation": (
        "outer product of leave-one-public-poison-image-out and "
        "leave-one-clean-family-out; inner five-fold renderer cross-fitting"
    ),
    "ablations": ABLATIONS,
    "gates": GATES,
    "selection_boundary": {
        "competition_source_mounted": False,
        "competition_test_enumerated": False,
        "competition_test_read": False,
        "test_labels_or_pseudo_labels": False,
        "test_derived_parameters": False,
        "leaderboard_derived_thresholds": False,
        "candidate_created": False,
        "competition_submission_created": False,
    },
}
(OUT / "selection_lock.json").write_text(json.dumps(LOCK, indent=2), encoding="utf-8")


def log(message, **kwargs):
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "message": message, **kwargs}
    print(json.dumps(row, default=str), flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_step8b_source():
    explicit = os.environ.get("STEP8B_SOURCE")
    if explicit:
        return Path(explicit)
    matches = []
    for path in Path("/kaggle/input").rglob("sample_manifest.json"):
        if sha256(path) == EXPECTED["sample_manifest.json"]:
            root = path.parent
            if all((root / name).exists() for name in EXPECTED):
                matches.append(root)
    assert len(matches) == 1, {"expected_one_step8b_source": [str(p) for p in matches]}
    return matches[0]


SOURCE = find_step8b_source()
artifact_audit = {}
for name, expected in EXPECTED.items():
    actual = sha256(SOURCE / name)
    artifact_audit[name] = {"path": str(SOURCE / name), "expected": expected, "actual": actual, "match": actual == expected}
    assert actual == expected, artifact_audit[name]

renderer_gate = json.loads((SOURCE / "renderer_gate.json").read_text(encoding="utf-8"))
assert renderer_gate["gate_passed"] is True
assert renderer_gate["rule_7a_guard_passed"] is True
assert renderer_gate["competition_test_enumerated"] is False
assert renderer_gate["competition_test_read"] is False
assert renderer_gate["competition_submission_created"] is False
(OUT / "input_artifact_audit.json").write_text(json.dumps(artifact_audit, indent=2), encoding="utf-8")
log("INPUT_ARTIFACTS_VALIDATED", source=str(SOURCE), count=len(artifact_audit))

# %%
probes = pd.read_csv(SOURCE / "public_probe_table.csv")
renderer = pd.read_csv(SOURCE / "renderer_feature_table.csv")
manifest = json.loads((SOURCE / "sample_manifest.json").read_text(encoding="utf-8"))["samples"]

assert len(probes) == len(renderer) == len(manifest) == 140
assert probes["sample_id"].is_unique and renderer["sample_id"].is_unique
manifest_ids = [row["sample_id"] for row in manifest]
assert probes["sample_id"].tolist() == manifest_ids
assert renderer["sample_id"].tolist() == manifest_ids
assert set(probes["family"]) == {"public_poison", *CLEAN_FAMILIES}
assert probes.groupby("family").size().to_dict() == {
    "public_poison": 20,
    **{family: 30 for family in CLEAN_FAMILIES},
}

table = probes[["sample_id", "family", "label_poison"]].merge(
    renderer[["sample_id", *RENDERER_FEATURES]], on="sample_id", validate="one_to_one"
)
table["pcgrad_collapse"] = 1.0 - np.clip(probes["ratio_pcgrad_median"].to_numpy(float), 0.0, 2.0)
table["v12_collapse"] = 1.0 - np.clip(probes["ratio_v12"].to_numpy(float), 0.0, 2.0)
table["v10_collapse"] = 1.0 - np.clip(probes["ratio_v10_mean"].to_numpy(float), 0.0, 2.0)
table.to_csv(OUT / "combined_feature_table.csv", index=False)
log("EXACT_BANK_VALIDATED", rows=len(table), poison=int(table.label_poison.sum()), clean=int((1-table.label_poison).sum()))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def fit_logistic(x, y, l2, steps, seed_offset=0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    center = np.median(x, axis=0)
    scale = np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)
    scale = np.where(scale > 1e-7, scale, np.std(x, axis=0))
    scale = np.where(scale > 1e-7, scale, 1.0)
    z = np.clip((x - center) / scale, -12.0, 12.0)
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = float(np.log((y.mean() + 1e-3) / (1.0 - y.mean() + 1e-3)))
    pos_weight = len(y) / max(2.0 * y.sum(), 1.0)
    neg_weight = len(y) / max(2.0 * (len(y) - y.sum()), 1.0)
    sample_weight = np.where(y > 0.5, pos_weight, neg_weight)
    for step in range(steps):
        pred = sigmoid(z @ weights + bias)
        err = (pred - y) * sample_weight
        lr = 0.08 / math.sqrt(1.0 + step / 80.0)
        weights -= lr * ((z.T @ err) / len(y) + l2 * weights / len(y))
        bias -= lr * float(err.mean())
    return center, scale, weights, bias


def predict_logistic(model, x):
    center, scale, weights, bias = model
    z = np.clip((np.asarray(x, dtype=np.float64) - center) / scale, -12.0, 12.0)
    return sigmoid(z @ weights + bias)


def auc_score(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = score[y == 1]
    neg = score[y == 0]
    assert len(pos) and len(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def fold_assignments(indices, labels, families, nfold=5):
    folds = np.zeros(len(indices), dtype=np.int64)
    counters = {}
    for local, global_idx in enumerate(indices):
        key = "poison" if labels[global_idx] == 1 else str(families[global_idx])
        count = counters.get(key, 0)
        folds[local] = count % nfold
        counters[key] = count + 1
    return folds


X_renderer = table[RENDERER_FEATURES].to_numpy(np.float64)
y = table["label_poison"].to_numpy(np.int64)
families = table["family"].astype(str).to_numpy()
sample_ids = table["sample_id"].astype(str).to_numpy()
base = table[["pcgrad_collapse", "v12_collapse", "v10_collapse"]].to_numpy(np.float64)
poison_indices = np.where(y == 1)[0]

# Renderer cross-fitting is identical for every combined-feature ablation, so
# compute each of the 80 outer episodes once and reuse it. This is a compute
# cache only; no labels, folds, thresholds, or models change between ablations.
EPISODE_CACHE = {}


def outer_episode_features(held_family, held_poison):
    key = (held_family, int(held_poison))
    if key in EPISODE_CACHE:
        return EPISODE_CACHE[key]
    clean_test = np.where(families == held_family)[0]
    test_idx = np.concatenate([[held_poison], clean_test])
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    inner_fold = fold_assignments(train_idx, y, families, GATES["inner_folds"])
    renderer_train_oof = np.zeros(len(train_idx), dtype=np.float64)
    for fold in range(GATES["inner_folds"]):
        fit_local = inner_fold != fold
        val_local = inner_fold == fold
        rmodel = fit_logistic(
            X_renderer[train_idx[fit_local]], y[train_idx[fit_local]],
            GATES["renderer_l2"], GATES["fit_steps"], fold,
        )
        renderer_train_oof[val_local] = predict_logistic(rmodel, X_renderer[train_idx[val_local]])
    rmodel_full = fit_logistic(
        X_renderer[train_idx], y[train_idx], GATES["renderer_l2"], GATES["fit_steps"], 99
    )
    renderer_test = predict_logistic(rmodel_full, X_renderer[test_idx])
    x_train_all = np.column_stack([renderer_train_oof, base[train_idx]])
    x_test_all = np.column_stack([renderer_test, base[test_idx]])
    EPISODE_CACHE[key] = (train_idx, test_idx, x_train_all, x_test_all)
    return EPISODE_CACHE[key]


def nested_episode(feature_names):
    feature_indices = [COMBINED_FEATURES.index(name) for name in feature_names]
    records, coefficients = [], []
    for held_family in CLEAN_FAMILIES:
        for held_poison in poison_indices:
            train_idx, test_idx, x_train_all, x_test_all = outer_episode_features(held_family, held_poison)
            cmodel = fit_logistic(
                x_train_all[:, feature_indices], y[train_idx],
                GATES["combined_l2"], GATES["fit_steps"], 199,
            )
            train_score = predict_logistic(cmodel, x_train_all[:, feature_indices])
            test_score = predict_logistic(cmodel, x_test_all[:, feature_indices])
            clean_train_score = train_score[y[train_idx] == 0]
            threshold = float(np.quantile(clean_train_score, GATES["clean_calibration_quantile"]))
            full_coef = np.zeros(len(COMBINED_FEATURES), dtype=np.float64)
            full_coef[feature_indices] = cmodel[2]
            coefficients.append(full_coef)
            for idx, score in zip(test_idx, test_score):
                records.append({
                    "heldout_clean_family": held_family,
                    "heldout_poison_id": sample_ids[held_poison],
                    "sample_id": sample_ids[idx],
                    "family": families[idx],
                    "label_poison": int(y[idx]),
                    "score": float(score),
                    "threshold": threshold,
                    "predicted_poison": bool(score >= threshold),
                })
    return pd.DataFrame(records), np.asarray(coefficients)


def collapse_repeated_predictions(predictions):
    return predictions.groupby(["sample_id", "family", "label_poison"], as_index=False).agg(
        score=("score", "mean"),
        predicted_poison_rate=("predicted_poison", "mean"),
        threshold=("threshold", "mean"),
    )


def summarize_episode(predictions, coefficients):
    agg = collapse_repeated_predictions(predictions)
    result = {"aggregate_auc": auc_score(agg.label_poison, agg.score), "family_results": {}}
    poison = agg[agg.label_poison == 1]
    for family in CLEAN_FAMILIES:
        clean = agg[agg.family == family]
        pair = pd.concat([poison, clean], ignore_index=True)
        raw = predictions[predictions.heldout_clean_family == family]
        poison_raw = raw[raw.label_poison == 1]
        clean_raw = raw[raw.label_poison == 0]
        result["family_results"][family] = {
            "auc": auc_score(pair.label_poison, pair.score),
            "poison_recall": float(poison_raw.predicted_poison.mean()),
            "clean_false_positive_rate": float(clean_raw.predicted_poison.mean()),
        }
    result["median_calibrated_poison_recall"] = float(np.median([v["poison_recall"] for v in result["family_results"].values()]))
    result["maximum_clean_family_fpr"] = float(max(v["clean_false_positive_rate"] for v in result["family_results"].values()))
    pred_pos = predictions[predictions.predicted_poison]
    result["calibrated_precision"] = float(pred_pos.label_poison.mean()) if len(pred_pos) else 0.0
    result["calibrated_unique_poison_support"] = int(pred_pos.loc[pred_pos.label_poison == 1, "sample_id"].nunique())

    norms = np.linalg.norm(coefficients, axis=1)
    unit = coefficients / np.maximum(norms[:, None], 1e-12)
    cosine = [float(unit[i] @ unit[j]) for i, j in combinations(range(len(unit)), 2)]
    result["coefficient_cosine_median"] = float(np.median(cosine))
    result["coefficient_median"] = {name: float(np.median(coefficients[:, i])) for i, name in enumerate(COMBINED_FEATURES)}
    return result, agg


# %%
all_results = {}
all_predictions = {}
all_coefficients = {}
for name, features in ABLATIONS.items():
    log("ABLATION_START", ablation=name, features=features)
    predictions, coefficients = nested_episode(features)
    result, aggregated = summarize_episode(predictions, coefficients)
    result["features"] = features
    all_results[name] = result
    all_predictions[name] = predictions
    all_coefficients[name] = coefficients
    predictions.to_csv(OUT / f"nested_predictions_{name}.csv", index=False)
    aggregated.to_csv(OUT / f"sample_scores_{name}.csv", index=False)
    log("ABLATION_END", ablation=name, aggregate_auc=result["aggregate_auc"], max_fpr=result["maximum_clean_family_fpr"])

(OUT / "ablation_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
np.savez_compressed(OUT / "combined_fold_coefficients.npz", **all_coefficients)

# %% [markdown]
# ## Label-permutation negative control
#
# The negative control shuffles labels before every grouped five-fold fit and
# uses all 32 physical features plus the three scalar checkpoint-family signals.
# It verifies the model cannot recover a stable target when label identity is
# destroyed. The original labels are never used for fitting these controls.

# %%
rng = np.random.default_rng(SEED + 1000)
raw_control_x = np.column_stack([X_renderer, base])
perm_aucs = []
base_folds = fold_assignments(np.arange(len(y)), y, families, GATES["inner_folds"])
for repeat in range(GATES["permutation_repeats"]):
    perm_y = rng.permutation(y)
    pred = np.zeros(len(y), dtype=np.float64)
    for fold in range(GATES["inner_folds"]):
        train = base_folds != fold
        valid = base_folds == fold
        model = fit_logistic(raw_control_x[train], perm_y[train], 1.0, GATES["fit_steps"], repeat * 10 + fold)
        pred[valid] = predict_logistic(model, raw_control_x[valid])
    perm_aucs.append(auc_score(perm_y, pred))

permutation_control = {
    "repeats": GATES["permutation_repeats"],
    "auc_mean": float(np.mean(perm_aucs)),
    "auc_median": float(np.median(perm_aucs)),
    "auc_p95": float(np.quantile(perm_aucs, 0.95)),
    "auc_max": float(np.max(perm_aucs)),
    "gate_p95_max": GATES["permutation_auc_p95_max"],
    "passed": bool(np.quantile(perm_aucs, 0.95) <= GATES["permutation_auc_p95_max"]),
    "aucs": perm_aucs,
}
(OUT / "permutation_control.json").write_text(json.dumps(permutation_control, indent=2), encoding="utf-8")

# %%
full = all_results["full"]
best_single = max(all_results[name]["aggregate_auc"] for name in ["renderer_only", "pcgrad_only", "v12_only", "v10_only"])
nonrenderer_directional = sum(
    all_results[name]["aggregate_auc"] >= GATES["nonrenderer_single_family_auc_min"]
    for name in ["pcgrad_only", "v12_only", "v10_only"]
)

checks = {
    "aggregate_auc": full["aggregate_auc"] >= GATES["aggregate_auc_min"],
    "each_clean_family_auc": all(v["auc"] >= GATES["each_clean_family_auc_min"] for v in full["family_results"].values()),
    "median_calibrated_poison_recall": full["median_calibrated_poison_recall"] >= GATES["median_calibrated_poison_recall_min"],
    "maximum_clean_family_fpr": full["maximum_clean_family_fpr"] <= GATES["maximum_clean_family_fpr"],
    "calibrated_precision": full["calibrated_precision"] >= GATES["calibrated_precision_min"],
    "calibrated_poison_support": full["calibrated_unique_poison_support"] >= GATES["calibrated_poison_support_min"],
    "coefficient_stability": full["coefficient_cosine_median"] >= GATES["coefficient_cosine_median_min"],
    "no_material_regression_vs_best_single": full["aggregate_auc"] >= best_single - GATES["full_vs_best_single_auc_tolerance"],
    "multiple_nonrenderer_directional_families": nonrenderer_directional >= GATES["nonrenderer_directional_families_min"],
    "permutation_negative_control": permutation_control["passed"],
}
gate_passed = bool(all(checks.values()))

gate = {
    "status": "pass" if gate_passed else "rejected",
    "gate_passed": gate_passed,
    "rows": len(table),
    "validation": LOCK["validation"],
    "full_result": full,
    "best_single_family_auc": best_single,
    "nonrenderer_directional_family_count": int(nonrenderer_directional),
    "checks": checks,
    "frozen_requirements": GATES,
    "permutation_control": {k: v for k, v in permutation_control.items() if k != "aucs"},
    "exact_step8b_bank_reproduced": True,
    "rule_7a_guard_passed": True,
    "competition_test_enumerated": False,
    "competition_test_read": False,
    "candidate_created": False,
    "competition_submission_created": False,
}
(OUT / "combined_ranker_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

# Fit one portable full-bank model for later *publicly gated* use. This file is
# not applied to competition test data here and is not a candidate.
folds = fold_assignments(np.arange(len(y)), y, families, GATES["inner_folds"])
renderer_oof = np.zeros(len(y), dtype=np.float64)
for fold in range(GATES["inner_folds"]):
    train = folds != fold
    valid = folds == fold
    model = fit_logistic(X_renderer[train], y[train], GATES["renderer_l2"], GATES["fit_steps"], fold)
    renderer_oof[valid] = predict_logistic(model, X_renderer[valid])
renderer_full = fit_logistic(X_renderer, y, GATES["renderer_l2"], GATES["fit_steps"], 900)
x_combined_oof = np.column_stack([renderer_oof, base])
combined_full = fit_logistic(x_combined_oof, y, GATES["combined_l2"], GATES["fit_steps"], 901)
portable_threshold = float(np.quantile(predict_logistic(combined_full, x_combined_oof)[y == 0], GATES["clean_calibration_quantile"]))
np.savez_compressed(
    OUT / "combined_ranker_model.npz",
    renderer_center=renderer_full[0], renderer_scale=renderer_full[1], renderer_weights=renderer_full[2], renderer_bias=renderer_full[3],
    combined_center=combined_full[0], combined_scale=combined_full[1], combined_weights=combined_full[2], combined_bias=combined_full[3],
    combined_threshold=np.asarray(portable_threshold),
    renderer_features=np.asarray(RENDERER_FEATURES), combined_features=np.asarray(COMBINED_FEATURES),
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
names = list(ABLATIONS)
axes[0].barh(names, [all_results[n]["aggregate_auc"] for n in names], color="#4c78a8")
axes[0].axvline(GATES["aggregate_auc_min"], color="#e45756", linestyle="--")
axes[0].set_xlim(0.45, 1.01)
axes[0].set_xlabel("Nested aggregate AUC")
axes[0].set_title("Feature-family ablations")
full_scores = collapse_repeated_predictions(all_predictions["full"])
for family in ["public_poison", *CLEAN_FAMILIES]:
    values = full_scores.loc[full_scores.family == family, "score"].to_numpy()
    axes[1].hist(values, bins=np.linspace(0, 1, 21), alpha=0.45, label=family)
axes[1].set_xlabel("Combined poison probability")
axes[1].set_title("Nested held-out sample scores")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "combined_ranker_gate.png", dpi=170)
plt.close(fig)

report = {
    "experiment": LOCK["experiment"],
    "status": gate["status"],
    "gate_passed": gate_passed,
    "interpretation": (
        "Step 10 is promotable for candidate construction under later gates."
        if gate_passed else
        "Step 10 is rejected; do not construct a candidate from this ranker."
    ),
    "artifact_audit": artifact_audit,
    "gate": gate,
    "ablations": all_results,
    "portable_model_written_but_not_applied_to_test": True,
    "selection_boundary": LOCK["selection_boundary"],
}
(OUT / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log("STEP10_COMPLETE", gate_passed=gate_passed, aggregate_auc=full["aggregate_auc"], max_fpr=full["maximum_clean_family_fpr"])
print(json.dumps(gate, indent=2))
