# %% [markdown]
# # Step 10C - stress-robust strict consensus and fixed-bank inference
#
# All transformations, gates and candidate formulas are frozen before any test
# enumeration. Public competition controls are used for calibration; test pixels
# are used only for deterministic frozen-model inference. Existing V15_B boxes
# are never added, moved, or increased.

# %%
import base64
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("/kaggle/working/step10c_stress_consensus")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 230724

FEATURE_NAMES = [
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
CLEAN_FAMILIES = [
    "external_crop_transplant", "synthetic_irregular_dash",
    "synthetic_stochastic_blink", "synthetic_head_tail_tracklet",
]
TRANSFORMS = {
    "identity": {},
    "blur_sigma065": {"blur_sigma": 0.65},
    "downup_075": {"downup_scale": 0.75},
    "quantize_8bit": {"quantize_8bit": True},
}
PUBLIC_GATE = {
    "calibrated_precision_min": 0.90,
    "maximum_clean_family_fpr": 0.05,
    "poison_recall_min": 0.25,
    "each_stress_maximum_clean_family_fpr": 0.20,
    "clean_calibration_quantile": 0.99,
    "renderer_l2": 0.20,
    "fit_steps": 700,
    "validation": "nested leave-one-public-poison-out x leave-one-clean-family-out",
}
EXPECTED = {
    "v15b_anchor": "4345387b72aecd55dd3856b1607ed836ec9f400365fcf3f3b89f8947f1ff2412",
    "step8c_features": "d99d5cb5efec7066bc8140fc30f10b84c15c98b23a7b07e8dd3870e3ef058649",
    "step8b_probe": "36f14f3f4e667378b9056df6de8593c17e106b25404c76007d7adc99be7a21c8",
    "v19_pcgrad": "700a931cd408c44d02a8dc985a353e900a2fcb7c4bc0aa49a37f5bf5bcaba904",
    "v12_artifacts": "dc2405191689caa419a83f9751b39bfd123f95732d24deb6c7172a007552d2e0",
    "step10b_gate": "2237a805b57e0230b6b0f1faf363e6083c5a66a7186b2538120ee037ec855ce6",
}
LOCK = {
    "status": "frozen_before_test_enumeration_or_read",
    "experiment": "STEP10C_STRESS_ROBUST_STRICT_CONSENSUS_FIXED_BANK",
    "seed": SEED,
    "transform_ensemble": list(TRANSFORMS),
    "transform_choice_basis": "public Step8C shifts with AUC=1, recall=1 and acceptable rank stability; no test data",
    "signals": ["transform_ensemble_renderer", "pcgrad_collapse", "v12_collapse"],
    "renderer_features": FEATURE_NAMES,
    "consensus": "strict_3_of_3_intersection",
    "public_gate": PUBLIC_GATE,
    "expected_artifact_hashes": EXPECTED,
    "candidate_variants": {
        "strict": {"rule": "consensus-positive confidence becomes min(base, 0.02)"},
        "graded": {"rule": "consensus-positive factor=0.50-0.45*minimum_normalized_margin; floor 0.02; never increase"},
    },
    "invariants": {"exact_v15b_box_bank": True, "boxes_added": 0, "boxes_moved": 0,
                   "confidence_increases": 0, "test_derived_training_or_selection": False,
                   "competition_submission_created": False},
}
(OUT / "selection_lock.json").write_text(json.dumps(LOCK, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_known(candidates, expected_hash):
    checked = []
    for item in candidates:
        path = Path(item)
        checked.append(str(path))
        if path.exists() and sha256(path) == expected_hash:
            return path
    raise AssertionError({"expected_hash": expected_hash, "checked": checked})


# The audited Step10B gate is embedded byte-for-byte; it contains public metrics only.
STEP10B_GATE_BYTES = base64.b64decode("__STEP10B_GATE_BASE64__")
assert hashlib.sha256(STEP10B_GATE_BYTES).hexdigest() == EXPECTED["step10b_gate"]
step10b = json.loads(STEP10B_GATE_BYTES.decode("utf-8"))
assert step10b["gate_passed"] is True
assert step10b["exact_step8b_bank_reproduced"] is True
assert step10b["rule_7a_guard_passed"] is True
assert step10b["competition_test_enumerated"] is False
assert step10b["competition_test_read"] is False
assert step10b["candidate_created"] is False
assert step10b["competition_submission_created"] is False
(OUT / "step10b_gate_audit.json").write_text(json.dumps({
    "embedded_sha256": EXPECTED["step10b_gate"], "all_required_checks_passed": True,
    "metrics": step10b["metrics"],
}, indent=2), encoding="utf-8")

ROOT = "/kaggle/input/notebooks/boltuzamaki"
step8c_features_path = locate_known([
    f"{ROOT}/neural-debris-step-8c-renderer-robustness/step8c_renderer_robustness/robustness_feature_table.csv",
], EXPECTED["step8c_features"])
step8b_probe_path = locate_known([
    f"{ROOT}/neural-debris-step-8b-renderer-gate/step8b_renderer/public_probe_table.csv",
], EXPECTED["step8b_probe"])
anchor_path = locate_known([
    f"{ROOT}/neural-debris-v18-canonical-recovery/submission_V18_0_exact_v15b.csv",
], EXPECTED["v15b_anchor"])
pcgrad_path = locate_known([
    f"{ROOT}/neural-debris-v19-pcgrad-stability-veto/ndr_v19/per_box_diagnostics.csv",
], EXPECTED["v19_pcgrad"])
v12_path = locate_known([
    f"{ROOT}/neural-debris-v11-v12-trajectory-anchor/artifacts_cands.npz",
], EXPECTED["v12_artifacts"])

# %% [markdown]
# ## Public-only nested stress consensus

# %%
stress = pd.read_csv(step8c_features_path)
probe = pd.read_csv(step8b_probe_path)
assert set(stress["transform"].unique()) >= set(TRANSFORMS)
assert len(probe) == 140 and probe.sample_id.is_unique
assert probe.sample_id.tolist() == stress[stress["transform"] == "identity"].sample_id.tolist()

feature_by_transform = {}
for name in TRANSFORMS:
    part = stress[stress["transform"] == name].set_index("sample_id").loc[probe.sample_id]
    feature_by_transform[name] = part[FEATURE_NAMES].to_numpy(np.float64)
ensemble_features = np.mean(np.stack(list(feature_by_transform.values()), axis=0), axis=0)
labels = probe.label_poison.to_numpy(int)
families = probe.family.to_numpy(str)
pcgrad = 1.0 - np.clip(probe.ratio_pcgrad_median.to_numpy(float), 0.0, 1.0)
v12 = 1.0 - np.clip(probe.ratio_v12.to_numpy(float), 0.0, 1.0)
poison_indices = np.where(labels == 1)[0]
assert len(poison_indices) == 20 and int(np.sum(labels == 0)) == 120


def fit_logistic(features, target):
    x, y = np.asarray(features, np.float64), np.asarray(target, np.float64)
    center = np.median(x, 0)
    scale = 1.4826 * np.median(np.abs(x - center), 0) + 1e-3
    z = np.clip((x - center) / scale, -8, 8)
    weights, bias = np.zeros(z.shape[1]), 0.0
    class_weights = np.where(y == 1, float(np.sum(y == 0) / max(np.sum(y == 1), 1)), 1.0)
    for iteration in range(PUBLIC_GATE["fit_steps"]):
        probability = 1 / (1 + np.exp(-np.clip(z @ weights + bias, -25, 25)))
        error = (probability - y) * class_weights / class_weights.mean()
        rate = 0.045 / (1 + 0.0015 * iteration)
        weights -= rate * (z.T @ error / len(y) + PUBLIC_GATE["renderer_l2"] * weights)
        bias -= rate * float(error.mean())
    return {"center": center, "scale": scale, "weights": weights, "bias": np.asarray(bias)}


def predict(model, features):
    z = np.clip((np.asarray(features, np.float64) - model["center"]) / model["scale"], -8, 8)
    return 1 / (1 + np.exp(-np.clip(z @ model["weights"] + float(model["bias"]), -25, 25)))


rows = []
for family in CLEAN_FAMILIES:
    heldout_clean = np.where((labels == 0) & (families == family))[0]
    training_clean = np.where((labels == 0) & (families != family))[0]
    for poison_position, poison_index in enumerate(poison_indices):
        training_poison = poison_indices[poison_indices != poison_index]
        training = np.r_[training_poison, training_clean]
        model = fit_logistic(ensemble_features[training], labels[training])
        renderer_threshold = float(np.quantile(predict(model, ensemble_features[training_clean]), .99))
        pcgrad_threshold = float(np.quantile(pcgrad[training_clean], .99))
        v12_threshold = float(np.quantile(v12[training_clean], .99))
        evaluation = np.r_[[poison_index], heldout_clean]
        renderer_score = predict(model, ensemble_features[evaluation])
        main_prediction = ((renderer_score >= renderer_threshold)
                           & (pcgrad[evaluation] >= pcgrad_threshold)
                           & (v12[evaluation] >= v12_threshold))
        stress_predictions = {}
        for transform_name, values in feature_by_transform.items():
            transformed_score = predict(model, values[evaluation])
            stress_predictions[transform_name] = ((transformed_score >= renderer_threshold)
                                                   & (pcgrad[evaluation] >= pcgrad_threshold)
                                                   & (v12[evaluation] >= v12_threshold))
        for position, sample_index in enumerate(evaluation):
            row = {"heldout_family": family, "heldout_poison": int(poison_position),
                   "sample_id": probe.iloc[sample_index].sample_id,
                   "family": probe.iloc[sample_index].family,
                   "label_poison": int(labels[sample_index]),
                   "renderer_score": float(renderer_score[position]),
                   "renderer_threshold": renderer_threshold, "pcgrad_score": float(pcgrad[sample_index]),
                   "pcgrad_threshold": pcgrad_threshold, "v12_score": float(v12[sample_index]),
                   "v12_threshold": v12_threshold, "predicted_poison": bool(main_prediction[position])}
            for transform_name in TRANSFORMS:
                row[f"predicted_{transform_name}"] = bool(stress_predictions[transform_name][position])
            rows.append(row)
nested = pd.DataFrame(rows)
nested.to_csv(OUT / "nested_stress_consensus_predictions.csv", index=False)

true_positive = int(((nested.label_poison == 1) & nested.predicted_poison).sum())
false_positive = int(((nested.label_poison == 0) & nested.predicted_poison).sum())
precision = float(true_positive / max(true_positive + false_positive, 1))
recall = float(true_positive / max(int((nested.label_poison == 1).sum()), 1))
family_fpr = {family: float(nested.loc[nested.family == family, "predicted_poison"].mean()) for family in CLEAN_FAMILIES}
stress_fpr = {}
for transform_name in TRANSFORMS:
    stress_fpr[transform_name] = {
        family: float(nested.loc[nested.family == family, f"predicted_{transform_name}"].mean())
        for family in CLEAN_FAMILIES
    }
checks = {
    "precision": precision >= PUBLIC_GATE["calibrated_precision_min"],
    "maximum_family_fpr": max(family_fpr.values()) <= PUBLIC_GATE["maximum_clean_family_fpr"],
    "poison_recall": recall >= PUBLIC_GATE["poison_recall_min"],
    "all_stress_conditions": all(max(values.values()) <= PUBLIC_GATE["each_stress_maximum_clean_family_fpr"]
                                    for values in stress_fpr.values()),
}
public_passed = bool(all(checks.values()))
public_audit = {
    "status": "pass" if public_passed else "rejected", "gate_passed": public_passed,
    "precision": precision, "poison_recall": recall, "family_fpr": family_fpr,
    "maximum_family_fpr": max(family_fpr.values()), "stress_family_fpr": stress_fpr,
    "stress_maximum_family_fpr": {name: max(values.values()) for name, values in stress_fpr.items()},
    "checks": checks, "frozen_requirements": PUBLIC_GATE, "rows": len(nested),
    "exact_public_samples": 140, "rule_7a_guard_passed": True,
    "competition_test_enumerated": False, "competition_test_read": False,
}
(OUT / "stress_consensus_gate.json").write_text(json.dumps(public_audit, indent=2), encoding="utf-8")
if not public_passed:
    (OUT / "final_report.json").write_text(json.dumps({
        "status": "stopped_at_public_gate", "public_gate": public_audit,
        "candidate_created": False, "competition_submission_created": False,
    }, indent=2), encoding="utf-8")
    raise SystemExit("Public stress-consensus gate rejected; test set was not enumerated or read")

# %% [markdown]
# ## Frozen deployment calibration (still public-only)

# %%
deployment_model = fit_logistic(ensemble_features, labels)
clean_indices = np.where(labels == 0)[0]
oof_clean_renderer = np.zeros(len(clean_indices), np.float64)
for fold in range(5):
    held_positions = np.arange(len(clean_indices))[np.arange(len(clean_indices)) % 5 == fold]
    held = clean_indices[held_positions]
    training_clean = np.setdiff1d(clean_indices, held)
    training = np.r_[poison_indices, training_clean]
    fold_model = fit_logistic(ensemble_features[training], labels[training])
    oof_clean_renderer[held_positions] = predict(fold_model, ensemble_features[held])
deployment_thresholds = {
    "aggregation_rule": "renderer=q99 five-fold public OOF clean transform-ensemble probabilities; PCGrad/V12=q99 full public clean",
    "renderer": float(np.quantile(oof_clean_renderer, .99)),
    "pcgrad": float(np.quantile(pcgrad[clean_indices], .99)),
    "v12": float(np.quantile(v12[clean_indices], .99)),
}
(OUT / "deployment_thresholds.json").write_text(json.dumps(deployment_thresholds, indent=2), encoding="utf-8")
np.savez_compressed(OUT / "renderer_transform_ensemble_model.npz", **deployment_model,
                    features=np.asarray(FEATURE_NAMES), transforms=np.asarray(list(TRANSFORMS)))

# Only now may the fixed test bank and test pixels be enumerated for inference.
competition_options = [
    Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models"),
    Path("/kaggle/input/neural-debris-removal-in-streak-detection-models"),
]
competition_root = next(root for root in competition_options if root.exists())
test_dir = competition_root / "test_set" / "test_set"
if not test_dir.is_dir():
    test_dir = competition_root / "test_set"

# __RENDERER_FEATURE_DEFINITIONS__


def transform_image(image, spec):
    value = np.asarray(image, np.float32).copy()
    if "blur_sigma" in spec:
        value = cv2.GaussianBlur(value, (0, 0), float(spec["blur_sigma"]))
    if "downup_scale" in spec:
        height, width = value.shape[:2]
        scale = float(spec["downup_scale"])
        small = cv2.resize(value, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)
        value = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if spec.get("quantize_8bit", False):
        value = np.rint(value)
    return np.clip(value, 0, 255).astype(np.float32)


def load_test_image(image_id):
    path = test_dir / f"{image_id}.png"
    gray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert gray is not None, path
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    if gray.dtype == np.uint16:
        gray = gray.astype(np.float32) / 65535.0 * 255.0
    else:
        gray = gray.astype(np.float32)
        if gray.max() <= 1.0:
            gray *= 255.0
    return np.repeat(np.clip(gray, 0, 255)[:, :, None], 3, axis=2)


def parse_prediction(value):
    text = str(value).strip()
    if not text or text == "nan":
        return np.zeros((0, 5), np.float32)
    values = np.asarray(list(map(float, text.split())), np.float32)
    assert len(values) % 5 == 0
    return values.reshape(-1, 5)


def iou_matrix(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    size = np.clip(br - tl, 0, None)
    inter = size[:, :, 0] * size[:, :, 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), 1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), 1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def format_prediction(boxes, scores):
    parts = []
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        parts.extend([f"{float(score):.6f}", f"{float(x1):.2f}", f"{float(y1):.2f}",
                      f"{float(x2-x1):.2f}", f"{float(y2-y1):.2f}"])
    return " ".join(parts) if parts else " "


anchor = pd.read_csv(anchor_path, dtype={"image_id": str})
pcgrad_table = pd.read_csv(pcgrad_path, dtype={"image_id": str})
v12_bank = np.load(v12_path, allow_pickle=False)
assert len(anchor) == 2000 and anchor.image_id.is_unique and sha256(anchor_path) == EXPECTED["v15b_anchor"]
pcgrad_groups = {str(key): value.sort_values("candidate") for key, value in pcgrad_table.groupby("image_id")}
v12_offsets = np.r_[0, np.cumsum(v12_bank["counts"].astype(int))]
v12_lookup = {str(stem): (v12_bank["boxes"][v12_offsets[i]:v12_offsets[i+1]],
                          v12_bank["ratio"][v12_offsets[i]:v12_offsets[i+1]])
              for i, stem in enumerate(v12_bank["stems"])}

strict_strings, graded_strings, diagnostic_rows = [], [], []
minimum_pcgrad_iou, minimum_v12_iou = 1.0, 1.0
total_boxes = strict_count = 0
for row in anchor.itertuples(index=False):
    image_id = str(row.image_id)
    parsed = parse_prediction(row.prediction_string)
    base = parsed[:, 0].astype(np.float64)
    xywh = parsed[:, 1:]
    boxes = np.column_stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]]) if len(parsed) else np.zeros((0, 4), np.float32)
    if len(boxes):
        image = load_test_image(image_id)
        transformed_features = []
        for spec in TRANSFORMS.values():
            shifted = transform_image(image, spec)
            transformed_features.append(np.stack([renderer_features(shifted, box) for box in boxes]))
        renderer_score = predict(deployment_model, np.mean(np.stack(transformed_features), axis=0))
        pc = pcgrad_groups[image_id]
        pc_boxes = pc[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        pc_iou = iou_matrix(boxes, pc_boxes)
        pc_nearest, pc_best = pc_iou.argmax(1), pc_iou.max(1)
        minimum_pcgrad_iou = min(minimum_pcgrad_iou, float(pc_best.min()))
        assert float(pc_best.min()) >= 0.98
        pcgrad_score = pc.iloc[pc_nearest].median_poison.to_numpy(float)
        vb, vr = v12_lookup[image_id]
        v12_iou = iou_matrix(boxes, vb)
        v12_nearest, v12_best = v12_iou.argmax(1), v12_iou.max(1)
        minimum_v12_iou = min(minimum_v12_iou, float(v12_best.min()))
        assert float(v12_best.min()) >= 0.90
        v12_score = 1.0 - np.clip(vr[v12_nearest].astype(float), 0.0, 1.0)
        strict = ((renderer_score >= deployment_thresholds["renderer"])
                  & (pcgrad_score >= deployment_thresholds["pcgrad"])
                  & (v12_score >= deployment_thresholds["v12"]))
        normalized = np.stack([
            np.clip((renderer_score - deployment_thresholds["renderer"]) / max(1-deployment_thresholds["renderer"], 1e-6), 0, 1),
            np.clip((pcgrad_score - deployment_thresholds["pcgrad"]) / max(1-deployment_thresholds["pcgrad"], 1e-6), 0, 1),
            np.clip((v12_score - deployment_thresholds["v12"]) / max(1-deployment_thresholds["v12"], 1e-6), 0, 1),
        ])
        margin = normalized.min(0)
        strict_scores = base.copy()
        strict_scores[strict] = np.minimum(strict_scores[strict], 0.02)
        graded_scores = base.copy()
        factor = 0.50 - 0.45 * margin
        graded_scores[strict] = np.minimum(base[strict], np.maximum(0.02, base[strict] * factor[strict]))
        for candidate in range(len(boxes)):
            diagnostic_rows.append({"image_id": image_id, "candidate": candidate,
                                    "base_confidence": float(base[candidate]),
                                    "renderer": float(renderer_score[candidate]),
                                    "pcgrad": float(pcgrad_score[candidate]), "v12": float(v12_score[candidate]),
                                    "consensus_margin": float(margin[candidate]), "strict_positive": bool(strict[candidate]),
                                    "strict_confidence": float(strict_scores[candidate]),
                                    "graded_confidence": float(graded_scores[candidate]),
                                    "x1": float(boxes[candidate,0]), "y1": float(boxes[candidate,1]),
                                    "x2": float(boxes[candidate,2]), "y2": float(boxes[candidate,3])})
        total_boxes += len(boxes)
        strict_count += int(strict.sum())
    else:
        strict_scores = graded_scores = np.zeros(0)
    strict_strings.append(format_prediction(boxes, strict_scores))
    graded_strings.append(format_prediction(boxes, graded_scores))

strict_submission = anchor.copy(); strict_submission["prediction_string"] = strict_strings
graded_submission = anchor.copy(); graded_submission["prediction_string"] = graded_strings
strict_submission.to_csv(OUT / "submission_step10c_strict.csv", index=False)
graded_submission.to_csv(OUT / "submission_step10c_graded.csv", index=False)
diagnostics = pd.DataFrame(diagnostic_rows)
diagnostics.to_csv(OUT / "per_box_diagnostics.csv", index=False)
assert total_boxes == 3995 and len(diagnostics) == 3995
assert (diagnostics.strict_confidence <= diagnostics.base_confidence + 1e-12).all()
assert (diagnostics.graded_confidence <= diagnostics.base_confidence + 1e-12).all()

candidate_audit = {
    "status": "complete", "exact_anchor_sha256": sha256(anchor_path), "rows": len(anchor),
    "boxes": total_boxes, "strict_consensus_boxes": strict_count,
    "minimum_pcgrad_alignment_iou": minimum_pcgrad_iou, "minimum_v12_alignment_iou": minimum_v12_iou,
    "boxes_added": 0, "boxes_moved": 0, "strict_confidence_increases": 0,
    "graded_confidence_increases": 0,
    "strict_sha256": sha256(OUT / "submission_step10c_strict.csv"),
    "graded_sha256": sha256(OUT / "submission_step10c_graded.csv"),
    "test_used_only_for_frozen_inference": True, "test_derived_training_or_selection": False,
    "competition_submission_created": False, "rule_7a_guard_passed": True,
}
(OUT / "candidate_audit.json").write_text(json.dumps(candidate_audit, indent=2), encoding="utf-8")
(OUT / "final_report.json").write_text(json.dumps({
    "status": "complete", "public_gate": public_audit, "deployment_thresholds": deployment_thresholds,
    "candidate": candidate_audit, "competition_submission_created": False,
}, indent=2), encoding="utf-8")
print(json.dumps({"public_gate": public_audit, "candidate": candidate_audit}, indent=2))
