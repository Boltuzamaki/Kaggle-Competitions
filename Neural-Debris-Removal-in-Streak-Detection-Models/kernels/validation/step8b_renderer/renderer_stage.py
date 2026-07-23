# %% [markdown]
# ## Step 8B - renderer fingerprints on the validated Step 3B bank
#
# Every score below is out-of-fold by poison image and out-of-domain by clean
# family. The frozen gate is stricter than V22 and no test data is available.

# %%
FEATURE_NAMES = LOCK["renderer_features"]


def bit_entropy(values, modulus):
    values = np.asarray(np.rint(values), np.int64).ravel() % modulus
    histogram = np.bincount(values, minlength=modulus).astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    histogram = histogram[histogram > 0]
    return float(-(histogram * np.log(histogram)).sum() / math.log(modulus))


def run_length_cv(mask):
    lengths, current = [], 0
    for value in np.asarray(mask, bool):
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    if len(lengths) < 2:
        return 0.0
    lengths = np.asarray(lengths, float)
    return float(lengths.std() / (lengths.mean() + 1e-6))


def renderer_features(image, box):
    gray = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    x1, y1, x2, y2 = map(float, box)
    pad = max(8, int(round(0.16 * max(x2 - x1, y2 - y1))))
    left, right = max(0, int(math.floor(x1)) - pad), min(width, int(math.ceil(x2)) + pad)
    top, bottom = max(0, int(math.floor(y1)) - pad), min(height, int(math.ceil(y2)) + pad)
    crop = gray[top:bottom, left:right]
    if min(crop.shape, default=0) < 5:
        return np.zeros(len(FEATURE_NAMES), np.float32)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    baseline = float(np.median(border))
    noise = 1.4826 * float(np.median(np.abs(border - baseline))) + 0.75
    z = np.clip((crop - baseline) / noise, -5.0, 40.0)
    positive = np.clip(z, 0.0, None)
    active = positive > 2.5
    yy, xx = np.indices(crop.shape)
    weights = np.clip(positive - 1.5, 0.0, None)
    total = float(weights.sum()) + 1e-6
    mx, my = float((xx * weights).sum() / total), float((yy * weights).sum() / total)
    dx, dy = xx - mx, yy - my
    covariance = np.asarray([
        [(weights * dx * dx).sum() / total, (weights * dx * dy).sum() / total],
        [(weights * dx * dy).sum() / total, (weights * dy * dy).sum() / total],
    ], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    rotation = cv2.getRotationMatrix2D((mx, my), -angle, 1.0)
    rotated = cv2.warpAffine(positive, rotation, (positive.shape[1], positive.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rotated_active = rotated > 2.5
    rows, columns = np.where(rotated_active.any(1))[0], np.where(rotated_active.any(0))[0]
    if not len(rows) or not len(columns):
        return np.zeros(len(FEATURE_NAMES), np.float32)
    streak = rotated[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
    streak_active = streak > 2.5
    longitudinal, transverse = streak.sum(0), streak.sum(1)
    longitudinal_norm = longitudinal / (longitudinal.mean() + 1e-6)
    transverse_norm = transverse / (transverse.max() + 1e-6)
    q = max(1, len(longitudinal) // 8)
    endpoint_left, endpoint_right = float(longitudinal[:q].mean()), float(longitudinal[-q:].mean())
    endpoint_ratio = float((endpoint_left + endpoint_right) / (2 * longitudinal.mean() + 1e-6))
    endpoint_asymmetry = float(abs(endpoint_left - endpoint_right) / (longitudinal.mean() + 1e-6))
    center = longitudinal[len(longitudinal) // 3:max(len(longitudinal) // 3 + 1, 2 * len(longitudinal) // 3)]
    center_ratio = float(center.mean() / (longitudinal.mean() + 1e-6))
    longitudinal_cv = float(longitudinal.std() / (longitudinal.mean() + 1e-6))
    longitudinal_tv = float(np.mean(np.abs(np.diff(longitudinal_norm)))) if len(longitudinal) > 1 else 0.0
    gap_mask = longitudinal < 0.25 * longitudinal.mean()
    spectrum = np.abs(np.fft.rfft(longitudinal - longitudinal.mean()))
    fft_periodicity = float(spectrum[1:].max() / (spectrum[1:].sum() + 1e-6)) if len(spectrum) > 1 else 0.0
    autocorrelation = np.correlate(longitudinal - longitudinal.mean(), longitudinal - longitudinal.mean(), mode="full")[len(longitudinal) - 1:]
    autocorrelation /= float(autocorrelation[0]) + 1e-6
    autocorrelation_peak = float(autocorrelation[2:max(3, len(autocorrelation) // 2)].max()) if len(autocorrelation) > 5 else 0.0
    widths = streak_active.sum(0).astype(float)
    nonzero = widths > 0
    width_mean = float(widths[nonzero].mean()) if nonzero.any() else 0.0
    width_cv = float(widths[nonzero].std() / (width_mean + 1e-6)) if nonzero.any() else 0.0
    if nonzero.sum() >= 3:
        width_drift = float(abs(np.polyfit(np.linspace(-1, 1, len(widths))[nonzero], widths[nonzero], 1)[0]) / (width_mean + 1e-6))
    else:
        width_drift = 0.0
    transverse_symmetry = float(np.mean(np.abs(transverse_norm - transverse_norm[::-1])))
    coordinate, valid_profile = np.linspace(-1, 1, len(transverse_norm)), transverse_norm > 0.05
    if valid_profile.sum() >= 3:
        coefficient = np.polyfit(coordinate[valid_profile] ** 2, np.log(transverse_norm[valid_profile] + 1e-4), 1)
        gaussian = np.exp(np.polyval(coefficient, coordinate ** 2)); gaussian /= gaussian.max() + 1e-6
        gaussian_error = float(np.mean(np.abs(transverse_norm - gaussian)))
    else:
        gaussian_error = 1.0
    active_values = streak[streak_active]
    inside_noise_cv = float(active_values.std() / (active_values.mean() + 1e-6)) if len(active_values) else 0.0
    dilated = cv2.dilate(active.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    side = z[~dilated]
    integer_crop = np.clip(np.rint(crop), 0, 255).astype(np.uint8)
    major_value, minor_value = float(max(eigenvalues.max(), 1e-6)), float(max(eigenvalues.min(), 0.0))
    length, short = max(float(x2 - x1), float(y2 - y1)), max(min(float(x2 - x1), float(y2 - y1)), 1e-3)
    values = [
        float(active.mean()), float(weights.sum() / max(weights.size, 1)),
        float(math.log1p(major_value / (minor_value + 1e-3))), float(math.sqrt(minor_value / major_value)),
        endpoint_ratio, endpoint_asymmetry, center_ratio, longitudinal_cv, longitudinal_tv,
        float(gap_mask.mean()), fft_periodicity, autocorrelation_peak, run_length_cv(gap_mask),
        width_mean, width_cv, width_drift, transverse_symmetry, gaussian_error, inside_noise_cv,
        float(np.mean(side > 2.5)) if len(side) else 0.0, float(side.std()) if len(side) else 0.0,
        float(len(np.unique(integer_crop)) / max(integer_crop.size, 1)), bit_entropy(integer_crop, 2),
        bit_entropy(integer_crop, 4), bit_entropy(integer_crop, 16), bit_entropy(integer_crop, 256),
        float((x1 % 1 + y1 % 1 + x2 % 1 + y2 % 1) / 4),
        float((np.abs(np.diff(longitudinal_norm[:q + 1])).mean() + np.abs(np.diff(longitudinal_norm[-q - 1:])).mean()) / 2),
        float(np.mean(np.abs(crop - np.rint(crop)) > 1e-4)),
        float(np.sqrt(minor_value) / (np.sqrt(major_value) + 1e-6)), math.log1p(length), math.log1p(length / short),
    ]
    result = np.nan_to_num(np.asarray(values, np.float32), nan=0, posinf=20, neginf=-20)
    assert len(result) == len(FEATURE_NAMES)
    return np.clip(result, -20, 20)


def fit_logistic(features, labels):
    features, labels = np.asarray(features, np.float64), np.asarray(labels, np.float64)
    center = np.median(features, 0)
    scale = 1.4826 * np.median(np.abs(features - center), 0) + 1e-3
    z = np.clip((features - center) / scale, -8, 8)
    weights, bias = np.zeros(z.shape[1]), 0.0
    class_weights = np.where(labels == 1, float(np.sum(labels == 0) / max(np.sum(labels == 1), 1)), 1.0)
    for iteration in range(RENDERER_GATE["steps"]):
        probability = 1 / (1 + np.exp(-np.clip(z @ weights + bias, -25, 25)))
        error = (probability - labels) * class_weights / class_weights.mean()
        rate = 0.045 / (1 + 0.0015 * iteration)
        weights -= rate * (z.T @ error / len(labels) + RENDERER_GATE["l2"] * weights)
        bias -= rate * float(error.mean())
    return {"center": center, "scale": scale, "weights": weights, "bias": bias}


def predict_logistic(model, features):
    z = np.clip((np.asarray(features, np.float64) - model["center"]) / model["scale"], -8, 8)
    return 1 / (1 + np.exp(-np.clip(z @ model["weights"] + model["bias"], -25, 25)))


with Heartbeat("renderer_features"):
    feature_matrix = np.stack([renderer_features(image, box) for image, box in
                               tqdm(zip(images, frame.target_box), total=len(frame), desc="renderer features")])
assert feature_matrix.shape == (140, len(FEATURE_NAMES))
feature_frame = frame[["sample_id", "family", "label_poison", "score_original"]].copy()
for index, name in enumerate(FEATURE_NAMES):
    feature_frame[name] = feature_matrix[:, index]
feature_frame.to_csv(OUT / "renderer_feature_table.csv", index=False)

labels, families_array = frame.label_poison.to_numpy(int), frame.family.to_numpy(str)
poison_indices = np.where(labels == 1)[0]
family_results, model_store, coefficient_rows = {}, {}, []
aggregate_poison, aggregate_clean = np.zeros(len(poison_indices)), []
with Heartbeat("renderer_grouped_validation"):
    for family_index, heldout_family in enumerate(CLEAN_FAMILIES):
        heldout_clean = np.where(families_array == heldout_family)[0]
        training_clean = np.where((labels == 0) & (families_array != heldout_family))[0]
        poison_scores, clean_sum, poison_hits, false_rates = np.zeros(len(poison_indices)), np.zeros(len(heldout_clean)), [], []
        for poison_position, poison_index in enumerate(poison_indices):
            training_poison = poison_indices[poison_indices != poison_index]
            training_indices = np.concatenate([training_poison, training_clean])
            model = fit_logistic(feature_matrix[training_indices], labels[training_indices])
            model_store[f"family{family_index}_poison{poison_position}"] = model
            poison_score = float(predict_logistic(model, feature_matrix[[poison_index]])[0])
            clean_scores = predict_logistic(model, feature_matrix[heldout_clean])
            calibration = predict_logistic(model, feature_matrix[training_clean])
            threshold = float(np.quantile(calibration, RENDERER_GATE["clean_calibration_quantile"]))
            poison_scores[poison_position] = poison_score
            clean_sum += clean_scores
            poison_hits.append(poison_score >= threshold)
            false_rates.append(float(np.mean(clean_scores >= threshold)))
            coefficient_rows.append(model["weights"] / (np.linalg.norm(model["weights"]) + 1e-9))
        clean_scores = clean_sum / len(poison_indices)
        family_results[heldout_family] = {
            "auc": auc_binary(np.r_[np.ones(len(poison_scores)), np.zeros(len(clean_scores))], np.r_[poison_scores, clean_scores]),
            "poison_recall_at_training_clean_q99": float(np.mean(poison_hits)),
            # Every rate is an exact integer count over 30 held-out controls.
            # Canonicalize the decimal so an exact 10% is not represented as
            # 0.10000000000000002 and incorrectly rejected against <= 0.10.
            "heldout_clean_false_positive_rate": float(round(float(np.mean(false_rates)), 12)),
            "poison_score_median": float(np.median(poison_scores)),
            "clean_score_median": float(np.median(clean_scores)),
        }
        aggregate_poison += poison_scores / len(CLEAN_FAMILIES)
        aggregate_clean.extend(clean_scores.tolist())

aggregate_clean = np.asarray(aggregate_clean)
aggregate_auc = auc_binary(np.r_[np.ones(len(aggregate_poison)), np.zeros(len(aggregate_clean))], np.r_[aggregate_poison, aggregate_clean])
coefficient_rows = np.asarray(coefficient_rows)
median_direction = np.median(coefficient_rows, 0); median_direction /= np.linalg.norm(median_direction) + 1e-9
coefficient_cosine_median = float(np.median(coefficient_rows @ median_direction))
median_recall = float(np.median([value["poison_recall_at_training_clean_q99"] for value in family_results.values()]))
maximum_fpr = float(round(max(value["heldout_clean_false_positive_rate"] for value in family_results.values()), 12))
renderer_passed = bool(gate_passed and aggregate_auc >= RENDERER_GATE["aggregate_auc_min"]
                       and all(value["auc"] >= RENDERER_GATE["each_clean_family_auc_min"] for value in family_results.values())
                       and median_recall >= RENDERER_GATE["median_family_recall_min"]
                       and maximum_fpr <= RENDERER_GATE["maximum_family_false_positive_rate"]
                       and coefficient_cosine_median >= RENDERER_GATE["coefficient_cosine_median_min"])
renderer_audit = {
    "status": "pass" if renderer_passed else "rejected", "gate_passed": renderer_passed,
    "rows": len(frame), "features": FEATURE_NAMES,
    "validation": "leave-one-poison-image-out nested inside leave-one-clean-family-out",
    "aggregate_auc": aggregate_auc, "family_results": family_results,
    "median_family_recall": median_recall, "maximum_family_false_positive_rate": maximum_fpr,
    "coefficient_cosine_median": coefficient_cosine_median, "frozen_requirements": RENDERER_GATE,
    "step3b_gate_reproduced": gate_passed, "rule_7a_guard_passed": True,
    "competition_test_enumerated": False, "competition_test_read": False,
    "competition_submission_created": False,
}
(OUT / "renderer_gate.json").write_text(json.dumps(renderer_audit, indent=2), encoding="utf-8")
np.savez_compressed(OUT / "renderer_models.npz", **{f"{key}_{field}": value for key, model in model_store.items() for field, value in model.items()})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
family_names = list(family_results)
axes[0].bar(family_names, [family_results[name]["auc"] for name in family_names], color="#47c2ff")
axes[0].axhline(RENDERER_GATE["each_clean_family_auc_min"], color="#ff6b6b", linestyle="--")
axes[0].set_ylim(0, 1); axes[0].set_ylabel("held-out family AUC"); axes[0].tick_params(axis="x", rotation=25)
importance = np.argsort(np.abs(median_direction))[-12:]
axes[1].barh([FEATURE_NAMES[index] for index in importance], median_direction[importance], color="#62e7b4")
axes[1].set_xlabel("median normalized coefficient")
fig.suptitle(f"Step 8B renderer gate: {'PASS' if renderer_passed else 'REJECT'} | aggregate AUC={aggregate_auc:.3f}")
fig.tight_layout(); fig.savefig(OUT / "renderer_gate.png", dpi=160); plt.close(fig)

combined_report = {
    "status": "complete", "step": "8B",
    "decision": "continue_to_tiny_ranker" if renderer_passed else "stop_renderer_branch",
    "step3b": audit, "renderer": renderer_audit, "rule_7a_guard_passed": True,
    "competition_test_read": False, "competition_submission_created": False,
}
(OUT / "final_report.json").write_text(json.dumps(combined_report, indent=2), encoding="utf-8")
log("STEP8B_COMPLETE", report=combined_report)
print(json.dumps(combined_report, indent=2))
