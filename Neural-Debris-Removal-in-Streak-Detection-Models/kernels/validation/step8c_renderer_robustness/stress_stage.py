# %% [markdown]
# ## Step 8C - frozen renderer robustness
#
# The exact Step 8B control bank and renderer ensemble are immutable here.
# Transformations and gates were serialized before any sample/checkpoint
# enumeration.  This stage never accesses competition test data and emits no
# candidate or submission.

# %%
assert len(frame) == 140
manifest_path = OUT / "sample_manifest.json"
manifest_sha = sha256(manifest_path)
assert manifest_sha == EXPECTED_STEP8B_MANIFEST_SHA256, {
    "expected": EXPECTED_STEP8B_MANIFEST_SHA256,
    "observed": manifest_sha,
}


def locate_frozen_artifact(filename, expected_sha):
    candidates = sorted(Path("/kaggle/input").rglob(filename))
    for candidate in candidates:
        if sha256(candidate) == expected_sha:
            return candidate
    raise AssertionError({"artifact": filename, "expected": expected_sha,
                          "candidates": [str(path) for path in candidates]})


models_path = locate_frozen_artifact("renderer_models.npz", EXPECTED_RENDERER_MODELS_SHA256)
source_manifest = locate_frozen_artifact("sample_manifest.json", EXPECTED_STEP8B_MANIFEST_SHA256)
assert json.loads(source_manifest.read_text(encoding="utf-8")) == json.loads(manifest_path.read_text(encoding="utf-8"))
(OUT / "frozen_artifact_manifest.json").write_text(json.dumps({
    "renderer_models": {"path": str(models_path), "sha256": sha256(models_path)},
    "step8b_sample_manifest": {"path": str(source_manifest), "sha256": sha256(source_manifest)},
    "reconstructed_sample_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
    "models_retrained_or_retuned": False,
}, indent=2), encoding="utf-8")
frozen = np.load(models_path, allow_pickle=False)


def frozen_model(family_index, poison_position):
    prefix = f"family{family_index}_poison{poison_position}"
    return {field: frozen[f"{prefix}_{field}"] for field in ("center", "scale", "weights", "bias")}


def frozen_predict(model, features):
    z = np.clip((np.asarray(features, np.float64) - model["center"]) / model["scale"], -8, 8)
    return 1 / (1 + np.exp(-np.clip(z @ model["weights"] + model["bias"], -25, 25)))


def transform_image(image, spec, sample_index):
    value = np.asarray(image, np.float32).copy()
    if "gain" in spec:
        value *= float(spec["gain"])
    if "gamma" in spec:
        normalized = np.clip(value / 255.0, 0.0, 1.0)
        value = 255.0 * np.power(normalized, float(spec["gamma"]))
    if "blur_sigma" in spec:
        value = cv2.GaussianBlur(value, (0, 0), float(spec["blur_sigma"]))
    if "noise_sigma" in spec:
        rng = np.random.default_rng(SEED * 1000 + sample_index)
        value += rng.normal(0.0, float(spec["noise_sigma"]), value.shape).astype(np.float32)
    if "downup_scale" in spec:
        height, width = value.shape[:2]
        scale = float(spec["downup_scale"])
        small = cv2.resize(value, (max(2, round(width * scale)), max(2, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
        value = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if spec.get("quantize_8bit", False):
        value = np.rint(value)
    return np.clip(value, 0.0, 255.0).astype(np.float32)


labels = frame.label_poison.to_numpy(int)
families_array = frame.family.to_numpy(str)
poison_indices = np.where(labels == 1)[0]
assert len(poison_indices) == 20


def aggregate_scores(features):
    scores = np.zeros(len(frame), np.float64)
    # Public poison: average the four exact leave-one-poison-out models.
    for poison_position, sample_index in enumerate(poison_indices):
        values = [float(frozen_predict(frozen_model(fi, poison_position), features[[sample_index]])[0])
                  for fi in range(len(CLEAN_FAMILIES))]
        scores[sample_index] = float(np.mean(values))
    # Public clean: average the twenty models from its held-out-family fold.
    for family_index, family in enumerate(CLEAN_FAMILIES):
        heldout = np.where((labels == 0) & (families_array == family))[0]
        for sample_index in heldout:
            values = [float(frozen_predict(frozen_model(family_index, pp), features[[sample_index]])[0])
                      for pp in range(len(poison_indices))]
            scores[sample_index] = float(np.mean(values))
    return scores


def rank_correlation(left, right):
    a = pd.Series(np.asarray(left, float)).rank(method="average").to_numpy(float)
    b = pd.Series(np.asarray(right, float)).rank(method="average").to_numpy(float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def auc_binary(labels_value, scores_value):
    """Dependency-free Mann-Whitney AUC, identical to the Step 8B helper."""
    labels_value = np.asarray(labels_value, int)
    scores_value = np.asarray(scores_value, float)
    positive = scores_value[labels_value == 1]
    negative = scores_value[labels_value == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = 0.0
    for value in positive:
        wins += float(np.sum(value > negative)) + 0.5 * float(np.sum(value == negative))
    return wins / (len(positive) * len(negative))


feature_sets = {}
long_rows = []
with Heartbeat("frozen_renderer_stress_features"):
    for transform_name, spec in STRESS_TRANSFORMS.items():
        transformed = [transform_image(image, spec, index) for index, image in enumerate(images)]
        features = np.stack([renderer_features(image, box) for image, box in
                             tqdm(zip(transformed, frame.target_box), total=len(frame), desc=transform_name)])
        feature_sets[transform_name] = features
        for index, sample in frame.iterrows():
            row = {"transform": transform_name, "sample_id": sample.sample_id,
                   "family": sample.family, "label_poison": int(sample.label_poison)}
            row.update({name: float(features[index, position]) for position, name in enumerate(FEATURE_NAMES)})
            long_rows.append(row)
pd.DataFrame(long_rows).to_csv(OUT / "robustness_feature_table.csv", index=False)

baseline_features = feature_sets["identity"]
baseline_scores = aggregate_scores(baseline_features)
baseline_gap = float(np.median(baseline_scores[labels == 1]) - np.median(baseline_scores[labels == 0]))
results = {}
with Heartbeat("frozen_renderer_stress_scoring"):
    for transform_name, features in feature_sets.items():
        scores = aggregate_scores(features)
        family_auc = {}
        family_recall = {}
        family_fpr = {}
        for family_index, family in enumerate(CLEAN_FAMILIES):
            heldout_clean = np.where((labels == 0) & (families_array == family))[0]
            training_clean = np.where((labels == 0) & (families_array != family))[0]
            poison_hits, false_rates = [], []
            for poison_position, poison_index in enumerate(poison_indices):
                model = frozen_model(family_index, poison_position)
                calibration = frozen_predict(model, baseline_features[training_clean])
                threshold = float(np.quantile(calibration, RENDERER_GATE["clean_calibration_quantile"]))
                poison_score = float(frozen_predict(model, features[[poison_index]])[0])
                clean_scores = frozen_predict(model, features[heldout_clean])
                poison_hits.append(poison_score >= threshold)
                false_rates.append(float(np.mean(clean_scores >= threshold)))
            subset = np.r_[poison_indices, heldout_clean]
            family_auc[family] = auc_binary(labels[subset], scores[subset])
            family_recall[family] = float(np.mean(poison_hits))
            family_fpr[family] = float(round(float(np.mean(false_rates)), 12))
        gap = float(np.median(scores[labels == 1]) - np.median(scores[labels == 0]))
        results[transform_name] = {
            "aggregate_auc": auc_binary(labels, scores),
            "family_auc": family_auc,
            "median_poison_recall": float(np.median(list(family_recall.values()))),
            "maximum_family_fpr": float(max(family_fpr.values())),
            "family_recall": family_recall,
            "family_fpr": family_fpr,
            "rank_correlation_vs_identity": 1.0 if transform_name == "identity" else rank_correlation(baseline_scores, scores),
            "median_absolute_score_delta": float(np.median(np.abs(scores - baseline_scores))),
            "poison_clean_median_gap": gap,
            "margin_retention": float(gap / max(baseline_gap, 1e-9)),
        }

identity_ok = results["identity"]["aggregate_auc"] >= ROBUSTNESS_GATE["identity_aggregate_auc_min"]
shift_results = [value for name, value in results.items() if name != "identity"]
robustness_passed = bool(
    identity_ok
    and all(value["aggregate_auc"] >= ROBUSTNESS_GATE["each_shift_aggregate_auc_min"] for value in shift_results)
    and all(min(value["family_auc"].values()) >= ROBUSTNESS_GATE["each_shift_each_family_auc_min"] for value in shift_results)
    and all(value["median_poison_recall"] >= ROBUSTNESS_GATE["each_shift_median_poison_recall_min"] for value in shift_results)
    and all(value["maximum_family_fpr"] <= ROBUSTNESS_GATE["each_shift_maximum_family_fpr_max"] for value in shift_results)
    and all(value["rank_correlation_vs_identity"] >= ROBUSTNESS_GATE["each_shift_rank_correlation_min"] for value in shift_results)
    and all(value["margin_retention"] >= ROBUSTNESS_GATE["each_shift_margin_retention_min"] for value in shift_results)
)
audit = {
    "status": "pass" if robustness_passed else "rejected",
    "gate_passed": robustness_passed,
    "sample_manifest_sha256": manifest_sha,
    "renderer_models_sha256": sha256(models_path),
    "samples": len(frame),
    "models_retrained_or_retuned": False,
    "results": results,
    "frozen_requirements": ROBUSTNESS_GATE,
    "rule_7a_guard_passed": True,
    "competition_test_enumerated": False,
    "competition_test_read": False,
    "competition_submission_created": False,
}
(OUT / "renderer_robustness.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

names = list(results)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].bar(names, [results[name]["aggregate_auc"] for name in names], color="#47c2ff")
axes[0].axhline(ROBUSTNESS_GATE["each_shift_aggregate_auc_min"], color="#ff6b6b", linestyle="--")
axes[0].set_ylabel("aggregate AUC")
axes[1].bar(names, [min(results[name]["family_auc"].values()) for name in names], color="#62e7b4")
axes[1].axhline(ROBUSTNESS_GATE["each_shift_each_family_auc_min"], color="#ff6b6b", linestyle="--")
axes[1].set_ylabel("minimum clean-family AUC")
axes[2].bar(names, [results[name]["rank_correlation_vs_identity"] for name in names], color="#f4b860")
axes[2].axhline(ROBUSTNESS_GATE["each_shift_rank_correlation_min"], color="#ff6b6b", linestyle="--")
axes[2].set_ylabel("rank correlation vs identity")
for axis in axes:
    axis.set_ylim(0, 1.03)
    axis.tick_params(axis="x", rotation=65)
fig.suptitle(f"Step 8C frozen renderer robustness: {'PASS' if robustness_passed else 'REJECT'}")
fig.tight_layout()
fig.savefig(OUT / "renderer_robustness.png", dpi=160)
plt.close(fig)

report = {
    "status": "complete",
    "step": "8C",
    "decision": "renderer_robustness_confirmed" if robustness_passed else "do_not_promote_renderer_to_ranker",
    "robustness": audit,
    "candidate_created": False,
    "competition_submission_created": False,
}
(OUT / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log("STEP8C_COMPLETE", report=report)
print(json.dumps(report, indent=2))
