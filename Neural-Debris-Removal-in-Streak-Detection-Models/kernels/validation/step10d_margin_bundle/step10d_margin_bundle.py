from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/step10d_margin_bundle")
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED_ANCHOR_SHA = "4345387b72aecd55dd3856b1607ed836ec9f400365fcf3f3b89f8947f1ff2412"
MID, FLOOR = 0.21, 0.02


def one(name: str) -> Path:
    hits = list(INPUT.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one {name}, found {len(hits)}: {hits}")
    return hits[0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(pred & labels))
    fp = int(np.sum(pred & ~labels))
    return {
        "tp": tp,
        "fp": fp,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(int(labels.sum()), 1)),
    }


# Public-only calibration. The lock is serialized before frozen test inference.
public = pd.read_csv(one("nested_stress_consensus_predictions.csv"))
labels = public.label_poison.to_numpy(bool)
excess = np.column_stack(
    [
        (public.renderer_score - public.renderer_threshold) / (1 - public.renderer_threshold),
        (public.pcgrad_score - public.pcgrad_threshold) / (1 - public.pcgrad_threshold),
        (public.v12_score - public.v12_threshold) / (1 - public.v12_threshold),
    ]
)
support2 = np.sort(excess, axis=1)[:, -2]
margin3 = excess.min(axis=1)
three = (excess >= 0).all(axis=1)

threshold_2of3 = float(np.nextafter(support2[~labels].max(), np.inf))
threshold_3of3_q10 = float(np.quantile(margin3[labels & three], 0.10))
threshold_high_q25 = float(np.quantile(support2[labels], 0.25))

pred_2of3 = support2 >= threshold_2of3
pred_3of3_q10 = three & (margin3 >= threshold_3of3_q10)
lock = {
    "status": "frozen_before_test_inference_artifacts_read",
    "experiment": "STEP10D_PUBLIC_MARGIN_DOSE_BUNDLE",
    "expected_anchor_sha256": EXPECTED_ANCHOR_SHA,
    "public_thresholds": {
        "two_of_three_public_clean_max": threshold_2of3,
        "three_of_three_public_poison_q10": threshold_3of3_q10,
        "two_of_three_public_poison_q25_for_high_tier": threshold_high_q25,
    },
    "public_metrics": {
        "two_of_three_core": metrics(pred_2of3, labels),
        "three_of_three_q10": metrics(pred_3of3_q10, labels),
    },
    "variants": {
        "D1": "2-of-3 public-clean-max; 0.21 tier -> 0.02",
        "D2": "3-of-3 public-poison q10 core; 0.21 tier -> 0.02",
        "D3": "D1 plus all 2-of-3-core higher tiers capped at 0.21",
        "D4": "D1 selection with frozen continuous margin dose on 0.21 tier",
        "D5": "D1 plus only public-poison-q25 high-tier support capped at 0.21",
    },
    "rule_7a": {
        "test_derived_thresholds": False,
        "leaderboard_derived_thresholds": False,
        "boxes_added": 0,
        "boxes_moved": 0,
        "confidence_increases": 0,
        "competition_submission_created": False,
    },
}
(OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")

# Frozen inference artifacts are opened only after public calibration is locked.
diagnostics = pd.read_csv(one("per_box_diagnostics.csv"))
thresholds = json.loads(one("deployment_thresholds.json").read_text(encoding="utf-8"))
template = pd.read_csv(one("submission_step10c_strict.csv"), dtype={"image_id": str})
if len(template) != 2000 or template.id.nunique() != 2000 or len(diagnostics) != 3995:
    raise RuntimeError("Unexpected fixed-bank dimensions")

test_excess = np.column_stack(
    [
        (diagnostics.renderer - thresholds["renderer"]) / (1 - thresholds["renderer"]),
        (diagnostics.pcgrad - thresholds["pcgrad"]) / (1 - thresholds["pcgrad"]),
        (diagnostics.v12 - thresholds["v12"]) / (1 - thresholds["v12"]),
    ]
)
test_support2 = np.sort(test_excess, axis=1)[:, -2]
test_margin3 = test_excess.min(axis=1)
base = diagnostics.base_confidence.to_numpy(float)
mid = np.isclose(base, MID, atol=1e-5)
high = base > MID + 1e-5
core2 = test_support2 >= threshold_2of3
core3 = (test_excess >= 0).all(axis=1) & (test_margin3 >= threshold_3of3_q10)

d1 = base.copy(); d1[mid & core2] = FLOOR
d2 = base.copy(); d2[mid & core3] = FLOOR
d3 = d1.copy(); d3[high & core2] = np.minimum(d3[high & core2], MID)
d4 = base.copy()
dose = np.clip(0.50 - 0.45 * np.clip(test_support2, 0, 1), 0.05, 0.50)
d4[mid & core2] = np.maximum(FLOOR, d4[mid & core2] * dose[mid & core2])
d5 = d1.copy()
safe_high = high & (test_support2 >= threshold_high_q25)
d5[safe_high] = np.minimum(d5[safe_high], MID)

targets = {
    "submission_step10d_D1_mid_2of3_core.csv": d1,
    "submission_step10d_D2_mid_3of3_q10.csv": d2,
    "submission_step10d_D3_full_tiered_cap.csv": d3,
    "submission_step10d_D4_mid_2of3_graded.csv": d4,
    "submission_step10d_D5_safe_high_q25_cap.csv": d5,
}


def render(target: np.ndarray) -> pd.DataFrame:
    result = template.copy()
    cursor = 0
    strings = []
    for prediction in template.prediction_string.fillna(""):
        tokens = str(prediction).strip().split()
        if len(tokens) % 5:
            raise RuntimeError("Malformed prediction string")
        count = len(tokens) // 5
        for candidate, score in enumerate(target[cursor : cursor + count]):
            tokens[5 * candidate] = f"{score:.6f}"
        strings.append(" ".join(tokens) if tokens else " ")
        cursor += count
    if cursor != 3995:
        raise RuntimeError(f"Rendered {cursor} boxes")
    result["prediction_string"] = strings
    return result


audit = {
    "expected_anchor_sha256": EXPECTED_ANCHOR_SHA,
    "rows": 2000,
    "unique_ids": 2000,
    "boxes": 3995,
    "variants": {},
    "competition_submission_created": False,
}
for filename, target in targets.items():
    if np.any(target > base + 1e-12):
        raise RuntimeError(f"Confidence increase in {filename}")
    candidate = render(target)
    path = OUT / filename
    candidate.to_csv(path, index=False)
    changed = target < base - 5e-7
    audit["variants"][filename] = {
        "changed_boxes": int(changed.sum()),
        "removed_confidence_mass": float(np.sum(base - target)),
        "sha256": sha256(path),
        "rows": int(len(candidate)),
        "unique_ids": int(candidate.id.nunique()),
        "boxes_added": 0,
        "boxes_moved": 0,
        "confidence_increases": 0,
    }

(OUT / "candidate_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
(OUT / "final_report.json").write_text(
    json.dumps({"status": "complete", "selection_lock": lock, "candidate_audit": audit}, indent=2),
    encoding="utf-8",
)
print(json.dumps(audit, indent=2))
