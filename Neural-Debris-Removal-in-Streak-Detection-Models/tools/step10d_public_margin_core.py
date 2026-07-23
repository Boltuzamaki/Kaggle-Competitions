"""Build public-calibrated Step10D candidates without test-derived tuning.

The calibration stage reads only Step10C nested public validation outputs.  It
writes the selection lock before opening the frozen test-inference diagnostics.
Candidate generation is then a deterministic application of those public
thresholds to the exact V15_B 3,995-box bank.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STEP10C = ROOT / "local_validation/kaggle_step10c_stress_consensus_inference/output_v1/step10c_stress_consensus"
ANCHOR = ROOT / "forensics/kaggle_ndr_v15_anchor_veto/output_local/submission_V15_B_hard_pcgrad90.csv"
OUT = ROOT / "local_validation/step10d_public_margin_core"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_ANCHOR_SHA = "4345387b72aecd55dd3856b1607ed836ec9f400365fcf3f3b89f8947f1ff2412"
MID_TIER = 0.21
FLOOR = 0.02


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_excess(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            (frame.renderer_score - frame.renderer_threshold) / (1.0 - frame.renderer_threshold),
            (frame.pcgrad_score - frame.pcgrad_threshold) / (1.0 - frame.pcgrad_threshold),
            (frame.v12_score - frame.v12_threshold) / (1.0 - frame.v12_threshold),
        ]
    )


# Phase A: public-only calibration.  No test diagnostics are opened above this line.
public = pd.read_csv(STEP10C / "nested_stress_consensus_predictions.csv")
public_y = public.label_poison.to_numpy(bool)
public_excess = normalized_excess(public)
public_support2 = np.sort(public_excess, axis=1)[:, -2]
public_margin3 = public_excess.min(axis=1)
public_three_positive = (public_excess >= 0.0).all(axis=1)

# Broad core: the smallest representable value strictly above every public-clean
# 2-of-3 support.  This produces zero public-clean false positives.
two_of_three_threshold = float(np.nextafter(public_support2[~public_y].max(), np.inf))

# Conservative core: retain the strongest 90% of already-correct public poison
# consensus margins.  The quantile and rule are fixed independently of test data.
three_positive_poison_margins = public_margin3[public_y & public_three_positive]
three_of_three_q10_threshold = float(np.quantile(three_positive_poison_margins, 0.10))


def public_metrics(prediction: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(prediction & public_y))
    fp = int(np.sum(prediction & ~public_y))
    return {
        "true_positives": tp,
        "false_positives": fp,
        "precision": float(tp / max(tp + fp, 1)),
        "poison_recall": float(tp / max(int(public_y.sum()), 1)),
    }


two_public = public_support2 >= two_of_three_threshold
three_public = public_three_positive & (public_margin3 >= three_of_three_q10_threshold)
lock = {
    "status": "frozen_before_test_diagnostics_read",
    "experiment": "STEP10D_PUBLIC_MARGIN_CORE",
    "source": "Step10C nested public-only validation",
    "anchor_sha256": EXPECTED_ANCHOR_SHA,
    "fixed_bank": {"rows": 2000, "boxes": 3995, "mid_tier": MID_TIER, "floor": FLOOR},
    "variants": {
        "D1_two_of_three_public_clean_max": {
            "support": "second-largest normalized threshold excess",
            "threshold": two_of_three_threshold,
            "threshold_rule": "nextafter(maximum nested public-clean support, +infinity)",
            "public_metrics": public_metrics(two_public),
        },
        "D2_three_of_three_public_poison_q10": {
            "support": "minimum normalized threshold excess",
            "threshold": three_of_three_q10_threshold,
            "threshold_rule": "10th percentile of correctly detected nested public-poison margins",
            "public_metrics": public_metrics(three_public),
        },
        "D3_two_of_three_tiered_cap": {
            "support": "same public-clean-max 2-of-3 core as D1",
            "threshold": two_of_three_threshold,
            "threshold_rule": "nextafter(maximum nested public-clean support, +infinity)",
            "confidence_rule": "mid-tier 0.21 becomes 0.02; higher tiers are capped at 0.21",
            "public_metrics": public_metrics(two_public),
        },
    },
    "invariants": {
        "mid_tier_only": True,
        "boxes_added": 0,
        "boxes_moved": 0,
        "confidence_increases": 0,
        "test_derived_thresholds": False,
        "leaderboard_derived_thresholds": False,
        "competition_submission_created": False,
    },
}
(OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")

# Phase B: deterministic frozen inference on the already-produced diagnostics.
assert sha256(ANCHOR) == EXPECTED_ANCHOR_SHA
diagnostics = pd.read_csv(STEP10C / "per_box_diagnostics.csv")
thresholds = json.loads((STEP10C / "deployment_thresholds.json").read_text(encoding="utf-8"))
test_excess = np.column_stack(
    [
        (diagnostics.renderer.to_numpy() - thresholds["renderer"]) / (1.0 - thresholds["renderer"]),
        (diagnostics.pcgrad.to_numpy() - thresholds["pcgrad"]) / (1.0 - thresholds["pcgrad"]),
        (diagnostics.v12.to_numpy() - thresholds["v12"]) / (1.0 - thresholds["v12"]),
    ]
)
test_support2 = np.sort(test_excess, axis=1)[:, -2]
test_margin3 = test_excess.min(axis=1)
base_confidence = diagnostics.base_confidence.to_numpy(float)
mid = np.isclose(base_confidence, MID_TIER, atol=1e-5)
two_core = test_support2 >= two_of_three_threshold
three_core = (test_excess >= 0.0).all(axis=1) & (test_margin3 >= three_of_three_q10_threshold)

d1_target = base_confidence.copy()
d1_target[mid & two_core] = FLOOR
d2_target = base_confidence.copy()
d2_target[mid & three_core] = FLOOR
d3_target = base_confidence.copy()
d3_target[mid & two_core] = FLOOR
d3_target[(base_confidence > MID_TIER + 1e-5) & two_core] = np.minimum(
    d3_target[(base_confidence > MID_TIER + 1e-5) & two_core], MID_TIER
)

targets = {
    "submission_step10d_D1_two_of_three_core.csv": d1_target,
    "submission_step10d_D2_three_of_three_q10.csv": d2_target,
    "submission_step10d_D3_two_of_three_tiered_cap.csv": d3_target,
}

anchor = pd.read_csv(ANCHOR, dtype={"image_id": str})


def replace_scores(target_scores: np.ndarray) -> pd.DataFrame:
    result = anchor.copy()
    cursor = 0
    strings: list[str] = []
    for prediction in result.prediction_string.fillna(""):
        tokens = str(prediction).strip().split()
        if len(tokens) % 5:
            raise ValueError("Malformed prediction string")
        boxes_here = len(tokens) // 5
        local = target_scores[cursor : cursor + boxes_here]
        for candidate, target in enumerate(local):
            current = float(tokens[5 * candidate])
            if target < current - 5e-7:
                tokens[5 * candidate] = f"{target:.6f}"
        strings.append(" ".join(tokens) if tokens else " ")
        cursor += boxes_here
    if cursor != len(target_scores):
        raise ValueError(f"Box count mismatch: {cursor} != {len(target_scores)}")
    result["prediction_string"] = strings
    return result


audit: dict[str, object] = {
    "anchor_sha256": EXPECTED_ANCHOR_SHA,
    "rows": int(len(anchor)),
    "unique_ids": int(anchor.id.nunique()),
    "boxes": int(len(diagnostics)),
    "candidates": {},
    "competition_submission_created": False,
}
for filename, target_scores in targets.items():
    changed = target_scores < base_confidence - 5e-7
    candidate = replace_scores(target_scores)
    path = OUT / filename
    candidate.to_csv(path, index=False)
    audit["candidates"][filename] = {
        "changed_boxes": int(changed.sum()),
        "removed_confidence_mass": float(np.sum(base_confidence - target_scores)),
        "sha256": sha256(path),
        "rows": int(len(candidate)),
        "unique_ids": int(candidate.id.nunique()),
        "boxes_added": 0,
        "boxes_moved": 0,
        "confidence_increases": 0,
    }

(OUT / "candidate_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps({"selection": lock["variants"], "candidate_audit": audit}, indent=2))
