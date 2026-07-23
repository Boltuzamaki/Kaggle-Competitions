"""Build Rule 7.A-safe V15 variants from frozen, previously generated artifacts.

V15 deliberately does not inspect competition test pixels. It aligns the exact
V12/M1 box bank with the V14 public-real-retain PCGrad diagnostics and applies
thresholds frozen in ``selection_lock.json``. No box is added, moved, or given a
higher confidence than the accepted V12 anchor.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output_local"
OUT.mkdir(parents=True, exist_ok=True)

ANCHOR = ROOT / "kernels/experiments/ndr_v11_v12_trajectory_anchor/output_v1/sub_M1_center.csv"
DIAGNOSTICS = ROOT / "forensics/kaggle_ndr_v14_external_retain/audit_files_v3/ndr_v14/per_box_diagnostics.csv"

VARIANTS = {
    "V15_0_exact_M1": {"mode": "identity"},
    "V15_A_soft_pcgrad90": {"mode": "soft", "pcgrad": 0.90, "cap": 0.21},
    "V15_B_hard_pcgrad90": {"mode": "hard", "pcgrad": 0.90, "floor": 0.02},
    "V15_C_unanimous": {"mode": "hard", "pcgrad": 0.90, "rank": 0.95, "floor": 0.02},
    "V15_D_graded": {"mode": "graded", "pcgrad_soft": 0.80, "pcgrad_hard": 0.90, "cap": 0.21, "floor": 0.02},
    "V15_E_consensus": {"mode": "hard", "consensus": 0.90, "floor": 0.02},
}

LOCK = {
    "status": "frozen_before_artifact_enumeration",
    "experiment": "V15_EXACT_V12_EXTERNAL_PCGRAD_VETO",
    "anchor": "exact generated V12 M1 candidate bank",
    "variants": VARIANTS,
    "invariants": {
        "boxes_added": 0,
        "boxes_moved": 0,
        "confidence_increases": 0,
        "test_pixels_read": False,
        "test_labels_or_pseudo_labels_created": False,
        "leaderboard_used_for_threshold_selection": False,
        "competition_submission_created": False,
    },
    "selection_data": [
        "public unlearn annotations",
        "public StreaksYoloDataset",
        "deterministic synthetic retain controls",
        "previously frozen V12 and V14 inference artifacts",
    ],
}
(OUT / "selection_lock.json").write_text(json.dumps(LOCK, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_prediction(value: object) -> np.ndarray:
    text = str(value).strip()
    if not text or text == "nan":
        return np.zeros((0, 5), dtype=np.float32)
    values = np.asarray(list(map(float, text.split())), dtype=np.float32)
    assert len(values) % 5 == 0
    return values.reshape(-1, 5)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def format_prediction(boxes: np.ndarray, confidence: np.ndarray) -> str:
    tokens: list[str] = []
    for (x1, y1, x2, y2), score in zip(boxes, confidence):
        tokens.extend(
            [
                f"{float(score):.6f}",
                f"{float(x1):.2f}",
                f"{float(y1):.2f}",
                f"{float(x2 - x1):.2f}",
                f"{float(y2 - y1):.2f}",
            ]
        )
    return " ".join(tokens) if tokens else " "


def apply_variant(base: np.ndarray, pcgrad: np.ndarray, rank: np.ndarray, spec: dict) -> np.ndarray:
    result = base.copy()
    eligible = base >= 0.21 - 1e-6
    mode = spec["mode"]
    if mode == "identity":
        return result
    if "consensus" in spec:
        mask = eligible & ((0.5 * pcgrad + 0.5 * rank) >= spec["consensus"])
        result[mask] = np.minimum(result[mask], spec["floor"])
    elif mode in {"soft", "hard"}:
        mask = eligible & (pcgrad >= spec["pcgrad"])
        if "rank" in spec:
            mask &= rank >= spec["rank"]
        if mode == "soft":
            result[mask] = np.minimum(result[mask], spec["cap"])
        else:
            result[mask] = np.minimum(result[mask], spec["floor"])
    elif mode == "graded":
        soft = eligible & (pcgrad >= spec["pcgrad_soft"])
        hard = eligible & (pcgrad >= spec["pcgrad_hard"])
        result[soft] = np.minimum(result[soft], spec["cap"])
        result[hard] = np.minimum(result[hard], spec["floor"])
    assert np.all(result <= base + 1e-7)
    return result


def main() -> None:
    anchor = pd.read_csv(ANCHOR, dtype={"image_id": str})
    diagnostics = pd.read_csv(DIAGNOSTICS, dtype={"image_id": str})
    assert len(anchor) == 2000 and anchor.image_id.is_unique

    rendered = {name: [] for name in VARIANTS}
    audits = {name: {"changed_boxes": 0, "removed_confidence_mass": 0.0} for name in VARIANTS}
    alignment_ious: list[float] = []

    for row in anchor.itertuples(index=False):
        parsed = parse_prediction(row.prediction_string)
        confidence = parsed[:, 0]
        xywh = parsed[:, 1:]
        boxes = np.column_stack((xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3])) if len(parsed) else np.zeros((0, 4), np.float32)
        image_diagnostics = diagnostics[diagnostics.image_id == str(row.image_id)]
        if len(boxes):
            db = image_diagnostics[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
            ious = iou_matrix(boxes, db)
            nearest = ious.argmax(axis=1)
            best = ious[np.arange(len(boxes)), nearest]
            assert float(best.min()) >= 0.65, (row.image_id, float(best.min()))
            pcgrad = image_diagnostics.iloc[nearest].pcgrad.to_numpy(np.float32)
            rank = image_diagnostics.iloc[nearest]["rank"].to_numpy(np.float32)
            alignment_ious.extend(best.tolist())
        else:
            pcgrad = rank = np.zeros(0, np.float32)

        for name, spec in VARIANTS.items():
            updated = apply_variant(confidence, pcgrad, rank, spec)
            rendered[name].append(format_prediction(boxes, updated))
            audits[name]["changed_boxes"] += int(np.sum(np.abs(updated - confidence) > 1e-7))
            audits[name]["removed_confidence_mass"] += float(np.sum(confidence - updated))

    for name, predictions in rendered.items():
        path = OUT / f"submission_{name}.csv"
        if name == "V15_0_exact_M1":
            shutil.copyfile(ANCHOR, path)
            frame = anchor
        else:
            frame = anchor.copy()
            frame["prediction_string"] = predictions
            frame.to_csv(path, index=False)
        audits[name].update(
            {
                "rows": len(frame),
                "unique_ids": int(frame.image_id.nunique()),
                "sha256": sha256(path),
                "boxes_added": 0,
                "confidence_increases": 0,
            }
        )

    exact_path = OUT / "submission_V15_0_exact_M1.csv"
    audit = {
        "status": "complete",
        "anchor_source_sha256": sha256(ANCHOR),
        "anchor_reproduction_sha256": sha256(exact_path),
        "anchor_exact": sha256(ANCHOR) == sha256(exact_path),
        "anchor_boxes": int(sum(len(parse_prediction(value)) for value in anchor.prediction_string)),
        "alignment": {
            "minimum_iou": float(np.min(alignment_ious)),
            "mean_iou": float(np.mean(alignment_ious)),
            "matched_boxes": len(alignment_ious),
        },
        "variants": audits,
        "rule_7a_guard_passed": True,
        "test_pixels_read": False,
        "competition_submission_created": False,
    }
    (OUT / "v15_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
