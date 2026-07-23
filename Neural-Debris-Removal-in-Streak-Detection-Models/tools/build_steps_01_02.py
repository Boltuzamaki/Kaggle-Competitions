"""Reproduce the frozen V15_B anchor and assemble one row per anchor box.

This is an audit/feature-assembly job. It does not read competition images,
does not fit a model, does not choose thresholds, and does not create a Kaggle
submission. Existing inference artifacts are aligned to the exact 3,995-box
V15_B bank with one frozen rule: maximum IoU >= 0.50 within the same image.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_V15B_SHA256 = (
    "4345387b72aecd55dd3856b1607ed836ec9f400365fcf3f3b89f8947f1ff2412"
)
EXPECTED_ROWS = 2_000
EXPECTED_BOXES = 3_995
EXPECTED_CHANGES = 71
MATCH_IOU = 0.50

ANCHOR = (
    ROOT
    / "forensics/kaggle_ndr_v15_anchor_veto/output_local/"
    "submission_V15_B_hard_pcgrad90.csv"
)
V12_BASE = (
    ROOT
    / "forensics/kaggle_ndr_v15_anchor_veto/output_local/"
    "submission_V15_0_exact_M1.csv"
)
V15_AUDIT = (
    ROOT / "forensics/kaggle_ndr_v15_anchor_veto/output_local/v15_audit.json"
)
V1_CSV = ROOT / "forensics/kaggle_final_submission/output_v1/submission.csv"
NDR229_CSV = ROOT / "forensics/kaggle_ndr229_exact_gpu/remote_v4_complete/submission.csv"
V10_NPZ = (
    ROOT
    / "kernels/experiments/ndr_v10_ensemble/output_v1/ndr_v10/"
    "per_box_diagnostics.npz"
)
V21_CSV = ROOT / "forensics/kaggle_ndr_v21_orthogonal_consensus/per_box_diagnostics.csv"
V19_CSV = (
    ROOT
    / "forensics/kaggle_ndr_v19_pcgrad_stability/output_v1/ndr_v19/"
    "per_box_diagnostics.csv"
)
V13_CSV = (
    ROOT
    / "forensics/kaggle_ndr_v13_breakthrough/output_v2_full/ndr_v13/"
    "per_box_diagnostics.csv"
)
TRACE_0 = (
    ROOT
    / "forensics/kaggle_ndr_v20_trace_shard0/output_v2/ndr_v20_shard0/"
    "trace_diagnostics_shard0.csv"
)
TRACE_1 = (
    ROOT
    / "forensics/kaggle_ndr_v20_trace_shard1/output_v2/ndr_v20_shard1/"
    "trace_diagnostics_shard1.csv"
)
RENDERER_CSV = (
    ROOT
    / "forensics/kaggle_ndr_v22_renderer_fingerprint/output_v1/ndr_v22/"
    "per_box_diagnostics.csv"
)
BOUNDARY_CSV = (
    ROOT
    / "forensics/kaggle_ndr_v23_boundary_margin/output_v1/ndr_v23/"
    "per_box_diagnostics.csv"
)
RAWPIXEL_CSV = (
    ROOT
    / "forensics/kaggle_ndr_v17_rawpixel_clean_gate/output_v1_retry/ndr_v17/"
    "per_box_diagnostics.csv"
)
CLEANHEAD_CSV = (
    ROOT
    / "forensics/kaggle_ndr_v18_canonical_recovery/output_v3/ndr_v18/"
    "per_box_diagnostics.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_prediction(value: object) -> np.ndarray:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.zeros((0, 5), dtype=np.float32)
    values = np.asarray([float(token) for token in text.split()], dtype=np.float32)
    if len(values) % 5:
        raise ValueError(f"Prediction string has {len(values)} tokens")
    return values.reshape(-1, 5)


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    if not len(xywh):
        return np.zeros((0, 4), dtype=np.float32)
    return np.column_stack(
        (xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3])
    ).astype(np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[:, :, 0] * size[:, :, 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return intersection / np.maximum(
        area_a[:, None] + area_b[None, :] - intersection, 1e-6
    )


def load_submission(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frame = pd.read_csv(path, dtype={"image_id": str}, keep_default_na=False)
    if len(frame) != EXPECTED_ROWS or not frame.image_id.is_unique:
        raise AssertionError(f"Invalid submission structure: {path}")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in frame.itertuples(index=False):
        parsed = parse_prediction(row.prediction_string)
        result[str(row.image_id)] = (xywh_to_xyxy(parsed[:, 1:]), parsed[:, 0])
    return result


def match_scores(
    reference_boxes: np.ndarray,
    source_boxes: np.ndarray,
    source_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source scores and IoU for each reference box.

    A zero score is semantically meaningful: the source detector did not retain
    a matching detection. The IoU column distinguishes a true unmatched zero.
    """
    scores = np.zeros(len(reference_boxes), dtype=np.float32)
    best_iou = np.zeros(len(reference_boxes), dtype=np.float32)
    if len(reference_boxes) and len(source_boxes):
        matrix = iou_matrix(reference_boxes, source_boxes)
        nearest = matrix.argmax(axis=1)
        best_iou = matrix[np.arange(len(reference_boxes)), nearest]
        matched = best_iou >= MATCH_IOU
        scores[matched] = source_scores[nearest[matched]]
    return scores, best_iou


def load_npz_by_image(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as payload:
        stems = payload["stems"].astype(str)
        counts = payload["counts"].astype(int)
        arrays = {
            key: np.asarray(payload[key])
            for key in ("boxes", "scores", "s_diff", "dash", "lin")
        }
    expected = int(counts.sum())
    for key, value in arrays.items():
        if value.shape[0] != expected:
            raise AssertionError((path, key, value.shape, expected))
    result: dict[str, dict[str, np.ndarray]] = {}
    start = 0
    for stem, count in zip(stems, counts):
        end = start + int(count)
        result[str(stem)] = {key: value[start:end] for key, value in arrays.items()}
        start = end
    return result


def assert_candidate_table(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["image_id"] = frame.image_id.astype(str)
    key = frame[["image_id", "candidate"]]
    if len(frame) != EXPECTED_BOXES or key.duplicated().any():
        raise AssertionError(f"{name} is not an exact 3,995-row candidate table")
    return frame.sort_values(["image_id", "candidate"], key=lambda s: s.map(lambda x: int(x)))


def assert_sparse_candidate_table(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["image_id"] = frame.image_id.astype(str)
    if frame[["image_id", "candidate"]].duplicated().any():
        raise AssertionError(f"{name} contains duplicate candidate keys")
    return frame.sort_values(["image_id", "candidate"], key=lambda s: s.map(lambda x: int(x)))


def main() -> None:
    required = [
        ANCHOR,
        V12_BASE,
        V15_AUDIT,
        V1_CSV,
        NDR229_CSV,
        V10_NPZ,
        V21_CSV,
        V19_CSV,
        V13_CSV,
        TRACE_0,
        TRACE_1,
        RENDERER_CSV,
        BOUNDARY_CSV,
        RAWPIXEL_CSV,
        CLEANHEAD_CSV,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    anchor_hash = sha256(ANCHOR)
    if anchor_hash != EXPECTED_V15B_SHA256:
        raise AssertionError((anchor_hash, EXPECTED_V15B_SHA256))
    anchor = pd.read_csv(ANCHOR, dtype={"image_id": str}, keep_default_na=False)
    base = pd.read_csv(V12_BASE, dtype={"image_id": str}, keep_default_na=False)
    if len(anchor) != EXPECTED_ROWS or not anchor.image_id.is_unique:
        raise AssertionError("V15_B must contain 2,000 unique image IDs")
    if not anchor.image_id.equals(base.image_id):
        raise AssertionError("V15_B and V12 base image order differs")

    anchor_arrays = [parse_prediction(value) for value in anchor.prediction_string]
    base_arrays = [parse_prediction(value) for value in base.prediction_string]
    box_count = sum(len(array) for array in anchor_arrays)
    if box_count != EXPECTED_BOXES:
        raise AssertionError((box_count, EXPECTED_BOXES))

    changed = 0
    added = moved = increased = 0
    anchor_records: list[dict[str, object]] = []
    for image_id, current, prior in zip(anchor.image_id, anchor_arrays, base_arrays):
        if len(current) != len(prior):
            added += max(0, len(current) - len(prior))
            raise AssertionError(f"Box count changed for image {image_id}")
        current_boxes = xywh_to_xyxy(current[:, 1:])
        prior_boxes = xywh_to_xyxy(prior[:, 1:])
        moved += int(np.sum(np.max(np.abs(current_boxes - prior_boxes), axis=1) > 1e-4))
        increased += int(np.sum(current[:, 0] > prior[:, 0] + 1e-7))
        changed += int(np.sum(np.abs(current[:, 0] - prior[:, 0]) > 1e-7))
        for candidate, row in enumerate(current):
            x, y, width, height = row[1:]
            anchor_records.append(
                {
                    "image_id": str(image_id),
                    "box_id": candidate,
                    "v15b_confidence": float(row[0]),
                    "x1": float(x),
                    "y1": float(y),
                    "x2": float(x + width),
                    "y2": float(y + height),
                }
            )
    if (changed, added, moved, increased) != (EXPECTED_CHANGES, 0, 0, 0):
        raise AssertionError((changed, added, moved, increased))

    control = OUT / "control_v15b.csv"
    shutil.copyfile(ANCHOR, control)
    if sha256(control) != EXPECTED_V15B_SHA256:
        raise AssertionError("Control copy is not byte-identical")
    step01 = {
        "status": "pass",
        "source": str(ANCHOR.relative_to(ROOT)),
        "control": str(control.relative_to(ROOT)),
        "sha256": sha256(control),
        "byte_identical": True,
        "images": EXPECTED_ROWS,
        "unique_ids": int(anchor.image_id.nunique()),
        "candidate_boxes": box_count,
        "added_boxes": added,
        "moved_boxes": moved,
        "confidence_increases": increased,
        "v15b_suppressions": changed,
        "known_public_score": 213.7088,
        "competition_submission_created": False,
    }
    (OUT / "step01_anchor_audit.json").write_text(
        json.dumps(step01, indent=2), encoding="utf-8"
    )

    table = pd.DataFrame(anchor_records)
    v1 = load_submission(V1_CSV)
    ndr = load_submission(NDR229_CSV)
    v10 = load_npz_by_image(V10_NPZ)
    v21 = assert_candidate_table(pd.read_csv(V21_CSV), "V21")
    v19 = assert_candidate_table(pd.read_csv(V19_CSV), "V19")
    trace = assert_sparse_candidate_table(
        pd.concat([pd.read_csv(TRACE_0), pd.read_csv(TRACE_1)], ignore_index=True),
        "TRACE",
    )
    renderer = assert_candidate_table(pd.read_csv(RENDERER_CSV), "renderer")
    boundary = assert_candidate_table(pd.read_csv(BOUNDARY_CSV), "boundary")
    rawpixel = assert_candidate_table(pd.read_csv(RAWPIXEL_CSV), "rawpixel")
    cleanhead = assert_candidate_table(pd.read_csv(CLEANHEAD_CSV), "cleanhead")
    v13 = pd.read_csv(V13_CSV, dtype={"image_id": str})
    v13_by = {key: group.reset_index(drop=True) for key, group in v13.groupby("image_id")}

    exact_tables = {
        "v21": v21,
        "v19": v19,
        "renderer": renderer,
        "boundary": boundary,
        "rawpixel": rawpixel,
        "cleanhead": cleanhead,
    }
    for name, frame in exact_tables.items():
        frame["join_key"] = frame.image_id.astype(str) + ":" + frame.candidate.astype(str)
        if set(frame.join_key) != set(table.image_id + ":" + table.box_id.astype(str)):
            raise AssertionError(f"{name} candidate IDs do not match the anchor")
    table["join_key"] = table.image_id + ":" + table.box_id.astype(str)
    trace["join_key"] = trace.image_id.astype(str) + ":" + trace.candidate.astype(str)
    if not set(trace.join_key).issubset(set(table.join_key)):
        raise AssertionError("TRACE contains candidates outside the exact anchor bank")

    v1_scores: list[float] = []
    v1_ious: list[float] = []
    ndr_scores: list[float] = []
    ndr_ious: list[float] = []
    v10_ratio: list[float] = []
    v10_replica_ratios: list[np.ndarray] = []
    dashedness: list[float] = []
    linearity: list[float] = []
    v13_projection: list[float] = []
    v13_task: list[float] = []
    v13_ious: list[float] = []

    for image_id, group in table.groupby("image_id", sort=False):
        indices = group.index.to_numpy()
        boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        scores, ious = match_scores(boxes, *v1[image_id])
        v1_scores.extend(scores.tolist())
        v1_ious.extend(ious.tolist())
        scores, ious = match_scores(boxes, *ndr[image_id])
        ndr_scores.extend(scores.tolist())
        ndr_ious.extend(ious.tolist())

        bank = v10[image_id]
        matrix = iou_matrix(boxes, bank["boxes"])
        nearest = matrix.argmax(axis=1)
        best = matrix[np.arange(len(boxes)), nearest]
        if float(best.min()) < MATCH_IOU:
            raise AssertionError(("V10", image_id, float(best.min())))
        collapse = np.asarray(bank["s_diff"])[nearest]
        ratios = np.clip(1.0 - collapse, 0.0, 2.5)
        v10_ratio.extend(np.mean(ratios, axis=1).tolist())
        v10_replica_ratios.extend(ratios)
        dashedness.extend(np.asarray(bank["dash"])[nearest].tolist())
        linearity.extend(np.asarray(bank["lin"])[nearest].tolist())

        source = v13_by[image_id]
        source_boxes = source[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        matrix = iou_matrix(boxes, source_boxes)
        nearest = matrix.argmax(axis=1)
        best = matrix[np.arange(len(boxes)), nearest]
        if float(best.min()) < MATCH_IOU:
            raise AssertionError(("V13", image_id, float(best.min())))
        v13_projection.extend(source.iloc[nearest].projection_poison.astype(float).tolist())
        v13_task.extend(source.iloc[nearest].task_poison.astype(float).tolist())
        v13_ious.extend(best.tolist())

    # Groupby preserves first-appearance order because sort=False. The anchor is
    # ordered by image and candidate, so these vectors map directly to `table`.
    table["v1_confidence"] = v1_scores
    table["v1_match_iou"] = v1_ious
    table["ndr229_confidence"] = ndr_scores
    table["ndr229_match_iou"] = ndr_ious
    table["v10_mean_ratio"] = v10_ratio
    replica_matrix = np.asarray(v10_replica_ratios, dtype=np.float32)
    for replica in range(replica_matrix.shape[1]):
        table[f"v10_replica_{replica}_ratio"] = replica_matrix[:, replica]
    table["dashedness"] = dashedness
    table["linearity"] = linearity
    table["v13_projection_poison"] = v13_projection
    table["v13_task_poison"] = v13_task
    table["v13_match_iou"] = v13_ious

    merge_columns = {
        "v21": [
            "v9_survivor",
            "v12_survivor",
            "v9_amplifier",
            "v12_prototype",
            "v19_pcgrad",
        ],
        "v19": ["original_model_confidence", "std_poison"],
        "renderer": ["renderer_probability"],
        "boundary": ["boundary_probability"],
        "rawpixel": ["raw_poison_probability"],
        "cleanhead": ["clean_probability"],
    }
    for name, columns in merge_columns.items():
        source = exact_tables[name].set_index("join_key")
        for column in columns:
            table[column] = table.join_key.map(source[column]).astype(float)

    trace_columns = [
        "trace_probability",
        "context_mean",
        "context_std",
        "context_fire",
        "focal_mean",
        "context_focal_gap",
    ]
    trace_source = trace.set_index("join_key")
    table["trace_observed"] = table.join_key.isin(trace_source.index)
    for column in trace_columns:
        table[column] = table.join_key.map(trace_source[column]).astype(float)

    original = np.maximum(table.original_model_confidence.to_numpy(float), 1e-6)
    table["v1_ratio"] = np.clip(table.v1_confidence / original, 0.0, 2.5)
    table["ndr229_ratio"] = np.clip(table.ndr229_confidence / original, 0.0, 2.5)
    table["v9_adversarial_ratio"] = np.clip(1.0 - table.v9_survivor, 0.0, 2.5)
    table["v12_ratio"] = np.clip(1.0 - table.v12_survivor, 0.0, 2.5)
    table["pcgrad_median_ratio"] = np.clip(1.0 - table.v19_pcgrad, 0.0, 2.5)

    ratio_columns = [
        "v1_ratio",
        "ndr229_ratio",
        "v10_mean_ratio",
        "v12_ratio",
        "pcgrad_median_ratio",
        "v9_adversarial_ratio",
    ]
    ratio_matrix = table[ratio_columns].to_numpy(float)
    table["survivor_min_ratio"] = np.min(ratio_matrix, axis=1)
    table["survivor_median_ratio"] = np.median(ratio_matrix, axis=1)
    table["survivor_ratio_std"] = np.std(ratio_matrix, axis=1)
    table["collapse_votes_035"] = np.sum(ratio_matrix < 0.35, axis=1)
    table["collapse_votes_050"] = np.sum(ratio_matrix < 0.50, axis=1)

    ordered = [
        "image_id",
        "box_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "original_model_confidence",
        "v15b_confidence",
        *ratio_columns,
        "v10_replica_0_ratio",
        "v10_replica_1_ratio",
        "v10_replica_2_ratio",
        "survivor_min_ratio",
        "survivor_median_ratio",
        "survivor_ratio_std",
        "collapse_votes_035",
        "collapse_votes_050",
        "v9_amplifier",
        "v12_prototype",
        "dashedness",
        "linearity",
        "trace_probability",
        "trace_observed",
        "context_mean",
        "context_std",
        "context_fire",
        "focal_mean",
        "context_focal_gap",
        "renderer_probability",
        "boundary_probability",
        "raw_poison_probability",
        "clean_probability",
        "v13_projection_poison",
        "v13_task_poison",
        "v1_match_iou",
        "ndr229_match_iou",
        "v13_match_iou",
    ]
    output_table = table[ordered].sort_values(
        ["image_id", "box_id"], key=lambda s: s.map(lambda value: int(value))
    )
    if len(output_table) != EXPECTED_BOXES:
        raise AssertionError(len(output_table))
    table_path = OUT / "candidate_probe_table.csv"
    output_table.to_csv(table_path, index=False)

    source_coverage = {
        "v1": {
            "matched": int(np.sum(output_table.v1_match_iou >= MATCH_IOU)),
            "total": EXPECTED_BOXES,
        },
        "ndr229": {
            "matched": int(np.sum(output_table.ndr229_match_iou >= MATCH_IOU)),
            "total": EXPECTED_BOXES,
        },
        "v10": {"matched": EXPECTED_BOXES, "total": EXPECTED_BOXES},
        "v12": {"matched": EXPECTED_BOXES, "total": EXPECTED_BOXES},
        "pcgrad": {"matched": EXPECTED_BOXES, "total": EXPECTED_BOXES},
        "v9_adversarial": {"matched": EXPECTED_BOXES, "total": EXPECTED_BOXES},
        "v13_projection": {"matched": EXPECTED_BOXES, "total": EXPECTED_BOXES},
        "trace": {
            "matched": int(output_table.trace_observed.sum()),
            "total": EXPECTED_BOXES,
            "missing": int((~output_table.trace_observed).sum()),
            "reason": "V20 measured only its frozen eligibility subset",
        },
        "renderer": {"matched": EXPECTED_BOXES, "total": EXPECTED_BOXES},
        "boundary_margin": {"matched": EXPECTED_BOXES, "total": EXPECTED_BOXES},
    }
    audit = {
        "status": "pass_core_probes_trace_sparse",
        "frozen_match_rule": {"same_image": True, "maximum_iou_at_least": MATCH_IOU},
        "rows": len(output_table),
        "unique_images": int(output_table.image_id.nunique()),
        "unique_candidate_keys": int(
            output_table[["image_id", "box_id"]].drop_duplicates().shape[0]
        ),
        "source_coverage": source_coverage,
        "ratio_columns": ratio_columns,
        "feature_columns": ordered[6:],
        "minimum_exact_source_iou": {
            "v10": MATCH_IOU,
            "v13": float(output_table.v13_match_iou.min()),
        },
        "unmatched_exported_detection_policy": (
            "ratio=0; match IoU retained so downstream validation can audit it"
        ),
        "test_pixels_read": False,
        "test_labels_or_pseudo_labels_created": False,
        "model_or_ranker_fit": False,
        "selection_or_threshold_tuning_performed": False,
        "competition_submission_created": False,
        "table_sha256": sha256(table_path),
    }
    (OUT / "step02_probe_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps({"step01": step01, "step02": audit}, indent=2))


if __name__ == "__main__":
    main()
