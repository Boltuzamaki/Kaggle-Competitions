# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Debris V7 - V1-centered recovery frontier
#
# Local preparation only. This notebook is intentionally research-only:
#
# - the failed E28 weight-pruning experiment is repaired and rerun;
# - E51-E54 explore the V1-centered inference and checkpoint frontier;
# - E43-E50 remain completed evidence and are not rerun;
# - only the supplied model, public unlearn set and frozen V1 checkpoints are used;
# - no competition test path is read;
# - no submission CSV or Kaggle API call is present;
# - every candidate is evaluated by the same suppression/retention gate;
# - the exact public maCADD formula is used only against a public pseudo-clean
#   reference and is not claimed to equal the hidden leaderboard metric.

# %%
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.0")

OUT = Path("/kaggle/working/v1_centered_frontier_v7")
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run.log"


def log(message, **fields):
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": str(message), **fields}
    print(f"[{row['time']}] {message}", flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


@contextlib.contextmanager
def heartbeat(label, seconds=30):
    stop = threading.Event()
    started = time.time()

    def worker():
        while not stop.wait(seconds):
            log("HEARTBEAT", label=label, elapsed_sec=round(time.time() - started, 1))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


log("Installing P100-compatible PyTorch and Detectron2")
with heartbeat("runtime installation"):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "--force-reinstall",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "setuptools<81"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/facebookresearch/detectron2.git",
        ],
        check=True,
    )

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("V5 requires a Kaggle GPU")

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
UNLEARN = ROOT / "unlearn_set"
WEIGHTS = ROOT / "poisoned_model/poisoned_model.pth"
SEED = 20260718
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FOLDS = {
    0: [15, 232, 428, 781],
    1: [104, 255, 523, 815],
    2: [108, 374, 592, 864],
    3: [147, 375, 610, 935],
    4: [200, 410, 767, 938],
}
ALL_IDS = sorted(sum(FOLDS.values(), []))

GATE = {
    "poison_fire_rate_020_max": 0.35,
    "poison_score_ratio_median_max": 0.25,
    "retain_match_rate_min": 0.90,
    "retain_score_ratio_median_min": 0.80,
    "retain_score_ratio_median_max": 1.20,
}

CONFIG = {
    "seed": SEED,
    "experiments": ["E28F", "E51", "E52", "E53", "E54"],
    "frozen_behavioral_anchor": "V1 step 200; Kaggle score 300.8064",
    "failed_historical_reference": "E33; Kaggle score 398.0498",
    "completed_not_rerun": [f"E{index}" for index in range(43, 51)],
    "disabled_experiments": [],
    "selection_data": "public unlearn images and within-set controls only",
    "test_data_read": False,
    "submission_generation": False,
    "participant_confidence_filter_in_public_metric": None,
    "hidden_clean_confidence_filter": "> 0.20",
    "selection_policy": (
        "No proxy winner. Produce a diverse V1-centered public-unlearn frontier "
        "for the next predeclared submission shortlist."
    ),
    "historical_gate_retained_for_diagnostics_only": GATE,
}
(OUT / "v7_config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")

digest = hashlib.sha256()
with WEIGHTS.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
(OUT / "original_weights_sha256.txt").write_text(digest.hexdigest() + "\n", encoding="utf-8")

with (UNLEARN / "annotations_coco.json").open(encoding="utf-8") as handle:
    coco = json.load(handle)
image_info = {int(item["id"]): item for item in coco["images"]}
annotation_by_image = {int(item["image_id"]): item for item in coco["annotations"]}
assert sorted(image_info) == ALL_IDS


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    scale = 65535.0 if image.dtype == np.uint16 else max(float(image.max()), 1.0)
    return np.clip(image.astype(np.float32) / scale * 255.0, 0, 255)


IMAGES = {
    image_id: read_gray(UNLEARN / image_info[image_id]["file_name"])
    for image_id in ALL_IDS
}
POISON_BOXES = {}
for image_id, annotation in annotation_by_image.items():
    x, y, width, height = map(float, annotation["bbox"])
    POISON_BOXES[image_id] = np.asarray([x, y, x + width, y + height], np.float32)


def make_cfg(score_threshold=0.001):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = str(WEIGHTS)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(score_threshold)
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    return cfg


def build_original(score_threshold=0.001):
    model = build_model(make_cfg(score_threshold))
    DetectionCheckpointer(model).load(str(WEIGHTS))
    model.to(DEVICE)
    model.eval()
    return model


def to_record(gray):
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return {"image": tensor, "height": gray.shape[0], "width": gray.shape[1]}


def infer(model, gray):
    model.eval()
    score_module = getattr(model.head, "cls_score", None)
    if hasattr(score_module, "reset"):
        score_module.reset()
    with torch.no_grad():
        output = model([to_record(gray)])[0]["instances"].to("cpu")
    return (
        output.pred_boxes.tensor.numpy().astype(np.float32),
        output.scores.numpy().astype(np.float32),
    )


def predictions_for_model(model, image_ids=ALL_IDS):
    return {image_id: infer(model, IMAGES[image_id]) for image_id in image_ids}


def box_iou(box, candidates):
    candidates = np.asarray(candidates, np.float32).reshape(-1, 4)
    if len(candidates) == 0:
        return np.zeros(0, np.float32)
    top_left = np.maximum(np.asarray(box)[None, :2], candidates[:, :2])
    bottom_right = np.minimum(np.asarray(box)[None, 2:], candidates[:, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[:, 0] * size[:, 1]
    area_a = max(float(np.prod(np.asarray(box)[2:] - np.asarray(box)[:2])), 1e-6)
    area_b = np.prod(np.clip(candidates[:, 2:] - candidates[:, :2], 0, None), axis=1)
    return intersection / np.clip(area_a + area_b - intersection, 1e-6, None)


TEACHER = build_original(0.001)
for parameter in TEACHER.parameters():
    parameter.requires_grad = False
TEACHER_PREDICTIONS = predictions_for_model(TEACHER)
ORIGINAL_STATE = {name: value.detach().cpu().clone() for name, value in TEACHER.state_dict().items()}
log("Original dense box bank ready", images=len(TEACHER_PREDICTIONS))

# %% [markdown]
# ## Exact public metric semantics
#
# The public scorer filters hidden-clean detections at confidence `> 0.20`.
# Participant detections are parsed and retained at every confidence. E43-E46
# exploit that asymmetry only through public-unlearn selection.

# %%
IOU_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ASYMMETRY = 10.0


def iou_matrix(boxes_a, boxes_b):
    boxes_a = np.asarray(boxes_a, np.float32).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, np.float32).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[:, :, 0] * size[:, :, 1]
    area_a = np.prod(np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / union, 0).astype(np.float32)


def greedy_pairs(matrix, threshold):
    rows, columns = np.where(matrix >= threshold)
    if len(rows) == 0:
        return []
    order = np.argsort(matrix[rows, columns])[::-1]
    used_rows, used_columns, pairs = set(), set(), []
    for row, column in zip(rows[order], columns[order]):
        if int(row) in used_rows or int(column) in used_columns:
            continue
        used_rows.add(int(row))
        used_columns.add(int(column))
        pairs.append((int(row), int(column)))
    return pairs


def public_acadd(reference_boxes, reference_scores, candidate_boxes, candidate_scores, threshold):
    matrix = iou_matrix(reference_boxes, candidate_boxes)
    pairs = greedy_pairs(matrix, threshold)
    matched_reference = {pair[0] for pair in pairs}
    matched_candidate = {pair[1] for pair in pairs}
    score = 0.0
    for reference_index, candidate_index in pairs:
        difference = float(reference_scores[reference_index]) - float(candidate_scores[candidate_index])
        score += abs(difference) / ASYMMETRY if difference > 0 else abs(difference)
    for reference_index in range(len(reference_scores)):
        if reference_index not in matched_reference:
            score += float(reference_scores[reference_index])
    for candidate_index in range(len(candidate_scores)):
        if candidate_index not in matched_candidate:
            if len(reference_boxes) == 0 or matrix[:, candidate_index].max() == 0:
                score += float(candidate_scores[candidate_index])
    return score


def pseudo_clean_reference():
    output = {}
    for image_id in ALL_IDS:
        reference_boxes, reference_scores = TEACHER_PREDICTIONS[image_id]
        overlap = box_iou(POISON_BOXES[image_id], reference_boxes)
        keep = (reference_scores > 0.20) & (overlap < 0.10)
        output[image_id] = (reference_boxes[keep].copy(), reference_scores[keep].copy())
    return output


PSEUDO_CLEAN = pseudo_clean_reference()


def pseudo_clean_macadd(predictions):
    totals = {threshold: 0.0 for threshold in IOU_THRESHOLDS}
    for image_id in ALL_IDS:
        reference_boxes, reference_scores = PSEUDO_CLEAN[image_id]
        candidate_boxes, candidate_scores = predictions[image_id]
        for threshold in IOU_THRESHOLDS:
            totals[threshold] += public_acadd(
                reference_boxes,
                reference_scores,
                candidate_boxes,
                candidate_scores,
                threshold,
            )
    return float(
        sum(threshold * totals[threshold] for threshold in IOU_THRESHOLDS)
        / sum(IOU_THRESHOLDS)
    )


def metrics_from_predictions(predictions, image_ids=ALL_IDS):
    poison_ratios, poison_scores = [], []
    retain_ratios = []
    retain_total = 0
    retain_matched = 0
    count_ratios = []
    for image_id in image_ids:
        reference_boxes, reference_scores = TEACHER_PREDICTIONS[image_id]
        candidate_boxes, candidate_scores = predictions[image_id]
        target_box = POISON_BOXES[image_id]

        reference_overlap = box_iou(target_box, reference_boxes)
        candidate_overlap = box_iou(target_box, candidate_boxes)
        reference_target_score = float(
            reference_scores[reference_overlap >= 0.20].max(initial=0)
        )
        candidate_target_score = float(
            candidate_scores[candidate_overlap >= 0.20].max(initial=0)
        )
        poison_scores.append(candidate_target_score)
        poison_ratios.append(candidate_target_score / max(reference_target_score, 1e-6))

        keep_reference = (reference_scores >= 0.20) & (reference_overlap < 0.10)
        retained_reference_boxes = reference_boxes[keep_reference]
        retained_reference_scores = reference_scores[keep_reference]
        retain_total += len(retained_reference_boxes)
        for reference_box, reference_score in zip(retained_reference_boxes, retained_reference_scores):
            overlaps = box_iou(reference_box, candidate_boxes)
            if len(overlaps) and overlaps.max() >= 0.50:
                index = int(overlaps.argmax())
                retain_matched += 1
                retain_ratios.append(float(candidate_scores[index] / max(reference_score, 1e-6)))
            else:
                retain_ratios.append(0.0)

        reference_count = max(int((reference_scores >= 0.20).sum()), 1)
        candidate_count = int((candidate_scores >= 0.20).sum())
        count_ratios.append(candidate_count / reference_count)

    positive_ratios = np.asarray(retain_ratios, np.float32)
    positive_ratios = positive_ratios[positive_ratios > 0]
    retain_ratio = float(np.median(positive_ratios)) if len(positive_ratios) else 0.0
    retain_match = retain_matched / max(retain_total, 1)
    poison_ratio = float(np.median(poison_ratios))
    poison_fire = float(np.mean(np.asarray(poison_scores) >= 0.20))
    proxy = (
        poison_ratio
        + 0.75 * (1 - retain_match)
        + 0.25 * abs(math.log(max(retain_ratio, 1e-3)))
    )
    return {
        "poison_score_ratio_median": poison_ratio,
        "poison_fire_rate_020": poison_fire,
        "retain_total": retain_total,
        "retain_matched": retain_matched,
        "retain_match_rate": retain_match,
        "retain_score_ratio_median": retain_ratio,
        "count_ratio_median": float(np.median(count_ratios)),
        "proxy": proxy,
    }


def passes_gate(metrics):
    return bool(
        metrics["poison_fire_rate_020"] <= GATE["poison_fire_rate_020_max"]
        and metrics["poison_score_ratio_median"] <= GATE["poison_score_ratio_median_max"]
        and metrics["retain_match_rate"] >= GATE["retain_match_rate_min"]
        and GATE["retain_score_ratio_median_min"]
        <= metrics["retain_score_ratio_median"]
        <= GATE["retain_score_ratio_median_max"]
    )


REGISTRY = []
PREDICTIONS = {"original_dense": TEACHER_PREDICTIONS}
CANDIDATE_SPECS = {}
FAILURES = []


def register(experiment, candidate, family, predictions, **extra):
    metrics = metrics_from_predictions(predictions)
    row = {
        "experiment": experiment,
        "candidate": candidate,
        "family": family,
        **metrics,
        "passes_gate": passes_gate(metrics),
        "local_pseudo_clean_macadd": pseudo_clean_macadd(predictions),
        **extra,
    }
    REGISTRY.append(row)
    PREDICTIONS[candidate] = predictions
    log(
        "candidate evaluated",
        experiment=experiment,
        candidate=candidate,
        passes=row["passes_gate"],
        proxy=round(row["proxy"], 6),
        macadd=round(row["local_pseudo_clean_macadd"], 6),
    )
    return row


def capture_failure(experiment, candidate, exc):
    row = {"experiment": experiment, "candidate": candidate, "error": repr(exc)}
    FAILURES.append(row)
    log("candidate failed", **row)


metric_audit = {
    "hidden_clean_filter": "strictly greater than 0.20",
    "participant_filter": None,
    "participant_low_confidence_boxes_participate_in_matching": True,
    "source": "exact public maCADD implementation",
    "interpretation": (
        "A low-confidence submitted box can prevent an unmatched hidden-clean "
        "false negative, but costs its own confidence when it has no overlap."
    ),
}
(OUT / "v7_metric_audit.json").write_text(json.dumps(metric_audit, indent=2), encoding="utf-8")

# %% [markdown]
# ## E43-E46 - metric-aware output candidates

# %%
def threshold_predictions(predictions, threshold):
    output = {}
    for image_id, (boxes, scores) in predictions.items():
        keep = scores >= float(threshold)
        output[image_id] = (boxes[keep].copy(), scores[keep].copy())
    return output


for threshold in []:
    name = f"e43_original_boxbank_t{threshold:g}"
    predictions = threshold_predictions(TEACHER_PREDICTIONS, threshold)
    CANDIDATE_SPECS[name] = {"type": "boxbank_threshold", "threshold": threshold}
    register("E43", name, "dense_original_box_bank", predictions, threshold=threshold)


def matched_source_scores(reference_boxes, source_boxes, source_scores, min_iou=0.50):
    output = np.zeros(len(reference_boxes), np.float32)
    matched = np.zeros(len(reference_boxes), bool)
    for index, reference_box in enumerate(reference_boxes):
        overlaps = box_iou(reference_box, source_boxes)
        if len(overlaps) and overlaps.max() >= min_iou:
            output[index] = float(source_scores[int(overlaps.argmax())])
            matched[index] = True
    return output, matched


def unmatched_aware_blend(indicator_predictions, alpha, beta, floor=0.001):
    output = {}
    for image_id in ALL_IDS:
        boxes, original_scores = TEACHER_PREDICTIONS[image_id]
        indicator_boxes, indicator_scores = indicator_predictions[image_id]
        matched_scores, matched = matched_source_scores(
            boxes, indicator_boxes, indicator_scores
        )
        scores = np.where(
            matched,
            (1 - float(alpha)) * original_scores + float(alpha) * matched_scores,
            float(beta) * original_scores,
        )
        keep = scores >= float(floor)
        output[image_id] = (boxes[keep].copy(), scores[keep].astype(np.float32))
    return output


def continuous_drop_rescore(indicator_predictions, alpha, floor):
    output = {}
    for image_id in ALL_IDS:
        boxes, original_scores = TEACHER_PREDICTIONS[image_id]
        indicator_boxes, indicator_scores = indicator_predictions[image_id]
        matched_scores, _ = matched_source_scores(boxes, indicator_boxes, indicator_scores)
        drop = np.clip(1 - matched_scores / np.clip(original_scores, 1e-6, None), 0, 1)
        scores = np.maximum(
            float(floor),
            original_scores * (1 - float(alpha) * drop),
        ).astype(np.float32)
        output[image_id] = (boxes.copy(), scores)
    return output


def find_v1_checkpoint(step):
    pattern = f"**/repair_matrix/final_full_cls_lr3e5_step{int(step)}.pth"
    matches = [
        path
        for path in Path("/kaggle/input").glob(pattern)
        if "metric-preservation" not in str(path)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one V1 step-{step} checkpoint, found {matches}")
    return matches[0]


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("model", checkpoint)


def model_from_state(state, score_threshold=0.001):
    model = build_original(score_threshold)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected state keys: {unexpected[:8]}")
    model.to(DEVICE).eval()
    return model


V1_PATHS = {step: find_v1_checkpoint(step) for step in [60, 120, 200]}
V1_STATES = {step: checkpoint_state(path) for step, path in V1_PATHS.items()}
V1_PREDICTIONS = {}
for step in [60, 120, 200]:
    model = model_from_state(V1_STATES[step])
    V1_PREDICTIONS[step] = predictions_for_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("V1 checkpoint imported", step=step, path=str(V1_PATHS[step]))


def logit_temperature(values, temperature):
    clipped = np.clip(values.astype(np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    return (1 / (1 + np.exp(-logits / float(temperature)))).astype(np.float32)


def frozen_e33_reference():
    original_standard_model = build_original(0.005)
    original_standard = predictions_for_model(original_standard_model)
    del original_standard_model
    gc.collect()
    torch.cuda.empty_cache()

    v1_standard_model = model_from_state(V1_STATES[200], score_threshold=0.005)
    v1_standard = predictions_for_model(v1_standard_model)
    del v1_standard_model
    gc.collect()
    torch.cuda.empty_cache()

    output = {}
    for image_id in ALL_IDS:
        boxes, original_scores = original_standard[image_id]
        indicator_boxes, indicator_scores = v1_standard[image_id]
        matched_scores, _ = matched_source_scores(
            boxes,
            indicator_boxes,
            indicator_scores,
        )
        ratio = matched_scores / np.clip(original_scores, 1e-6, None)
        scores = original_scores.copy()
        scores[ratio <= 0.35] *= 0.25
        scores = logit_temperature(scores, 0.85)
        keep = scores >= 0.05
        output[image_id] = (boxes[keep].copy(), scores[keep].copy())
    return output


E33_REFERENCE = frozen_e33_reference()
e33_row = register(
    "REFERENCE",
    "e33_threshold_0.05",
    "frozen_v4_incumbent",
    E33_REFERENCE,
    source="validated V4 completion audit",
    eligible_for_selection=True,
)
expected_e33 = {
    "poison_score_ratio_median": 0.19950863887810824,
    "poison_fire_rate_020": 0.20,
    "retain_match_rate": 1.0,
    "retain_score_ratio_median": 0.823215901851654,
    "proxy": 0.24814283326079045,
}
for metric_name, expected_value in expected_e33.items():
    if abs(float(e33_row[metric_name]) - float(expected_value)) > 5e-4:
        raise AssertionError(
            f"E33 reference drift for {metric_name}: "
            f"{e33_row[metric_name]} vs {expected_value}"
        )
if not e33_row["passes_gate"]:
    raise AssertionError(f"Frozen E33 reference no longer passes: {e33_row}")
log("Frozen E33 reference reproduced", **e33_row)

v1_anchor_model = model_from_state(V1_STATES[200], score_threshold=0.05)
V1_ANCHOR = predictions_for_model(v1_anchor_model)
del v1_anchor_model
gc.collect()
torch.cuda.empty_cache()
v1_anchor_row = register(
    "REFERENCE",
    "v1_step200_smoke_anchor",
    "leaderboard_validated_behavioral_anchor",
    V1_ANCHOR,
    public_score=300.8064,
    selector_role="coarse family anchor only",
)
log("Frozen V1 behavioral anchor reproduced", **v1_anchor_row)


for alpha in []:
    for beta in [0.25, 0.50, 0.75]:
        name = f"e44_step200_a{alpha:g}_b{beta:g}"
        predictions = unmatched_aware_blend(V1_PREDICTIONS[200], alpha, beta)
        CANDIDATE_SPECS[name] = {
            "type": "unmatched_aware_blend",
            "checkpoint_step": 200,
            "alpha": alpha,
            "beta": beta,
        }
        register("E44", name, "unmatched_aware_output_blend", predictions, alpha=alpha, beta=beta)


for alpha in []:
    for floor in [0.03, 0.05, 0.10, 0.15]:
        name = f"e45_continuous_a{alpha:g}_f{floor:g}"
        predictions = continuous_drop_rescore(V1_PREDICTIONS[200], alpha, floor)
        CANDIDATE_SPECS[name] = {
            "type": "continuous_drop_rescore",
            "checkpoint_step": 200,
            "alpha": alpha,
            "floor": floor,
        }
        register("E45", name, "continuous_v1_drop_rescore", predictions, alpha=alpha, floor=floor)


for step in []:
    for alpha in [0.02, 0.05, 0.10, 0.20]:
        for beta in [0.50, 0.75]:
            name = f"e46_step{step}_a{alpha:g}_b{beta:g}"
            predictions = unmatched_aware_blend(V1_PREDICTIONS[step], alpha, beta)
            CANDIDATE_SPECS[name] = {
                "type": "early_checkpoint_blend",
                "checkpoint_step": step,
                "alpha": alpha,
                "beta": beta,
            }
            register(
                "E46",
                name,
                "early_checkpoint_output_blend",
                predictions,
                checkpoint_step=step,
                alpha=alpha,
                beta=beta,
            )

# %% [markdown]
# ## E47-E48 - low-dimensional model and structure candidates

# %%
def interpolated_state(checkpoint_state_dict, alpha):
    output = {}
    for name, original_value in ORIGINAL_STATE.items():
        candidate_value = checkpoint_state_dict.get(name)
        is_classification = name.startswith("head.cls_") or ".head.cls_" in name
        if (
            is_classification
            and candidate_value is not None
            and candidate_value.shape == original_value.shape
            and torch.is_floating_point(original_value)
        ):
            output[name] = (
                original_value.float()
                + float(alpha) * (candidate_value.float() - original_value.float())
            ).to(original_value.dtype)
        else:
            output[name] = original_value.clone()
    return output


for step in []:
    for alpha in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        name = f"e47_step{step}_weight_a{alpha:g}"
        try:
            state = interpolated_state(V1_STATES[step], alpha)
            model = model_from_state(state)
            predictions = predictions_for_model(model)
            CANDIDATE_SPECS[name] = {
                "type": "classification_weight_interpolation",
                "checkpoint_step": step,
                "alpha": alpha,
            }
            register(
                "E47",
                name,
                "classification_weight_interpolation",
                predictions,
                checkpoint_step=step,
                alpha=alpha,
            )
            del model, state
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            capture_failure("E47", name, exc)


class LevelAffineScore(torch.nn.Module):
    def __init__(self, base, temperatures, biases):
        super().__init__()
        self.base = base
        self.temperatures = tuple(float(value) for value in temperatures)
        self.biases = tuple(float(value) for value in biases)
        self.call_index = 0

    def reset(self):
        self.call_index = 0

    def forward(self, value):
        output = self.base(value)
        level = self.call_index % 5
        self.call_index += 1
        return output / self.temperatures[level] + self.biases[level]


def affine_model(p3_temperature, p4_temperature, p3_bias, p4_bias):
    model = build_original(0.001)
    model.head.cls_score = LevelAffineScore(
        model.head.cls_score,
        [p3_temperature, p4_temperature, 1.0, 1.0, 1.0],
        [p3_bias, p4_bias, 0.0, 0.0, 0.0],
    )
    model.to(DEVICE).eval()
    return model


AFFINE_SPECS = []
for temperature in []:
    for bias in [-0.10, -0.25, -0.50]:
        AFFINE_SPECS.append((temperature, temperature, bias, bias))
for p3_temperature, p4_temperature in []:
    AFFINE_SPECS.append((p3_temperature, p4_temperature, -0.10, -0.10))

for p3_temperature, p4_temperature, p3_bias, p4_bias in AFFINE_SPECS:
    name = (
        f"e48_p3t{p3_temperature:g}_p4t{p4_temperature:g}"
        f"_p3b{abs(p3_bias):g}_p4b{abs(p4_bias):g}"
    )
    try:
        model = affine_model(p3_temperature, p4_temperature, p3_bias, p4_bias)
        predictions = predictions_for_model(model)
        spec = {
            "type": "p3_p4_affine_score_head",
            "p3_temperature": p3_temperature,
            "p4_temperature": p4_temperature,
            "p3_bias": p3_bias,
            "p4_bias": p4_bias,
        }
        CANDIDATE_SPECS[name] = spec
        register("E48", name, "tiny_affine_head_repair", predictions, **spec)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        capture_failure("E48", name, exc)

# %% [markdown]
# ## E49-E50 - grouped-CV projected repair
#
# The forget gradient is projected only when it conflicts with the retain
# gradient. Backbone, FPN and box regression remain frozen. E50 additionally
# enforces a retain-loss budget and parameter anchor.

# %%
def dense_logits(model, record):
    image_list = model.preprocess_image([record])
    features_dict = model.backbone(image_list.tensor)
    features = [features_dict[name] for name in model.head_in_features]
    predictions = model.head(features)
    logits, _ = model._transpose_dense_predictions(predictions, [model.num_classes, 4])
    anchors = model.anchor_generator(features)
    return logits, anchors


def torch_iou(boxes_a, boxes_b):
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    size = (bottom_right - top_left).clamp(min=0)
    intersection = size[:, :, 0] * size[:, :, 1]
    area_a = (
        (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    ).clamp(min=1e-6)
    area_b = (
        (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ).clamp(min=1e-6)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp(min=1e-6)


def anchor_masks(anchor_boxes, target_box, poison_level):
    target = torch.as_tensor(target_box[None], dtype=torch.float32, device=DEVICE)
    centers = (anchor_boxes[:, :2] + anchor_boxes[:, 2:]) / 2
    target_center = (target[:, :2] + target[:, 2:]) / 2
    target_size = (target[:, 2:] - target[:, :2]) * 1.5
    expanded = torch.cat([target_center - target_size / 2, target_center + target_size / 2], dim=1)
    inside = (
        (centers[:, 0] >= expanded[0, 0])
        & (centers[:, 0] <= expanded[0, 2])
        & (centers[:, 1] >= expanded[0, 1])
        & (centers[:, 1] <= expanded[0, 3])
    )
    poison = torch.zeros(len(anchor_boxes), dtype=torch.bool, device=DEVICE)
    if poison_level:
        overlap = torch_iou(anchor_boxes, target)[:, 0]
        poison |= overlap >= 0.05
        poison[torch.topk(overlap, k=min(32, len(overlap))).indices] = True
    return poison, ~inside


def configure_projected_scope(model, scope):
    parameters = []
    names = []
    for name, parameter in model.named_parameters():
        enabled = (
            name.startswith("head.cls_score")
            if scope == "score"
            else name.startswith("head.cls_")
        )
        parameter.requires_grad = enabled
        if enabled:
            parameters.append(parameter)
            names.append(name)
    if not parameters:
        raise RuntimeError(f"No trainable parameters for {scope}")
    return parameters, names


def projected_losses(student, gray, target_box, original_parameters, anchor_weight):
    record = to_record(gray)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        teacher_logits, _ = dense_logits(TEACHER, record)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        student_logits, anchors = dense_logits(student, record)

    forget_terms = []
    retain_terms = []
    for level_name, student_level, teacher_level, anchor_level in zip(
        student.head_in_features,
        student_logits,
        teacher_logits,
        anchors,
    ):
        poison, outside = anchor_masks(
            anchor_level.tensor,
            target_box,
            level_name in {"p3", "p4"},
        )
        student_values = student_level[0, :, 0].float()
        teacher_values = teacher_level[0, :, 0].float()
        if poison.any():
            forget_terms.append(F.softplus(student_values[poison]).mean())
        teacher_probability = torch.sigmoid(teacher_values)
        retain = outside & (teacher_probability >= 0.01)
        if retain.any():
            retain_terms.append(
                F.smooth_l1_loss(
                    student_values[retain],
                    teacher_values[retain],
                    reduction="mean",
                )
            )

    forget_loss = torch.stack(forget_terms).mean()
    retain_loss = (
        torch.stack(retain_terms).mean()
        if retain_terms
        else torch.zeros((), device=DEVICE)
    )
    anchor_terms = []
    for name, parameter in student.named_parameters():
        if parameter.requires_grad:
            anchor_terms.append((parameter - original_parameters[name]).float().pow(2).mean())
    anchor_loss = (
        torch.stack(anchor_terms).mean()
        if anchor_terms
        else torch.zeros((), device=DEVICE)
    )
    retain_objective = retain_loss + float(anchor_weight) * anchor_loss
    return forget_loss, retain_loss, anchor_loss, retain_objective


def combine_projected_gradients(forget_grads, retain_grads, forget_weight):
    dot = torch.zeros((), device=DEVICE)
    retain_norm = torch.zeros((), device=DEVICE)
    for forget_grad, retain_grad in zip(forget_grads, retain_grads):
        if forget_grad is not None and retain_grad is not None:
            dot += (forget_grad.float() * retain_grad.float()).sum()
            retain_norm += retain_grad.float().pow(2).sum()
    conflict = bool(dot.item() < 0)
    coefficient = dot / retain_norm.clamp(min=1e-12) if conflict else torch.zeros_like(dot)
    combined = []
    for forget_grad, retain_grad in zip(forget_grads, retain_grads):
        if forget_grad is None and retain_grad is None:
            combined.append(None)
            continue
        projected_forget = (
            torch.zeros_like(retain_grad)
            if forget_grad is None
            else forget_grad
        )
        if conflict and retain_grad is not None:
            projected_forget = projected_forget - coefficient.to(projected_forget.dtype) * retain_grad
        retain_component = (
            torch.zeros_like(projected_forget)
            if retain_grad is None
            else retain_grad
        )
        combined.append(retain_component + float(forget_weight) * projected_forget)
    return combined, conflict, float(dot.item())


def train_projected(spec, train_ids, run_name):
    student = build_original(0.001)
    parameters, trainable_names = configure_projected_scope(student, spec["scope"])
    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.Adam(parameters, lr=float(spec["lr"]))
    history = []
    accepted_parameters = [parameter.detach().clone() for parameter in parameters]

    with heartbeat(run_name):
        for step in range(1, int(spec["steps"]) + 1):
            image_id = train_ids[(step - 1) % len(train_ids)]
            student.train()
            optimizer.zero_grad(set_to_none=True)
            forget_loss, retain_loss, anchor_loss, retain_objective = projected_losses(
                student,
                IMAGES[image_id],
                POISON_BOXES[image_id],
                original_parameters,
                spec["anchor_weight"],
            )
            forget_grads = torch.autograd.grad(
                forget_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            retain_grads = torch.autograd.grad(
                retain_objective,
                parameters,
                allow_unused=True,
            )
            combined, conflict, dot = combine_projected_gradients(
                forget_grads,
                retain_grads,
                spec["forget_weight"],
            )
            for parameter, gradient in zip(parameters, combined):
                parameter.grad = None if gradient is None else gradient.detach()
            optimizer.step()

            budget_rejected = False
            if spec.get("retain_budget") is not None:
                check_forget, check_retain, check_anchor, _ = projected_losses(
                    student,
                    IMAGES[image_id],
                    POISON_BOXES[image_id],
                    original_parameters,
                    spec["anchor_weight"],
                )
                if float(check_retain.item()) > float(spec["retain_budget"]):
                    with torch.no_grad():
                        for parameter, accepted in zip(parameters, accepted_parameters):
                            parameter.copy_(accepted)
                    budget_rejected = True
                else:
                    accepted_parameters = [
                        parameter.detach().clone() for parameter in parameters
                    ]

            row = {
                "run": run_name,
                "step": step,
                "image_id": image_id,
                "forget_loss": float(forget_loss.item()),
                "retain_loss": float(retain_loss.item()),
                "anchor_loss": float(anchor_loss.item()),
                "gradient_conflict": conflict,
                "gradient_dot": dot,
                "budget_rejected": budget_rejected,
            }
            history.append(row)
            if step == 1 or step % 10 == 0 or step == int(spec["steps"]):
                log("projected training step", **row)
    student.eval()
    return student, history, trainable_names


log("Starting V1-centered Bundle A")


def adversarial_weight_saliency(sample_ids):
    model = build_original(0.001)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name == "head.cls_score.weight"
    model.train()
    model.zero_grad(set_to_none=True)
    for image_id in sample_ids:
        logits, anchors = dense_logits(model, to_record(IMAGES[image_id]))
        terms = []
        for level_name, level_logits, level_anchors in zip(
            model.head_in_features,
            logits,
            anchors,
        ):
            poison_mask, _ = anchor_masks(
                level_anchors.tensor,
                POISON_BOXES[image_id],
                level_name in {"p3", "p4"},
            )
            if poison_mask.any():
                terms.append(
                    torch.sigmoid(level_logits[0, :, 0].float()[poison_mask]).mean()
                )
        torch.stack(terms).mean().backward()
    gradient = model.head.cls_score.weight.grad.detach()
    saliency = (
        gradient.abs() * model.head.cls_score.weight.detach().abs()
    ).cpu().numpy()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return saliency


# E28F closes the only failed E00-E50 experiment. The explicit copies are the
# fix for the reversed NumPy view that failed in V4.
try:
    weight_saliency = adversarial_weight_saliency(ALL_IDS[:10])
    flat_order = np.argsort(weight_saliency.reshape(-1))[::-1].copy()
    total_weights = int(weight_saliency.size)
    for percentage in [0.001, 0.005, 0.01]:
        name = f"e28f_weight_{percentage * 100:g}pct"
        model = None
        try:
            count = max(1, round(total_weights * percentage))
            indices = flat_order[:count].copy()
            model = build_original(0.001)
            with torch.no_grad():
                flat = model.head.cls_score.weight.data.view(-1)
                flat[
                    torch.as_tensor(
                        indices,
                        dtype=torch.long,
                        device=DEVICE,
                    )
                ] = 0
            predictions = predictions_for_model(model)
            spec = {
                "type": "fixed_weight_level_pruning",
                "percentage": percentage,
                "weight_count": count,
            }
            CANDIDATE_SPECS[name] = spec
            register("E28F", name, "fixed_weight_level_pruning", predictions, **spec)
        except Exception as exc:
            capture_failure("E28F", name, exc)
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
except Exception as exc:
    capture_failure("E28F", "weight_saliency", exc)


# E51 keeps the V1 detector intact and explores only export threshold and NMS.
for threshold in [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]:
    name = f"e51_v1_threshold_{threshold:g}"
    predictions = threshold_predictions(V1_PREDICTIONS[200], threshold)
    spec = {
        "type": "v1_direct_threshold",
        "checkpoint_step": 200,
        "threshold": threshold,
    }
    CANDIDATE_SPECS[name] = spec
    register("E51", name, "v1_direct_export_frontier", predictions, **spec)


def model_from_state_with_nms(state, score_threshold, nms_threshold):
    cfg = make_cfg(score_threshold)
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = float(nms_threshold)
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected state keys: {unexpected[:8]}")
    model.to(DEVICE).eval()
    return model


for nms_threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    name = f"e51_v1_nms_{nms_threshold:g}"
    model = None
    try:
        model = model_from_state_with_nms(
            V1_STATES[200],
            score_threshold=0.05,
            nms_threshold=nms_threshold,
        )
        predictions = predictions_for_model(model)
        spec = {
            "type": "v1_nms_frontier",
            "checkpoint_step": 200,
            "threshold": 0.05,
            "nms_threshold": nms_threshold,
        }
        CANDIDATE_SPECS[name] = spec
        register("E51", name, "v1_direct_export_frontier", predictions, **spec)
    except Exception as exc:
        capture_failure("E51", name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def calibrated_predictions(predictions, temperature, bias, threshold):
    output = {}
    for image_id, (boxes, scores) in predictions.items():
        clipped = np.clip(scores.astype(np.float64), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        calibrated = (
            1 / (1 + np.exp(-(logits / float(temperature) + float(bias))))
        ).astype(np.float32)
        keep = calibrated >= float(threshold)
        output[image_id] = (boxes[keep].copy(), calibrated[keep].copy())
    return output


# E52 changes only V1 confidence scale; no original boxes are restored.
for temperature in [0.75, 1.0, 1.25, 1.5]:
    for bias in [-0.50, -0.25, 0.0]:
        name = f"e52_v1_t{temperature:g}_b{abs(bias):g}"
        predictions = calibrated_predictions(
            V1_PREDICTIONS[200],
            temperature,
            bias,
            threshold=0.05,
        )
        spec = {
            "type": "v1_logit_calibration",
            "temperature": temperature,
            "bias": bias,
            "threshold": 0.05,
        }
        CANDIDATE_SPECS[name] = spec
        register("E52", name, "v1_score_calibration", predictions, **spec)


def classification_checkpoint_soup(weights_by_step):
    base = V1_STATES[200]
    output = {}
    for key, base_value in base.items():
        values = []
        weights = []
        for step, weight in weights_by_step.items():
            value = V1_STATES[step].get(key)
            if value is not None and value.shape == base_value.shape:
                values.append(value)
                weights.append(float(weight))
        is_classification = key.startswith("head.cls_") or ".head.cls_" in key
        if is_classification and values and torch.is_floating_point(base_value):
            total = sum(weights)
            mixed = sum(
                value.float() * (weight / total)
                for value, weight in zip(values, weights)
            )
            output[key] = mixed.to(base_value.dtype)
        else:
            output[key] = base_value.clone()
    return output


SOUP_SPECS = [
    ("s60_s120_equal", {60: 1, 120: 1}),
    ("s120_s200_equal", {120: 1, 200: 1}),
    ("s60_s200_equal", {60: 1, 200: 1}),
    ("all_equal", {60: 1, 120: 1, 200: 1}),
    ("favor_s200", {60: 1, 120: 2, 200: 4}),
]
for suffix, weights_by_step in SOUP_SPECS:
    name = f"e53_{suffix}"
    model = None
    try:
        state = classification_checkpoint_soup(weights_by_step)
        model = model_from_state(state, score_threshold=0.05)
        predictions = predictions_for_model(model)
        spec = {
            "type": "v1_classification_checkpoint_soup",
            "weights_by_step": weights_by_step,
            "threshold": 0.05,
        }
        CANDIDATE_SPECS[name] = spec
        register("E53", name, "v1_checkpoint_model_soup", predictions, **spec)
        del state
    except Exception as exc:
        capture_failure("E53", name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def checkpoint_output_ensemble(weights_by_step, threshold):
    output = {}
    for image_id in ALL_IDS:
        boxes, base_scores = V1_PREDICTIONS[200][image_id]
        weighted = np.zeros(len(boxes), np.float32)
        total = 0.0
        for step, weight in weights_by_step.items():
            source_boxes, source_scores = V1_PREDICTIONS[step][image_id]
            matched, _ = matched_source_scores(
                boxes,
                source_boxes,
                source_scores,
                min_iou=0.50,
            )
            weighted += float(weight) * matched
            total += float(weight)
        scores = weighted / max(total, 1e-6)
        keep = scores >= float(threshold)
        output[image_id] = (boxes[keep].copy(), scores[keep].copy())
    return output


# E54 is deliberately separate from E53: it averages matched inference scores
# while retaining step-200 coordinates.
for suffix, weights_by_step in SOUP_SPECS:
    for threshold in [0.025, 0.05]:
        name = f"e54_{suffix}_t{threshold:g}"
        predictions = checkpoint_output_ensemble(weights_by_step, threshold)
        spec = {
            "type": "v1_checkpoint_output_ensemble",
            "weights_by_step": weights_by_step,
            "threshold": threshold,
        }
        CANDIDATE_SPECS[name] = spec
        register("E54", name, "v1_checkpoint_output_ensemble", predictions, **spec)


PROJECTED_SPECS_ARCHIVE = [
    # E49: conflict-projected forget gradients, classification score only.
    {
        "experiment": "E49",
        "name": "e49_score_lr1e6_fw002_s30",
        "scope": "score",
        "lr": 1e-6,
        "steps": 30,
        "forget_weight": 0.02,
        "anchor_weight": 0.0,
        "retain_budget": None,
    },
    {
        "experiment": "E49",
        "name": "e49_score_lr3e6_fw002_s30",
        "scope": "score",
        "lr": 3e-6,
        "steps": 30,
        "forget_weight": 0.02,
        "anchor_weight": 0.0,
        "retain_budget": None,
    },
    {
        "experiment": "E49",
        "name": "e49_score_lr3e6_fw005_s45",
        "scope": "score",
        "lr": 3e-6,
        "steps": 45,
        "forget_weight": 0.05,
        "anchor_weight": 0.0,
        "retain_budget": None,
    },
    {
        "experiment": "E49",
        "name": "e49_score_lr1e5_fw002_s30",
        "scope": "score",
        "lr": 1e-5,
        "steps": 30,
        "forget_weight": 0.02,
        "anchor_weight": 0.0,
        "retain_budget": None,
    },
    {
        "experiment": "E49",
        "name": "e49_head_lr1e6_fw001_s30",
        "scope": "head",
        "lr": 1e-6,
        "steps": 30,
        "forget_weight": 0.01,
        "anchor_weight": 0.1,
        "retain_budget": None,
    },
    {
        "experiment": "E49",
        "name": "e49_head_lr3e6_fw001_s30",
        "scope": "head",
        "lr": 3e-6,
        "steps": 30,
        "forget_weight": 0.01,
        "anchor_weight": 0.1,
        "retain_budget": None,
    },
    # E50: the same projected update with rollback on a fixed retain-loss budget.
    {
        "experiment": "E50",
        "name": "e50_score_lr3e6_fw002_b1e4",
        "scope": "score",
        "lr": 3e-6,
        "steps": 40,
        "forget_weight": 0.02,
        "anchor_weight": 1.0,
        "retain_budget": 1e-4,
    },
    {
        "experiment": "E50",
        "name": "e50_score_lr3e6_fw005_b5e4",
        "scope": "score",
        "lr": 3e-6,
        "steps": 40,
        "forget_weight": 0.05,
        "anchor_weight": 1.0,
        "retain_budget": 5e-4,
    },
    {
        "experiment": "E50",
        "name": "e50_score_lr1e5_fw002_b1e3",
        "scope": "score",
        "lr": 1e-5,
        "steps": 40,
        "forget_weight": 0.02,
        "anchor_weight": 5.0,
        "retain_budget": 1e-3,
    },
    {
        "experiment": "E50",
        "name": "e50_head_lr1e6_fw001_b1e4",
        "scope": "head",
        "lr": 1e-6,
        "steps": 30,
        "forget_weight": 0.01,
        "anchor_weight": 10.0,
        "retain_budget": 1e-4,
    },
    {
        "experiment": "E50",
        "name": "e50_head_lr3e6_fw001_b5e4",
        "scope": "head",
        "lr": 3e-6,
        "steps": 30,
        "forget_weight": 0.01,
        "anchor_weight": 10.0,
        "retain_budget": 5e-4,
    },
    {
        "experiment": "E50",
        "name": "e50_head_lr1e6_fw002_b1e3_s45",
        "scope": "head",
        "lr": 1e-6,
        "steps": 45,
        "forget_weight": 0.02,
        "anchor_weight": 50.0,
        "retain_budget": 1e-3,
    },
]
PROJECTED_SPECS = []

PROJECTED_HISTORY = []
PROJECTED_CV_ROWS = []
log(
    "Projected repair matrix ready",
    experiments=["E49", "E50"],
    grouped_cv_candidates=len(PROJECTED_SPECS),
    folds=len(FOLDS),
)
for spec in PROJECTED_SPECS:
    combined_validation_predictions = {}
    try:
        for fold, validation_ids in FOLDS.items():
            train_ids = [image_id for image_id in ALL_IDS if image_id not in validation_ids]
            run_name = f"{spec['name']}_fold{fold}"
            model, history, trainable_names = train_projected(spec, train_ids, run_name)
            PROJECTED_HISTORY.extend(history)
            combined_validation_predictions.update(predictions_for_model(model, validation_ids))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        cv_metrics = metrics_from_predictions(combined_validation_predictions)
        cv_row = {
            "experiment": spec["experiment"],
            "candidate": spec["name"],
            **cv_metrics,
            "local_pseudo_clean_macadd": pseudo_clean_macadd(
                combined_validation_predictions
            ),
            "passes_gate": passes_gate(cv_metrics),
        }
        PROJECTED_CV_ROWS.append(cv_row)
        log("projected grouped CV complete", **cv_row)
    except Exception as exc:
        capture_failure(spec["experiment"], spec["name"] + "_cv", exc)


if PROJECTED_CV_ROWS:
    cv_frame = pd.DataFrame(PROJECTED_CV_ROWS).sort_values(
        ["passes_gate", "local_pseudo_clean_macadd", "proxy"],
        ascending=[False, True, True],
    )
    best_projected_name = str(cv_frame.iloc[0].candidate)
    best_projected_spec = next(spec for spec in PROJECTED_SPECS if spec["name"] == best_projected_name)
    try:
        final_name = best_projected_name + "_all20"
        final_model, history, trainable_names = train_projected(
            best_projected_spec,
            ALL_IDS,
            final_name,
        )
        PROJECTED_HISTORY.extend(history)
        predictions = predictions_for_model(final_model)
        CANDIDATE_SPECS[final_name] = {
            "type": "conflict_projected_repair",
            **best_projected_spec,
            "trainable_names": trainable_names,
        }
        row = register(
            best_projected_spec["experiment"],
            final_name,
            "conflict_projected_repair",
            predictions,
            selected_by="five-fold public-unlearn CV",
            scope=best_projected_spec["scope"],
            lr=best_projected_spec["lr"],
            steps=best_projected_spec["steps"],
        )
        torch.save(
            {
                "model": final_model.state_dict(),
                "spec": CANDIDATE_SPECS[final_name],
                "metrics": row,
                "original_sha256": digest.hexdigest(),
            },
            OUT / "v6_projected_candidate.pth",
        )
        del final_model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        capture_failure(best_projected_spec["experiment"], best_projected_name + "_all20", exc)

# %% [markdown]
# ## Frozen local selection
#
# No test inference follows this cell. A Kaggle submission remains a separate,
# explicitly user-authorized action.

# %%
'''
V6_ARCHIVED_SELECTION_BLOCK = r"""
registry = pd.DataFrame(REGISTRY)
registry.to_csv(OUT / "v6_experiment_registry.csv", index=False)
pd.DataFrame(PROJECTED_CV_ROWS).to_csv(OUT / "v6_projected_cv.csv", index=False)
pd.DataFrame(PROJECTED_HISTORY).to_csv(OUT / "v6_projected_training.csv", index=False)
(OUT / "v6_candidate_specs.json").write_text(
    json.dumps(CANDIDATE_SPECS, indent=2, default=str),
    encoding="utf-8",
)
(OUT / "v6_failures.json").write_text(json.dumps(FAILURES, indent=2), encoding="utf-8")


def pareto_front(frame):
    objectives = np.column_stack(
        [
            frame.poison_score_ratio_median.to_numpy(float),
            1 - frame.retain_match_rate.to_numpy(float),
            np.abs(
                np.log(
                    np.clip(
                        frame.retain_score_ratio_median.to_numpy(float),
                        1e-3,
                        None,
                    )
                )
            ),
            frame.local_pseudo_clean_macadd.to_numpy(float),
        ]
    )
    keep = np.ones(len(frame), bool)
    for index in range(len(frame)):
        dominated = np.all(objectives <= objectives[index], axis=1) & np.any(
            objectives < objectives[index],
            axis=1,
        )
        dominated[index] = False
        keep[index] = not dominated.any()
    return frame.loc[keep].copy()


pareto = pareto_front(registry).sort_values(
    ["passes_gate", "local_pseudo_clean_macadd", "proxy"],
    ascending=[False, True, True],
)
pareto.to_csv(OUT / "v6_pareto_front.csv", index=False)

eligible = registry[registry.passes_gate].sort_values(
    ["local_pseudo_clean_macadd", "proxy"],
    ascending=[True, True],
)
selection_pool = eligible if len(eligible) else pareto
best = selection_pool.iloc[0]
new_registry = registry[registry.experiment.isin(["E49", "E50"])]
new_eligible = new_registry[new_registry.passes_gate].sort_values(
    ["local_pseudo_clean_macadd", "proxy"],
    ascending=[True, True],
)
best_new = new_eligible.iloc[0] if len(new_eligible) else None
e33_reference_row = registry[registry.candidate == "e33_threshold_0.05"].iloc[0]
beats_e33 = bool(
    best_new is not None
    and float(best_new.local_pseudo_clean_macadd)
    < float(e33_reference_row.local_pseudo_clean_macadd)
)
selection_lock = {
    "status": "local_research_selection_frozen",
    "best_candidate": str(best.candidate),
    "passes_all_gates": bool(best.passes_gate),
    "metrics": {
        key: float(best[key])
        for key in [
            "poison_score_ratio_median",
            "poison_fire_rate_020",
            "retain_match_rate",
            "retain_score_ratio_median",
            "local_pseudo_clean_macadd",
            "proxy",
        ]
    },
    "selection_source": "public unlearn set and within-set controls only",
    "frozen_incumbent": {
        "candidate": "e33_threshold_0.05",
        "local_pseudo_clean_macadd": float(
            e33_reference_row.local_pseudo_clean_macadd
        ),
    },
    "best_new_candidate": (
        None
        if best_new is None
        else {
            "experiment": str(best_new.experiment),
            "candidate": str(best_new.candidate),
            "local_pseudo_clean_macadd": float(
                best_new.local_pseudo_clean_macadd
            ),
            "proxy": float(best_new.proxy),
        }
    ),
    "new_candidate_beats_e33": beats_e33,
    "executed_experiments": sorted(
        {str(row["experiment"]) for row in PROJECTED_CV_ROWS}
    ),
    "grouped_cv_candidate_count": len(PROJECTED_CV_ROWS),
    "test_data_read": False,
    "submission_generated": False,
    "leaderboard_used": False,
    "metric_note": (
        "Exact public maCADD formula against a public pseudo-clean reference; "
        "not the hidden leaderboard score."
    ),
}
(OUT / "v6_selection_lock.json").write_text(
    json.dumps(selection_lock, indent=2),
    encoding="utf-8",
)

coverage = {
    "expected_experiments": ["E49", "E50"],
    "executed_experiments": sorted(
        {str(row["experiment"]) for row in PROJECTED_CV_ROWS}
    ),
    "planned_grouped_cv_candidates": len(PROJECTED_SPECS),
    "completed_grouped_cv_candidates": len(PROJECTED_CV_ROWS),
    "failure_count": len(FAILURES),
    "completed_not_rerun": [f"E{index}" for index in range(43, 49)],
    "test_data_read": False,
    "submission_generated": False,
}
(OUT / "v6_experiment_coverage.json").write_text(
    json.dumps(coverage, indent=2),
    encoding="utf-8",
)

figure, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = np.where(registry.passes_gate, "#62e7b4", "#f4b860")
axes[0].scatter(
    registry.retain_score_ratio_median,
    registry.poison_score_ratio_median,
    c=colors,
    s=34,
    alpha=0.85,
)
axes[0].axvspan(0.8, 1.2, color="#62e7b4", alpha=0.08)
axes[0].axhspan(0, 0.25, color="#62e7b4", alpha=0.08)
axes[0].set_xlabel("Retained confidence ratio")
axes[0].set_ylabel("Poison score ratio")
axes[0].set_title("E49-E50 suppression versus preservation")

family_best = (
    registry.sort_values(["local_pseudo_clean_macadd", "proxy"])
    .groupby("experiment", as_index=False)
    .first()
)
axes[1].bar(
    family_best.experiment,
    family_best.local_pseudo_clean_macadd,
    color=np.where(family_best.passes_gate, "#62e7b4", "#f4b860"),
)
axes[1].set_ylabel("Public-only pseudo-clean maCADD")
axes[1].set_title("Best candidate per experiment")
figure.tight_layout()
figure.savefig(OUT / "v6_tradeoff.png", dpi=180)
plt.close(figure)

final_report = {
    "status": "complete",
    "experiment_count": 2,
    "candidate_count": int(len(registry)),
    "grouped_cv_candidate_count": int(len(PROJECTED_CV_ROWS)),
    "passing_candidate_count": int(registry.passes_gate.sum()),
    "new_candidate_count": int(len(new_registry)),
    "new_passing_candidate_count": int(new_registry.passes_gate.sum()),
    "new_candidate_beats_e33": beats_e33,
    "selection": selection_lock,
    "metric_audit": metric_audit,
    "failures": FAILURES,
    "kaggle_write_performed": False,
}
(OUT / "v6_final_report.json").write_text(
    json.dumps(final_report, indent=2),
    encoding="utf-8",
)
log(
    "E49-E50 KAGGLE RESEARCH COMPLETE",
    selected=selection_lock["best_candidate"],
    passes=selection_lock["passes_all_gates"],
    candidates=len(registry),
)
"""
'''

# %%
# V7 deliberately freezes a diverse research frontier. It does not pretend that
# a public-unlearn proxy can identify the hidden-test winner.
registry = pd.DataFrame(REGISTRY)
anchor = registry.loc[
    registry.candidate == "v1_step200_smoke_anchor"
].iloc[0]
registry["v1_anchor_distance"] = (
    (
        registry.poison_score_ratio_median.astype(float)
        - float(anchor.poison_score_ratio_median)
    ).abs()
    / max(float(anchor.poison_score_ratio_median), 0.05)
    + (
        registry.poison_fire_rate_020.astype(float)
        - float(anchor.poison_fire_rate_020)
    ).abs()
    + (
        np.log(
            np.clip(
                registry.retain_score_ratio_median.astype(float),
                1e-3,
                None,
            )
        )
        - math.log(max(float(anchor.retain_score_ratio_median), 1e-3))
    ).abs()
    + 0.5
    * (
        registry.retain_match_rate.astype(float)
        - float(anchor.retain_match_rate)
    ).abs()
)
registry.to_csv(OUT / "v7_experiment_registry.csv", index=False)
(OUT / "v7_candidate_specs.json").write_text(
    json.dumps(CANDIDATE_SPECS, indent=2, default=str),
    encoding="utf-8",
)
(OUT / "v7_failures.json").write_text(
    json.dumps(FAILURES, indent=2),
    encoding="utf-8",
)


def v7_pareto_front(frame):
    objectives = np.column_stack(
        [
            frame.poison_score_ratio_median.to_numpy(float),
            frame.poison_fire_rate_020.to_numpy(float),
            1 - frame.retain_match_rate.to_numpy(float),
            frame.v1_anchor_distance.to_numpy(float),
        ]
    )
    keep = np.ones(len(frame), bool)
    for index in range(len(frame)):
        dominated = np.all(objectives <= objectives[index], axis=1) & np.any(
            objectives < objectives[index],
            axis=1,
        )
        dominated[index] = False
        keep[index] = not dominated.any()
    return frame.loc[keep].copy()


pareto = v7_pareto_front(registry).sort_values(
    ["v1_anchor_distance", "poison_score_ratio_median"],
    ascending=[True, True],
)
pareto.to_csv(OUT / "v7_pareto_front.csv", index=False)


def pick_first(frame, sort_columns):
    if frame.empty:
        return None
    return frame.sort_values(sort_columns, ascending=True).iloc[0]


new_registry = registry[
    registry.experiment.isin(CONFIG["experiments"])
].copy()
shortlist_rows = [anchor]

e51_alternative = pick_first(
    new_registry[
        (new_registry.experiment == "E51")
        & (new_registry.v1_anchor_distance > 1e-9)
    ],
    ["v1_anchor_distance", "poison_score_ratio_median"],
)
if e51_alternative is not None:
    shortlist_rows.append(e51_alternative)

retention_safe = new_registry[new_registry.retain_match_rate >= 0.90]
suppression_candidate = pick_first(
    retention_safe,
    [
        "poison_score_ratio_median",
        "poison_fire_rate_020",
        "v1_anchor_distance",
    ],
)
if suppression_candidate is not None:
    shortlist_rows.append(suppression_candidate)

soup_candidate = pick_first(
    new_registry[new_registry.experiment.isin(["E53", "E54"])],
    ["v1_anchor_distance", "poison_score_ratio_median"],
)
if soup_candidate is not None:
    shortlist_rows.append(soup_candidate)

pruning_candidate = pick_first(
    new_registry[new_registry.experiment == "E28F"],
    ["v1_anchor_distance", "poison_score_ratio_median"],
)
if pruning_candidate is not None:
    shortlist_rows.append(pruning_candidate)

shortlist = (
    pd.DataFrame(shortlist_rows)
    .drop_duplicates(subset=["candidate"])
    .reset_index(drop=True)
)
shortlist["shortlist_rank"] = np.arange(1, len(shortlist) + 1)
shortlist.to_csv(OUT / "v7_diversity_shortlist.csv", index=False)


def compact_candidate(row):
    return {
        "experiment": str(row.experiment),
        "candidate": str(row.candidate),
        "family": str(row.family),
        "poison_score_ratio_median": float(row.poison_score_ratio_median),
        "poison_fire_rate_020": float(row.poison_fire_rate_020),
        "retain_match_rate": float(row.retain_match_rate),
        "retain_score_ratio_median": float(row.retain_score_ratio_median),
        "local_pseudo_clean_macadd": float(row.local_pseudo_clean_macadd),
        "v1_anchor_distance": float(row.v1_anchor_distance),
    }


selection_lock = {
    "status": "research_frontier_frozen_no_hidden_winner",
    "winner_declared": False,
    "behavioral_anchor": {
        "candidate": "v1_step200_smoke_anchor",
        "public_score": 300.8064,
        "role": "coarse family anchor only; not presumed optimal",
    },
    "historical_rejected_selector": {
        "candidate": "e33_threshold_0.05",
        "public_score": 398.0498,
        "reason": "substantially worse public score than the V1 smoke anchor",
    },
    "shortlist": [
        compact_candidate(row)
        for _, row in shortlist.iterrows()
    ],
    "selection_source": (
        "public unlearn images and within-set controls; aggregate Kaggle scores "
        "used only to reject E33's selector family and establish a coarse V1 anchor"
    ),
    "executed_experiments": sorted(
        set(new_registry.experiment.astype(str))
    ),
    "test_data_read": False,
    "submission_generated": False,
    "test_predictions_used_for_selection": False,
    "leaderboard_per_image_feedback_used": False,
    "metric_note": (
        "Exact public maCADD formula against a public pseudo-clean reference; "
        "diagnostic only and not the hidden leaderboard score."
    ),
}
(OUT / "v7_selection_lock.json").write_text(
    json.dumps(selection_lock, indent=2),
    encoding="utf-8",
)

executed_experiments = sorted(set(new_registry.experiment.astype(str)))
candidate_counts = {
    experiment: int((new_registry.experiment == experiment).sum())
    for experiment in CONFIG["experiments"]
}
coverage = {
    "expected_experiments": CONFIG["experiments"],
    "executed_experiments": executed_experiments,
    "candidate_counts": candidate_counts,
    "all_expected_experiments_executed": (
        set(executed_experiments) == set(CONFIG["experiments"])
    ),
    "failure_count": len(FAILURES),
    "completed_not_rerun": CONFIG["completed_not_rerun"],
    "archived_projected_candidate_count": len(PROJECTED_SPECS_ARCHIVE),
    "archived_projected_candidates_executed": len(PROJECTED_CV_ROWS),
    "test_data_read": False,
    "submission_generated": False,
}
(OUT / "v7_experiment_coverage.json").write_text(
    json.dumps(coverage, indent=2),
    encoding="utf-8",
)

figure, axes = plt.subplots(1, 2, figsize=(13, 5))
palette = {
    "REFERENCE": "#ffffff",
    "E28F": "#ff6b6b",
    "E51": "#62e7b4",
    "E52": "#4da3ff",
    "E53": "#f4b860",
    "E54": "#c084fc",
}
for experiment, group in registry.groupby("experiment"):
    axes[0].scatter(
        group.retain_score_ratio_median,
        group.poison_score_ratio_median,
        color=palette.get(str(experiment), "#9aa4b2"),
        label=str(experiment),
        s=74 if experiment == "REFERENCE" else 38,
        marker="*" if experiment == "REFERENCE" else "o",
        edgecolor="#18202b",
        linewidth=0.5,
        alpha=0.9,
    )
axes[0].set_xlabel("Retained confidence ratio")
axes[0].set_ylabel("Poison score ratio")
axes[0].set_title("V1-centered suppression and preservation frontier")
axes[0].legend(fontsize=8, ncol=2)

family_best = (
    registry.sort_values(["v1_anchor_distance", "poison_score_ratio_median"])
    .groupby("experiment", as_index=False)
    .first()
)
axes[1].bar(
    family_best.experiment,
    family_best.v1_anchor_distance,
    color=[
        palette.get(str(experiment), "#9aa4b2")
        for experiment in family_best.experiment
    ],
)
axes[1].set_ylabel("Distance from V1 behavioral anchor")
axes[1].set_title("Closest candidate in each experiment")
figure.tight_layout()
figure.savefig(OUT / "v7_frontier.png", dpi=180)
plt.close(figure)

final_report = {
    "status": "complete",
    "experiment_count": len(CONFIG["experiments"]),
    "candidate_count": int(len(registry)),
    "new_candidate_count": int(len(new_registry)),
    "passing_candidate_count": int(registry.passes_gate.sum()),
    "new_passing_candidate_count": int(new_registry.passes_gate.sum()),
    "winner_declared": False,
    "next_bundle": ["E55", "E56", "E57", "E58", "E59", "E60"],
    "selection": selection_lock,
    "coverage": coverage,
    "metric_audit": metric_audit,
    "failures": FAILURES,
    "kaggle_write_performed": False,
}
(OUT / "v7_final_report.json").write_text(
    json.dumps(final_report, indent=2),
    encoding="utf-8",
)
log(
    "V7 BUNDLE A KAGGLE RESEARCH COMPLETE",
    winner_declared=False,
    experiments=executed_experiments,
    candidates=len(registry),
)
