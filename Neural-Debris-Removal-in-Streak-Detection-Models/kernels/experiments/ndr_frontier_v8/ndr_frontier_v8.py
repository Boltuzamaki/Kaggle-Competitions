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
# # Neural Debris NDR Frontier V8
# ## Compartmentalized repair heads, synthetic retention and ensemble survival
#
# Lower is better in this competition. The exact public NDR_trial1 reproduction
# scored 229.2314, while the earlier E33 pipeline scored 398.0498.
#
# This notebook searches a legal neighborhood around NDR229. It keeps original
# poisoned-model boxes and localization, and uses several independently repaired
# classification heads only as auxiliary survival probes. Model selection uses
# the public unlearn set and deterministic synthetic controls. Test images are
# not read until the selection lock has been written.
#
# The kernel creates finalist CSVs but never submits them.

# %% [markdown]
# ## 1. Stable Kaggle T4 setup

# %%
import importlib.util
import os
import subprocess
import sys

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
os.environ["MAX_JOBS"] = "2"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "setuptools<81"],
    check=True,
)
if importlib.util.find_spec("detectron2") is not None:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "detectron2"],
        check=True,
    )
print("[SETUP] Building Detectron2 for Tesla T4 (SM 7.5)", flush=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-build-isolation",
        "git+https://github.com/facebookresearch/detectron2.git",
    ],
    check=True,
)

# %% [markdown]
# ## 2. Imports, paths, logging and frozen experiment plan

# %%
import copy
import gc
import hashlib
import itertools
import json
import logging
import math
import random
import shutil
import threading
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_train_loader,
    detection_utils as utils,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.structures import BoxMode

logging.getLogger("detectron2").setLevel(logging.ERROR)
logging.getLogger("fvcore").setLevel(logging.ERROR)

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
POISONED_WEIGHTS = ROOT / "poisoned_model" / "poisoned_model.pth"
UNLEARN_DIR = ROOT / "unlearn_set"
TEST_DIR = ROOT / "test_set" / "test_set"
SAMPLE_SUB_PATH = ROOT / "sample_submission.csv"

RUN_DIR = Path("/kaggle/working/ndr_frontier_v8")
MODEL_DIR = RUN_DIR / "models"
SYNTH_DIR = RUN_DIR / "synthetic"
RUN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

RUN_LOG = RUN_DIR / "run.jsonl"
TRAIN_LOG = RUN_DIR / "training_history.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_W = IMG_H = 1024
BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
NUM_CLASSES = 1
BATCH_SIZE = 4
IOU_THRESHOLDS = np.arange(0.2, 0.91, 0.1)
A_FACTOR = 10.0

MODEL_SPECS = [
    {
        "experiment": "E56",
        "candidate": "ndr_bug15_seed42",
        "pruning_mode": "public_bug_two_layer",
        "score_mode": "public_bug",
        "prune_frac": 0.15,
        "seed": 42,
        "lr": 2.5e-4,
        "iters": 20,
        "ewc_lambda": 500.0,
        "grad_clip_norm": None,
        "train_dataset": "unlearn_empty",
    },
    {
        "experiment": "E57",
        "candidate": "ndr_bug10_seed137",
        "pruning_mode": "public_bug_two_layer",
        "score_mode": "public_bug",
        "prune_frac": 0.10,
        "seed": 137,
        "lr": 2.0e-4,
        "iters": 20,
        "ewc_lambda": 1000.0,
        "grad_clip_norm": 1.0,
        "train_dataset": "unlearn_empty",
    },
    {
        "experiment": "E58",
        "candidate": "ndr_aligned4_p3p4_5pct",
        "pruning_mode": "aligned_four_layer",
        "score_mode": "corrected_p3p4",
        "prune_frac": 0.05,
        "seed": 271,
        "lr": 1.5e-4,
        "iters": 20,
        "ewc_lambda": 2000.0,
        "grad_clip_norm": 1.0,
        "train_dataset": "unlearn_empty",
    },
]

SYNTH_RECOVERY_SPEC = {
    "experiment": "E59",
    "candidate": "ndr_synthetic_recovery",
    "base_candidate": "ndr_bug15_seed42",
    "seed": 811,
    "lr": 1.0e-4,
    "iters": 30,
    "ewc_lambda": 1500.0,
    "grad_clip_norm": 1.0,
    "train_dataset": "mixed_synthetic_recovery",
}

POSTPROCESS_GRID = {
    "aggregator": ["max_survival", "mean_survival", "median_survival"],
    "geometry_weight": [0.0, 0.05],
    "p_lo": [0.20, 0.30],
    "p_hi": [0.55, 0.70],
    "min_keep": [0.15, 0.20],
    "eps": [0.005, 0.010],
}

RULE_GUARD = {
    "external_models": False,
    "manual_test_labels": False,
    "automatic_external_test_labels": False,
    "test_pixels_read_before_selection_lock": False,
    "selection_sources": [
        "public unlearn annotations",
        "within-unlearn inpaint controls",
        "deterministic synthetic streak controls",
        "provided poisoned model predictions on public controls",
    ],
    "leaderboard_used_for_selection": False,
    "competition_submission_created": False,
}

CONFIG = {
    "bundle": "NDR_FRONTIER_V8",
    "score_direction": "lower_is_better",
    "incumbent_submission": {
        "method": "E55_NDR229_EXACT",
        "public_score": 229.2314,
    },
    "rejected_history": {
        "method": "E33",
        "public_score": 398.0498,
    },
    "model_specs": MODEL_SPECS,
    "synthetic_recovery_spec": SYNTH_RECOVERY_SPEC,
    "postprocess_grid": POSTPROCESS_GRID,
    "rule_7a_guard": RULE_GUARD,
}
(RUN_DIR / "v8_config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")


def log(message, **fields):
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "message": str(message),
        **fields,
    }
    print(json.dumps(row, default=str), flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def heartbeat(label, stop_event, started):
    while not stop_event.wait(30):
        log("HEARTBEAT", stage=label, elapsed_sec=round(time.time() - started, 1))


def run_with_heartbeat(label, fn):
    stop_event = threading.Event()
    started = time.time()
    worker = threading.Thread(
        target=heartbeat,
        args=(label, stop_event, started),
        daemon=True,
    )
    worker.start()
    try:
        return fn()
    finally:
        stop_event.set()
        worker.join(timeout=2)
        log("STAGE_COMPLETE", stage=label, elapsed_sec=round(time.time() - started, 1))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if DEVICE != "cuda":
    raise RuntimeError("This notebook requires a Kaggle GPU")
gpu_capability = torch.cuda.get_device_capability(0)
gpu_arch = f"sm_{gpu_capability[0]}{gpu_capability[1]}"
if gpu_arch not in torch.cuda.get_arch_list():
    raise RuntimeError(f"PyTorch does not contain kernels for {gpu_arch}")
cuda_probe = (torch.arange(16, device="cuda", dtype=torch.float32).square() + 1).sum()
torch.cuda.synchronize()
assert float(cuda_probe.cpu()) == 1256.0

required_inputs = [
    POISONED_WEIGHTS,
    UNLEARN_DIR / "annotations_coco.json",
    SAMPLE_SUB_PATH,
]
missing = [str(path) for path in required_inputs if not path.exists()]
if missing:
    raise FileNotFoundError(missing)

log(
    "RUN_START",
    gpu=torch.cuda.get_device_name(0),
    gpu_arch=gpu_arch,
    torch=torch.__version__,
    cuda=torch.version.cuda,
    config=CONFIG,
)

# %% [markdown]
# ## 3. Public unlearn data and deterministic synthetic controls

# %%
def load_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0 * 255.0
    else:
        im = im.astype(np.float32)
    im = np.clip(im, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return np.ascontiguousarray(im)


with (UNLEARN_DIR / "annotations_coco.json").open() as handle:
    coco = json.load(handle)

poison_boxes = {}
for ann in coco["annotations"]:
    poison_boxes.setdefault(int(ann["image_id"]), []).append(
        [float(value) for value in ann["bbox"]]
    )
assert len(coco["images"]) == 20
assert sum(map(len, poison_boxes.values())) == 20

unlearn_records = [
    {
        "file_name": str(UNLEARN_DIR / image["file_name"]),
        "height": int(image["height"]),
        "width": int(image["width"]),
        "image_id": int(image["id"]),
        "annotations": [],
    }
    for image in coco["images"]
]


def register_records(name, records):
    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    DatasetCatalog.register(name, lambda records=records: records)
    MetadataCatalog.get(name).set(thing_classes=["streak"])


UNLEARN_DATASET = "v8_unlearn_empty"
register_records(UNLEARN_DATASET, unlearn_records)


class UInt16Mapper(DatasetMapper):
    def __call__(self, dataset_dict):
        item = copy.deepcopy(dataset_dict)
        image = load_image(item["file_name"])
        item["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
        item["instances"] = utils.annotations_to_instances(
            item.get("annotations", []),
            image.shape[:2],
        )
        return item


def inpaint_poison(image, boxes):
    gray = image[:, :, 0].astype(np.uint8)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for x, y, width, height in boxes:
        pad = 8
        x1 = max(0, int(math.floor(x)) - pad)
        y1 = max(0, int(math.floor(y)) - pad)
        x2 = min(gray.shape[1], int(math.ceil(x + width)) + pad)
        y2 = min(gray.shape[0], int(math.ceil(y + height)) + pad)
        mask[y1:y2, x1:x2] = 255
    clean = cv2.inpaint(gray, mask, 9, cv2.INPAINT_TELEA)
    return np.repeat(clean[:, :, None].astype(np.float32), 3, axis=2)


def draw_synthetic_streak(base, rng, dashed_probability=0.20):
    image = base[:, :, 0].copy()
    height, width = image.shape
    length = float(rng.uniform(24, 160))
    angle = float(rng.uniform(0, math.pi))
    thickness = int(rng.integers(1, 5))
    margin = int(length / 2 + 12)
    cx = int(rng.integers(margin, max(margin + 1, width - margin)))
    cy = int(rng.integers(margin, max(margin + 1, height - margin)))
    dx = math.cos(angle) * length / 2
    dy = math.sin(angle) * length / 2
    p1 = np.array([cx - dx, cy - dy], dtype=np.float32)
    p2 = np.array([cx + dx, cy + dy], dtype=np.float32)
    local = image[
        max(0, cy - 48):min(height, cy + 49),
        max(0, cx - 48):min(width, cx + 49),
    ]
    intensity = float(
        np.clip(np.percentile(local, 99.5) + rng.uniform(35, 120), 90, 255)
    )
    canvas = image.copy()
    if rng.random() < dashed_probability:
        segments = int(rng.integers(3, 7))
        for segment in range(segments):
            lo = segment / segments
            hi = min(1.0, lo + 0.58 / segments)
            q1 = p1 * (1 - lo) + p2 * lo
            q2 = p1 * (1 - hi) + p2 * hi
            cv2.line(
                canvas,
                tuple(np.round(q1).astype(int)),
                tuple(np.round(q2).astype(int)),
                intensity,
                thickness,
                cv2.LINE_AA,
            )
    else:
        cv2.line(
            canvas,
            tuple(np.round(p1).astype(int)),
            tuple(np.round(p2).astype(int)),
            intensity,
            thickness,
            cv2.LINE_AA,
        )
    sigma = float(rng.uniform(0.35, 1.10))
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma, sigmaY=sigma)
    x1 = float(max(0, min(p1[0], p2[0]) - thickness - 5))
    y1 = float(max(0, min(p1[1], p2[1]) - thickness - 5))
    x2 = float(min(width, max(p1[0], p2[0]) + thickness + 5))
    y2 = float(min(height, max(p1[1], p2[1]) + thickness + 5))
    rgb = np.repeat(canvas[:, :, None], 3, axis=2).astype(np.float32)
    return rgb, [x1, y1, x2 - x1, y2 - y1]


def build_synthetic_split(split, per_source, seed):
    rng = np.random.default_rng(seed)
    split_dir = SYNTH_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    records = []
    boxes = {}
    for record in unlearn_records:
        source_id = int(record["image_id"])
        clean = inpaint_poison(load_image(record["file_name"]), poison_boxes[source_id])
        for variant in range(per_source):
            synthetic, bbox = draw_synthetic_streak(clean, rng)
            image_id = source_id * 1000 + variant + (0 if split == "train" else 500)
            path = split_dir / f"{image_id}.png"
            cv2.imwrite(
                str(path),
                np.clip(synthetic[:, :, 0] * 257.0, 0, 65535).astype(np.uint16),
            )
            records.append(
                {
                    "file_name": str(path),
                    "height": IMG_H,
                    "width": IMG_W,
                    "image_id": image_id,
                    "annotations": [
                        {
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": 0,
                        }
                    ],
                    "source_image_id": source_id,
                }
            )
            boxes[str(image_id)] = bbox
    return records, boxes


synthetic_train_records, synthetic_train_boxes = build_synthetic_split(
    "train", per_source=8, seed=20260720
)
synthetic_val_records, synthetic_val_boxes = build_synthetic_split(
    "validation", per_source=4, seed=20260721
)

mixed_records = []
for repeat in range(4):
    for record in unlearn_records:
        item = copy.deepcopy(record)
        item["image_id"] = f"forget_{record['image_id']}_{repeat}"
        mixed_records.append(item)
mixed_records.extend(synthetic_train_records)

SYNTH_TRAIN_DATASET = "v8_synthetic_recovery"
register_records(SYNTH_TRAIN_DATASET, mixed_records)
log(
    "DATA_READY",
    unlearn_images=len(unlearn_records),
    synthetic_train=len(synthetic_train_records),
    synthetic_validation=len(synthetic_val_records),
    mixed_training=len(mixed_records),
)

# %% [markdown]
# ## 4. RetinaNet helpers and exact public maCADD implementation

# %%
def build_cfg(weights, score_thresh=0.02, train_dataset=None, output_dir=None):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = DEVICE
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.DATASETS.TRAIN = (train_dataset or UNLEARN_DATASET,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.STEPS = []
    cfg.OUTPUT_DIR = str(output_dir or RUN_DIR)
    return cfg


def load_model(weights):
    cfg = build_cfg(weights)
    model = build_model(cfg).to(DEVICE)
    DetectionCheckpointer(model).load(str(weights))
    return model


def build_predictor(weights, threshold=0.02):
    return DefaultPredictor(build_cfg(weights, score_thresh=threshold))


def iou_matrix(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def greedy_matches(ious, threshold):
    rows, cols = np.where(ious >= threshold)
    if len(rows) == 0:
        return []
    order = np.argsort(ious[rows, cols], kind="stable")[::-1]
    used_rows, used_cols, matches = set(), set(), []
    for index in order:
        row, col = int(rows[index]), int(cols[index])
        if row in used_rows or col in used_cols:
            continue
        matches.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
    return matches


def acadd_at_threshold(clean_boxes, clean_scores, pred_boxes, pred_scores, threshold):
    ious = iou_matrix(clean_boxes, pred_boxes)
    matches = greedy_matches(ious, threshold)
    matched_clean = {row for row, _ in matches}
    matched_pred = {col for _, col in matches}
    score = 0.0
    for clean_index, pred_index in matches:
        difference = float(clean_scores[clean_index]) - float(pred_scores[pred_index])
        score += difference if difference > 0 else (-difference / A_FACTOR)
    score += sum(
        float(clean_scores[index])
        for index in range(len(clean_scores))
        if index not in matched_clean
    )
    score += sum(
        float(pred_scores[index])
        for index in range(len(pred_scores))
        if index not in matched_pred
    )
    return score


def macadd(clean_predictions, participant_predictions):
    weights = IOU_THRESHOLDS / IOU_THRESHOLDS.sum()
    values = []
    for image_id, (clean_boxes, clean_scores) in clean_predictions.items():
        clean_keep = clean_scores > 0.20
        clean_boxes = clean_boxes[clean_keep]
        clean_scores = clean_scores[clean_keep]
        pred_boxes, pred_scores = participant_predictions.get(
            image_id,
            (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)),
        )
        values.append(
            sum(
                weight
                * acadd_at_threshold(
                    clean_boxes,
                    clean_scores,
                    pred_boxes,
                    pred_scores,
                    threshold,
                )
                for weight, threshold in zip(weights, IOU_THRESHOLDS)
            )
        )
    return float(np.mean(values)) if values else math.inf


def predict_records(predictor, records, label):
    output = {}
    for record in tqdm(records, desc=label):
        image = load_image(record["file_name"])
        instances = predictor(image)["instances"].to("cpu")
        output[str(record["image_id"])] = (
            instances.pred_boxes.tensor.numpy().astype(np.float32),
            instances.scores.numpy().astype(np.float32),
        )
    return output

# %% [markdown]
# ## 5. Public-bug and corrected P3/P4 activation scores

# %%
def collect_public_bug_activations(model):
    model.eval()
    layers = [module for module in model.head.cls_subnet if isinstance(module, nn.Conv2d)]
    stored = {index: [] for index in range(len(layers))}
    hooks = [
        layer.register_forward_hook(
            lambda module, inputs, output, index=index: stored[index].append(
                output.detach().cpu()
            )
        )
        for index, layer in enumerate(layers)
    ]
    try:
        with torch.no_grad():
            for record in tqdm(unlearn_records, desc="Public-bug activations"):
                image = load_image(record["file_name"])
                tensor = torch.as_tensor(image.transpose(2, 0, 1)).to(DEVICE)
                model([{"image": tensor}])
    finally:
        for hook in hooks:
            hook.remove()
    return stored


def public_bug_scores(stored, seed):
    score_by_layer = {}
    for layer_index, activation_list in stored.items():
        foreground = None
        background = None
        foreground_count = 0
        background_count = 0
        for activation, record in zip(activation_list, unlearn_records):
            activation = activation[0]
            channels, activation_h, activation_w = activation.shape
            scale_x = activation_w / record["width"]
            scale_y = activation_h / record["height"]
            if foreground is None:
                foreground = torch.zeros(channels)
                background = torch.zeros(channels)
            boxes = poison_boxes[int(record["image_id"])]
            for x, y, width, height in boxes:
                x1 = max(0, int(x * scale_x))
                y1 = max(0, int(y * scale_y))
                x2 = min(activation_w, int((x + width) * scale_x) + 1)
                y2 = min(activation_h, int((y + height) * scale_y) + 1)
                foreground += activation[:, y1:y2, x1:x2].relu().mean((1, 2))
                foreground_count += 1
            # The public notebook resets the RNG for each activation/record
            # pair. Preserving that quirk for seed 42 keeps E56 score-faithful;
            # E57 changes only the predeclared background seed.
            rng = np.random.default_rng(seed)
            patch_w = max(1, int(16 * scale_x))
            patch_h = max(1, int(16 * scale_y))
            x1 = int(rng.integers(0, max(1, activation_w - patch_w + 1)))
            y1 = int(rng.integers(0, max(1, activation_h - patch_h + 1)))
            background += activation[:, y1:y1 + patch_h, x1:x1 + patch_w].relu().mean(
                (1, 2)
            )
            background_count += 1
        score_by_layer[layer_index] = (
            foreground / max(1, foreground_count)
            - background / max(1, background_count)
        ).numpy()
    return score_by_layer


def corrected_p3p4_scores(model, seed):
    model.eval()
    layers = [module for module in model.head.cls_subnet if isinstance(module, nn.Conv2d)]
    accumulators = {
        index: {
            "foreground": torch.zeros(layer.out_channels),
            "background": torch.zeros(layer.out_channels),
            "foreground_count": 0,
            "background_count": 0,
        }
        for index, layer in enumerate(layers)
    }
    context = {"record": None, "call_by_layer": {}}

    def make_hook(layer_index):
        def hook(module, inputs, output):
            call_index = context["call_by_layer"].get(layer_index, 0)
            context["call_by_layer"][layer_index] = call_index + 1
            level_index = call_index % 5
            if level_index not in (0, 1):
                return
            record = context["record"]
            activation = output.detach().cpu()[0].relu()
            channels, activation_h, activation_w = activation.shape
            scale_x = activation_w / record["width"]
            scale_y = activation_h / record["height"]
            rng = np.random.default_rng(
                seed + 1009 * int(record["image_id"]) + 31 * level_index
            )
            for x, y, width, height in poison_boxes[int(record["image_id"])]:
                x1 = max(0, int(x * scale_x))
                y1 = max(0, int(y * scale_y))
                x2 = min(activation_w, int((x + width) * scale_x) + 1)
                y2 = min(activation_h, int((y + height) * scale_y) + 1)
                patch_w = max(1, x2 - x1)
                patch_h = max(1, y2 - y1)
                accumulators[layer_index]["foreground"] += activation[
                    :, y1:y2, x1:x2
                ].mean((1, 2))
                accumulators[layer_index]["foreground_count"] += 1
                bx = int(rng.integers(0, max(1, activation_w - patch_w + 1)))
                by = int(rng.integers(0, max(1, activation_h - patch_h + 1)))
                accumulators[layer_index]["background"] += activation[
                    :, by:by + patch_h, bx:bx + patch_w
                ].mean((1, 2))
                accumulators[layer_index]["background_count"] += 1

        return hook

    hooks = [
        layer.register_forward_hook(make_hook(index))
        for index, layer in enumerate(layers)
    ]
    try:
        with torch.no_grad():
            for record in tqdm(unlearn_records, desc="Corrected P3/P4 activations"):
                context["record"] = record
                context["call_by_layer"] = {}
                image = load_image(record["file_name"])
                tensor = torch.as_tensor(image.transpose(2, 0, 1)).to(DEVICE)
                model([{"image": tensor}])
    finally:
        for hook in hooks:
            hook.remove()
    return {
        index: (
            values["foreground"] / max(1, values["foreground_count"])
            - values["background"] / max(1, values["background_count"])
        ).numpy()
        for index, values in accumulators.items()
    }


def prune_model(model, scores, mode, fraction):
    records = []
    if mode == "public_bug_two_layer":
        targets = [
            (sequential_index, module)
            for sequential_index, module in enumerate(model.head.cls_subnet)
            if isinstance(module, nn.Conv2d)
        ]
        iterator = [
            (sequential_index, sequential_index, module)
            for sequential_index, module in targets
            if sequential_index in scores
        ]
    elif mode == "aligned_four_layer":
        layers = [
            module for module in model.head.cls_subnet if isinstance(module, nn.Conv2d)
        ]
        iterator = [
            (layer_index, layer_index, module)
            for layer_index, module in enumerate(layers)
        ]
    else:
        raise ValueError(mode)
    for target_index, score_key, layer in iterator:
        values = scores[score_key]
        count = max(1, int(len(values) * fraction))
        channel_ids = np.argsort(values, kind="stable")[-count:].copy()
        with torch.no_grad():
            ids = torch.as_tensor(channel_ids, device=layer.weight.device)
            layer.weight.data[ids] = 0.0
            if layer.bias is not None:
                layer.bias.data[ids] = 0.0
        records.append(
            {
                "target_index": int(target_index),
                "score_key": int(score_key),
                "channels_pruned": int(count),
                "channels_total": int(len(values)),
                "channel_ids": [int(value) for value in channel_ids],
            }
        )
    return model, records

# %% [markdown]
# ## 6. Classification-only EWC training

# %%
all_training_rows = []


class ClassifierEWCTrainer(DefaultTrainer):
    anchor_weights = None
    ewc_lambda = 0.0
    grad_clip_norm = None
    candidate_name = ""

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "cls_subnet" in name or "cls_score" in name
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        assert trainable == 2_376_455, trainable
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        records = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper = UInt16Mapper(cfg, is_train=True, augmentations=[])
        return build_detection_train_loader(cfg, mapper=mapper, dataset=records)

    def run_step(self):
        assert self.model.training
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)
        loss_dict = self.model(data)
        ewc_loss = torch.tensor(0.0, device=DEVICE)
        if self.anchor_weights:
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and name in self.anchor_weights:
                    ewc_loss += ((parameter - self.anchor_weights[name]) ** 2).sum()
        loss_dict["loss_ewc"] = self.ewc_lambda * ewc_loss
        total_loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ],
                max_norm=float(self.grad_clip_norm),
            )
        self.optimizer.step()
        row = {
            "candidate": self.candidate_name,
            "iteration": int(self.iter),
            "loss_total": float(total_loss.detach().cpu()),
            **{
                key: float(value.detach().cpu())
                for key, value in loss_dict.items()
            },
        }
        all_training_rows.append(row)
        log("TRAIN_STEP", **row)


def train_from_checkpoint(
    checkpoint,
    candidate,
    dataset_name,
    learning_rate,
    iterations,
    ewc_lambda,
    grad_clip_norm,
):
    anchor_model = load_model(checkpoint)
    anchor_weights = {
        name: parameter.detach().clone()
        for name, parameter in anchor_model.named_parameters()
        if "cls_subnet" in name or "cls_score" in name
    }
    del anchor_model
    torch.cuda.empty_cache()
    output_dir = MODEL_DIR / candidate
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(
        checkpoint,
        train_dataset=dataset_name,
        output_dir=output_dir,
    )
    cfg.SOLVER.BASE_LR = float(learning_rate)
    cfg.SOLVER.MAX_ITER = int(iterations)
    trainer = ClassifierEWCTrainer(cfg)
    trainer.anchor_weights = anchor_weights
    trainer.ewc_lambda = float(ewc_lambda)
    trainer.grad_clip_norm = grad_clip_norm
    trainer.candidate_name = candidate
    trainer.resume_or_load(resume=False)
    run_with_heartbeat(f"training_{candidate}", trainer.train)
    final_path = output_dir / "model_final.pth"
    if not final_path.exists():
        raise FileNotFoundError(final_path)
    del trainer, anchor_weights
    gc.collect()
    torch.cuda.empty_cache()
    return final_path


model_paths = {}
pruning_audits = {}
for spec in MODEL_SPECS:
    set_seed(spec["seed"])
    candidate = spec["candidate"]
    log("MODEL_START", spec=spec)
    model = load_model(POISONED_WEIGHTS)
    if spec["score_mode"] == "public_bug":
        stored = run_with_heartbeat(
            f"activation_{candidate}",
            lambda model=model: collect_public_bug_activations(model),
        )
        scores = public_bug_scores(stored, spec["seed"])
        del stored
    else:
        scores = run_with_heartbeat(
            f"activation_{candidate}",
            lambda model=model, seed=spec["seed"]: corrected_p3p4_scores(model, seed),
        )
    model, audit_rows = prune_model(
        model,
        scores,
        spec["pruning_mode"],
        spec["prune_frac"],
    )
    pruning_audits[candidate] = {
        "mode": spec["pruning_mode"],
        "score_mode": spec["score_mode"],
        "executed": audit_rows,
    }
    pruned_path = MODEL_DIR / candidate / "pruned_model.pth"
    pruned_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, pruned_path)
    del model, scores
    gc.collect()
    torch.cuda.empty_cache()
    final_path = train_from_checkpoint(
        pruned_path,
        candidate,
        UNLEARN_DATASET,
        spec["lr"],
        spec["iters"],
        spec["ewc_lambda"],
        spec["grad_clip_norm"],
    )
    model_paths[candidate] = final_path
    pruned_path.unlink(missing_ok=True)
    log(
        "MODEL_COMPLETE",
        candidate=candidate,
        model_sha256=sha256(final_path),
        pruning=pruning_audits[candidate],
    )

set_seed(SYNTH_RECOVERY_SPEC["seed"])
recovery_candidate = SYNTH_RECOVERY_SPEC["candidate"]
recovery_path = train_from_checkpoint(
    model_paths[SYNTH_RECOVERY_SPEC["base_candidate"]],
    recovery_candidate,
    SYNTH_TRAIN_DATASET,
    SYNTH_RECOVERY_SPEC["lr"],
    SYNTH_RECOVERY_SPEC["iters"],
    SYNTH_RECOVERY_SPEC["ewc_lambda"],
    SYNTH_RECOVERY_SPEC["grad_clip_norm"],
)
model_paths[recovery_candidate] = recovery_path
pruning_audits[recovery_candidate] = {
    "mode": "inherits_public_bug_two_layer",
    "base_candidate": SYNTH_RECOVERY_SPEC["base_candidate"],
    "executed": pruning_audits[SYNTH_RECOVERY_SPEC["base_candidate"]]["executed"],
}

pd.DataFrame(all_training_rows).to_csv(TRAIN_LOG, index=False)
(RUN_DIR / "v8_pruning_audit.json").write_text(
    json.dumps(pruning_audits, indent=2),
    encoding="utf-8",
)

# %% [markdown]
# ## 7. Public-only model validation and ensemble membership lock

# %%
original_predictor = build_predictor(POISONED_WEIGHTS, threshold=0.02)
original_unlearn = predict_records(
    original_predictor,
    unlearn_records,
    "Original on unlearn",
)
original_synthetic = predict_records(
    original_predictor,
    synthetic_val_records,
    "Original on synthetic validation",
)


def xywh_to_xyxy(box):
    x, y, width, height = box
    return np.array([x, y, x + width, y + height], dtype=np.float32)


def target_confidence(prediction, target_box, threshold=0.10):
    boxes, scores = prediction
    if len(boxes) == 0:
        return 0.0
    overlaps = iou_matrix(
        xywh_to_xyxy(target_box).reshape(1, 4),
        boxes,
    )[0]
    if (float(overlaps.max()) if len(overlaps) else 0.0) < threshold:
        return 0.0
    return float(scores[int(overlaps.argmax())])


def validate_model(candidate, predictions_unlearn, predictions_synthetic):
    original_poison = []
    candidate_poison = []
    for record in unlearn_records:
        image_id = str(record["image_id"])
        box = poison_boxes[int(record["image_id"])][0]
        original_poison.append(target_confidence(original_unlearn[image_id], box))
        candidate_poison.append(target_confidence(predictions_unlearn[image_id], box))
    original_poison = np.asarray(original_poison, dtype=np.float32)
    candidate_poison = np.asarray(candidate_poison, dtype=np.float32)
    poison_ratio = float(
        np.median(candidate_poison / np.maximum(original_poison, 1e-6))
    )
    poison_fire = float(np.mean(candidate_poison >= 0.20))

    original_positive = []
    candidate_positive = []
    for record in synthetic_val_records:
        image_id = str(record["image_id"])
        box = synthetic_val_boxes[image_id]
        original_positive.append(target_confidence(original_synthetic[image_id], box, 0.20))
        candidate_positive.append(
            target_confidence(predictions_synthetic[image_id], box, 0.20)
        )
    original_positive = np.asarray(original_positive, dtype=np.float32)
    candidate_positive = np.asarray(candidate_positive, dtype=np.float32)
    eligible = original_positive >= 0.20
    if eligible.any():
        synthetic_recall = float(np.mean(candidate_positive[eligible] >= 0.20))
        synthetic_ratio = float(
            np.median(
                candidate_positive[eligible] / np.maximum(original_positive[eligible], 1e-6)
            )
        )
    else:
        synthetic_recall = 0.0
        synthetic_ratio = 0.0
    passes = (
        poison_ratio <= 0.50
        and poison_fire <= 0.55
        and synthetic_recall >= 0.65
        and 0.50 <= synthetic_ratio <= 1.50
    )
    proxy = (
        poison_ratio
        + poison_fire
        + max(0.0, 0.80 - synthetic_recall)
        + abs(math.log(max(synthetic_ratio, 1e-6)))
    )
    return {
        "experiment": next(
            (
                spec["experiment"]
                for spec in MODEL_SPECS + [SYNTH_RECOVERY_SPEC]
                if spec["candidate"] == candidate
            ),
            "UNKNOWN",
        ),
        "candidate": candidate,
        "poison_score_ratio_median": poison_ratio,
        "poison_fire_rate_020": poison_fire,
        "synthetic_eligible_count": int(eligible.sum()),
        "synthetic_recall_020": synthetic_recall,
        "synthetic_score_ratio_median": synthetic_ratio,
        "passes_model_gate": bool(passes),
        "model_proxy": float(proxy),
        "checkpoint": str(model_paths[candidate]),
    }


validation_predictions = {}
model_registry = []
for candidate, path in model_paths.items():
    predictor = build_predictor(path, threshold=0.02)
    predictions_unlearn = predict_records(
        predictor,
        unlearn_records,
        f"{candidate} on unlearn",
    )
    predictions_synthetic = predict_records(
        predictor,
        synthetic_val_records,
        f"{candidate} on synthetic validation",
    )
    validation_predictions[candidate] = {
        "unlearn": predictions_unlearn,
        "synthetic": predictions_synthetic,
    }
    model_registry.append(
        validate_model(candidate, predictions_unlearn, predictions_synthetic)
    )
    del predictor
    gc.collect()
    torch.cuda.empty_cache()

model_registry_frame = pd.DataFrame(model_registry).sort_values(
    ["passes_model_gate", "model_proxy"],
    ascending=[False, True],
)
model_registry_frame.to_csv(RUN_DIR / "v8_model_registry.csv", index=False)

ensemble_members = model_registry_frame.loc[
    model_registry_frame["passes_model_gate"], "candidate"
].tolist()
if len(ensemble_members) < 2:
    ensemble_members = model_registry_frame["candidate"].head(3).tolist()
if "ndr_bug15_seed42" not in ensemble_members:
    ensemble_members = ["ndr_bug15_seed42"] + ensemble_members
ensemble_members = list(dict.fromkeys(ensemble_members))[:4]
log("MODEL_SELECTION_LOCKED", ensemble_members=ensemble_members)

# %% [markdown]
# ## 8. Public-only post-processing grid

# %%
logwh = np.log(
    np.asarray(
        [annotation["bbox"][2:4] for annotation in coco["annotations"]],
        dtype=np.float32,
    )
)
geometry_mu = logwh.mean(0)
geometry_cov_inv = np.linalg.inv(np.cov(logwh.T) + 1e-3 * np.eye(2))


def geometry_probability(boxes):
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    widths_heights = np.stack(
        [boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]],
        axis=1,
    )
    delta = np.log(np.maximum(widths_heights, 1e-3)) - geometry_mu
    distance = np.einsum("ni,ij,nj->n", delta, geometry_cov_inv, delta)
    return np.exp(-0.25 * distance).astype(np.float32)


def survival_matrix(original, repairs):
    boxes, scores = original
    matrix = np.zeros((len(repairs), len(boxes)), dtype=np.float32)
    for row_index, repair in enumerate(repairs):
        repair_boxes, repair_scores = repair
        if len(boxes) == 0 or len(repair_boxes) == 0:
            continue
        overlaps = iou_matrix(boxes, repair_boxes)
        best_index = overlaps.argmax(1)
        best_iou = overlaps[np.arange(len(boxes)), best_index]
        matched = best_iou >= 0.50
        matrix[row_index, matched] = (
            repair_scores[best_index[matched]]
            / np.maximum(scores[matched], 1e-6)
        )
    return np.clip(matrix, 0.0, 1.5)


def aggregate_survival(matrix, mode):
    clipped = np.clip(matrix, 0.0, 1.0)
    if len(clipped) == 0:
        return np.zeros(clipped.shape[1], dtype=np.float32)
    if mode == "max_survival":
        return clipped.max(0)
    if mode == "mean_survival":
        return clipped.mean(0)
    if mode == "median_survival":
        return np.median(clipped, axis=0)
    if mode == "single_exact":
        return clipped[0]
    raise ValueError(mode)


def remap_predictions(original, repairs, spec):
    boxes, scores = original
    if len(boxes) == 0:
        return boxes.copy(), scores.copy()
    survival = aggregate_survival(survival_matrix(original, repairs), spec["aggregator"])
    confidence_drop = np.clip(1.0 - survival, 0.0, 1.0)
    geometry = geometry_probability(boxes)
    geometry_weight = float(spec["geometry_weight"])
    poison_probability = (
        (1.0 - geometry_weight) * confidence_drop + geometry_weight * geometry
    )
    new_scores = np.zeros(len(scores), dtype=np.float32)
    for index, (score, probability) in enumerate(zip(scores, poison_probability)):
        if score < float(spec["min_keep"]):
            continue
        if probability >= float(spec["p_hi"]):
            new_scores[index] = float(spec["eps"])
        elif probability <= float(spec["p_lo"]):
            new_scores[index] = float(score)
        else:
            fraction = (
                probability - float(spec["p_lo"])
            ) / max(float(spec["p_hi"]) - float(spec["p_lo"]), 1e-6)
            new_scores[index] = max(
                float(spec["eps"]),
                float(score) * (1.0 - fraction),
            )
    keep = new_scores > 0
    demoted = np.where((new_scores <= float(spec["eps"]) + 1e-8) & keep)[0]
    strong = np.where(new_scores > 0.20)[0]
    if len(demoted) and len(strong):
        overlaps = iou_matrix(boxes[demoted], boxes[strong]).max(1)
        keep[demoted[overlaps >= 0.20]] = False
    return boxes[keep], new_scores[keep]


def public_clean_reference(original_predictions, target_boxes_by_id):
    reference = {}
    for image_id, (boxes, scores) in original_predictions.items():
        keep = np.ones(len(boxes), dtype=bool)
        target = target_boxes_by_id.get(image_id)
        if target is not None and len(boxes):
            overlap = iou_matrix(
                xywh_to_xyxy(target).reshape(1, 4),
                boxes,
            )[0]
            keep &= overlap < 0.10
        reference[image_id] = (boxes[keep], scores[keep])
    return reference


poison_target_by_id = {
    str(record["image_id"]): poison_boxes[int(record["image_id"])][0]
    for record in unlearn_records
}
pseudo_clean_unlearn = public_clean_reference(original_unlearn, poison_target_by_id)
pseudo_clean_synthetic = original_synthetic


def apply_spec_to_validation(original_predictions, split, spec):
    output = {}
    for image_id, original in original_predictions.items():
        repairs = [
            validation_predictions[candidate][split][image_id]
            for candidate in ensemble_members
        ]
        output[image_id] = remap_predictions(original, repairs, spec)
    return output


def output_gate_metrics(unlearn_output, synthetic_output):
    original_poison = []
    participant_poison = []
    for record in unlearn_records:
        image_id = str(record["image_id"])
        box = poison_target_by_id[image_id]
        original_poison.append(target_confidence(original_unlearn[image_id], box))
        participant_poison.append(target_confidence(unlearn_output[image_id], box))
    original_poison = np.asarray(original_poison)
    participant_poison = np.asarray(participant_poison)
    poison_ratio = float(
        np.median(participant_poison / np.maximum(original_poison, 1e-6))
    )
    poison_fire = float(np.mean(participant_poison >= 0.20))

    original_positive = []
    participant_positive = []
    for record in synthetic_val_records:
        image_id = str(record["image_id"])
        box = synthetic_val_boxes[image_id]
        original_positive.append(target_confidence(original_synthetic[image_id], box, 0.20))
        participant_positive.append(
            target_confidence(synthetic_output[image_id], box, 0.20)
        )
    original_positive = np.asarray(original_positive)
    participant_positive = np.asarray(participant_positive)
    eligible = original_positive >= 0.20
    recall = float(np.mean(participant_positive[eligible] >= 0.20)) if eligible.any() else 0
    ratio = (
        float(
            np.median(
                participant_positive[eligible]
                / np.maximum(original_positive[eligible], 1e-6)
            )
        )
        if eligible.any()
        else 0.0
    )
    return poison_ratio, poison_fire, int(eligible.sum()), recall, ratio


candidate_specs = []
for values in itertools.product(*POSTPROCESS_GRID.values()):
    spec = dict(zip(POSTPROCESS_GRID.keys(), values))
    candidate_specs.append(spec)

exact_control_spec = {
    "aggregator": "single_exact",
    "geometry_weight": 0.10,
    "p_lo": 0.25,
    "p_hi": 0.55,
    "min_keep": 0.20,
    "eps": 0.01,
}

postprocess_rows = []
for candidate_index, spec in enumerate(candidate_specs):
    unlearn_output = apply_spec_to_validation(original_unlearn, "unlearn", spec)
    synthetic_output = apply_spec_to_validation(original_synthetic, "synthetic", spec)
    poison_ratio, poison_fire, eligible_count, recall, ratio = output_gate_metrics(
        unlearn_output,
        synthetic_output,
    )
    public_macadd = macadd(pseudo_clean_unlearn, unlearn_output)
    synthetic_macadd = macadd(pseudo_clean_synthetic, synthetic_output)
    passes = (
        poison_ratio <= 0.40
        and poison_fire <= 0.40
        and recall >= 0.75
        and 0.60 <= ratio <= 1.40
    )
    postprocess_rows.append(
        {
            "experiment": "E60",
            "candidate": f"v8_gate_{candidate_index:03d}",
            **spec,
            "poison_score_ratio_median": poison_ratio,
            "poison_fire_rate_020": poison_fire,
            "synthetic_eligible_count": eligible_count,
            "synthetic_recall_020": recall,
            "synthetic_score_ratio_median": ratio,
            "public_unlearn_macadd": public_macadd,
            "synthetic_retention_macadd": synthetic_macadd,
            "selection_metric": public_macadd + 0.50 * synthetic_macadd,
            "passes_output_gate": bool(passes),
        }
    )

postprocess_frame = pd.DataFrame(postprocess_rows).sort_values(
    ["passes_output_gate", "selection_metric"],
    ascending=[False, True],
)
postprocess_frame.to_csv(RUN_DIR / "v8_postprocess_registry.csv", index=False)

passing = postprocess_frame[postprocess_frame["passes_output_gate"]]
if len(passing):
    best_row = passing.iloc[0]
else:
    best_row = postprocess_frame.iloc[0]
best_spec = {
    key: (
        str(best_row[key])
        if key == "aggregator"
        else float(best_row[key])
    )
    for key in POSTPROCESS_GRID
}

diverse_pool = postprocess_frame[
    (postprocess_frame["aggregator"] != best_spec["aggregator"])
    | (postprocess_frame["min_keep"].astype(float) != best_spec["min_keep"])
]
if len(passing):
    diverse_passing = diverse_pool[diverse_pool["passes_output_gate"]]
    diverse_row = diverse_passing.iloc[0] if len(diverse_passing) else diverse_pool.iloc[0]
else:
    diverse_row = diverse_pool.iloc[0]
diverse_spec = {
    key: (
        str(diverse_row[key])
        if key == "aggregator"
        else float(diverse_row[key])
    )
    for key in POSTPROCESS_GRID
}

SELECTION_LOCKED = True
selection_lock = {
    "status": "frozen_before_test_read",
    "score_direction": "lower_is_better",
    "ensemble_members": ensemble_members,
    "best_candidate": str(best_row["candidate"]),
    "best_spec": best_spec,
    "best_public_metrics": {
        key: (
            bool(best_row[key])
            if key == "passes_output_gate"
            else float(best_row[key])
        )
        for key in [
            "poison_score_ratio_median",
            "poison_fire_rate_020",
            "synthetic_recall_020",
            "synthetic_score_ratio_median",
            "public_unlearn_macadd",
            "synthetic_retention_macadd",
            "selection_metric",
            "passes_output_gate",
        ]
    },
    "diverse_candidate": str(diverse_row["candidate"]),
    "diverse_spec": diverse_spec,
    "exact_control_spec": exact_control_spec,
    "selection_sources": RULE_GUARD["selection_sources"],
    "test_images_read": False,
    "test_predictions_used_for_selection": False,
    "leaderboard_used_for_selection": False,
    "competition_submission_created": False,
}
(RUN_DIR / "v8_selection_lock.json").write_text(
    json.dumps(selection_lock, indent=2),
    encoding="utf-8",
)
(RUN_DIR / "v8_candidate_specs.json").write_text(
    json.dumps(
        {
            "best": best_spec,
            "diverse": diverse_spec,
            "exact_control": exact_control_spec,
        },
        indent=2,
    ),
    encoding="utf-8",
)
log("SELECTION_LOCKED", selection=selection_lock)

fig, axis = plt.subplots(figsize=(10, 7))
colors = np.where(
    postprocess_frame["passes_output_gate"],
    "#2dd4bf",
    "#64748b",
)
axis.scatter(
    postprocess_frame["synthetic_retention_macadd"],
    postprocess_frame["public_unlearn_macadd"],
    c=colors,
    alpha=0.65,
)
axis.scatter(
    [float(best_row["synthetic_retention_macadd"])],
    [float(best_row["public_unlearn_macadd"])],
    marker="*",
    s=220,
    color="#f59e0b",
    label="Frozen best",
)
axis.set_xlabel("Synthetic retention maCADD (lower)")
axis.set_ylabel("Public unlearn pseudo-clean maCADD (lower)")
axis.set_title("NDR Frontier V8: suppression vs retention")
axis.grid(alpha=0.2)
axis.legend()
fig.tight_layout()
fig.savefig(RUN_DIR / "v8_frontier.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 9. Frozen test inference
#
# The selection lock exists before this cell. No per-image test statistic can
# modify the ensemble, thresholds or finalist ordering.

# %%
if not SELECTION_LOCKED or not (RUN_DIR / "v8_selection_lock.json").exists():
    raise RuntimeError("Selection must be frozen before test inference")

sample_submission = pd.read_csv(SAMPLE_SUB_PATH, dtype={"image_id": str})
test_files = sorted(TEST_DIR.glob("*.png"))
if len(test_files) != 2000:
    raise RuntimeError(f"Expected 2000 test images, found {len(test_files)}")

repair_predictors = {
    candidate: build_predictor(model_paths[candidate], threshold=0.02)
    for candidate in ensemble_members
}
exact_predictor = repair_predictors.get("ndr_bug15_seed42")
if exact_predictor is None:
    exact_predictor = build_predictor(model_paths["ndr_bug15_seed42"], threshold=0.02)

best_predictions = {}
diverse_predictions = {}
control_predictions = {}
test_diagnostics = []
inference_started = time.time()
for image_index, image_path in enumerate(
    tqdm(test_files, desc="Frozen test inference"),
    start=1,
):
    image_id = image_path.stem
    image = load_image(image_path)
    original_instances = original_predictor(image)["instances"].to("cpu")
    original = (
        original_instances.pred_boxes.tensor.numpy().astype(np.float32),
        original_instances.scores.numpy().astype(np.float32),
    )
    repaired = []
    repaired_by_candidate = {}
    for candidate, predictor in repair_predictors.items():
        instances = predictor(image)["instances"].to("cpu")
        prediction = (
            instances.pred_boxes.tensor.numpy().astype(np.float32),
            instances.scores.numpy().astype(np.float32),
        )
        repaired.append(prediction)
        repaired_by_candidate[candidate] = prediction
    best = remap_predictions(original, repaired, best_spec)
    diverse = remap_predictions(original, repaired, diverse_spec)

    control_keep = original[1] >= 0.05
    control_original = (original[0][control_keep], original[1][control_keep])
    exact_repair_raw = repaired_by_candidate["ndr_bug15_seed42"]
    exact_repair_keep = exact_repair_raw[1] >= 0.05
    exact_repair = (
        exact_repair_raw[0][exact_repair_keep],
        exact_repair_raw[1][exact_repair_keep],
    )
    control = remap_predictions(
        control_original,
        [exact_repair],
        exact_control_spec,
    )
    best_predictions[image_id] = best
    diverse_predictions[image_id] = diverse
    control_predictions[image_id] = control
    test_diagnostics.append(
        {
            "image_id": image_id,
            "original_candidates": int(len(original[1])),
            "best_boxes": int(len(best[1])),
            "diverse_boxes": int(len(diverse[1])),
            "control_boxes": int(len(control[1])),
            "best_confidence_sum": float(best[1].sum()),
            "diverse_confidence_sum": float(diverse[1].sum()),
            "control_confidence_sum": float(control[1].sum()),
        }
    )
    if image_index % 100 == 0 or image_index == len(test_files):
        log(
            "INFERENCE_PROGRESS",
            completed=image_index,
            total=len(test_files),
            elapsed_sec=round(time.time() - inference_started, 1),
        )

pd.DataFrame(test_diagnostics).to_csv(RUN_DIR / "v8_test_diagnostics.csv", index=False)

# %% [markdown]
# ## 10. Submission packaging and integrity audit

# %%
def format_predictions(boxes, scores):
    parts = []
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1 = float(np.clip(x1, 0, IMG_W))
        y1 = float(np.clip(y1, 0, IMG_H))
        x2 = float(np.clip(x2, 0, IMG_W))
        y2 = float(np.clip(y2, 0, IMG_H))
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0 or not (0 < score <= 1):
            continue
        parts.extend(
            [
                f"{float(score):.6f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{width:.2f}",
                f"{height:.2f}",
            ]
        )
    return " ".join(parts) or " "


def write_submission(predictions, output_path):
    frame = sample_submission.copy()
    frame["prediction_string"] = frame["image_id"].map(
        lambda image_id: format_predictions(
            *predictions.get(
                str(image_id),
                (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros(0, dtype=np.float32),
                ),
            )
        )
    )
    frame.to_csv(output_path, index=False)
    return frame


def validate_submission(frame):
    assert list(frame.columns) == list(sample_submission.columns)
    assert len(frame) == 2000
    assert frame["image_id"].astype(str).is_unique
    assert (
        frame["image_id"].astype(str).tolist()
        == sample_submission["image_id"].astype(str).tolist()
    )
    assert frame["prediction_string"].isna().sum() == 0
    total_boxes = 0
    empty_rows = 0
    for prediction in frame["prediction_string"].astype(str):
        if not prediction.strip():
            empty_rows += 1
            continue
        values = [float(value) for value in prediction.split()]
        assert len(values) % 5 == 0
        for offset in range(0, len(values), 5):
            confidence, x, y, width, height = values[offset:offset + 5]
            assert 0 < confidence <= 1
            assert 0 <= x <= IMG_W and 0 <= y <= IMG_H
            assert width > 0 and height > 0
            assert x + width <= IMG_W + 0.05
            assert y + height <= IMG_H + 0.05
            total_boxes += 1
    return total_boxes, empty_rows


submission_paths = {
    "best": Path("/kaggle/working/submission_best.csv"),
    "diverse": Path("/kaggle/working/submission_diverse.csv"),
    "exact_control": Path("/kaggle/working/submission_ndr229_control.csv"),
}
submission_frames = {
    "best": write_submission(best_predictions, submission_paths["best"]),
    "diverse": write_submission(diverse_predictions, submission_paths["diverse"]),
    "exact_control": write_submission(
        control_predictions,
        submission_paths["exact_control"],
    ),
}
submission_audit = {}
for name, frame in submission_frames.items():
    total_boxes, empty_rows = validate_submission(frame)
    submission_audit[name] = {
        "path": str(submission_paths[name]),
        "sha256": sha256(submission_paths[name]),
        "rows": int(len(frame)),
        "unique_image_ids": int(frame["image_id"].astype(str).nunique()),
        "total_boxes": int(total_boxes),
        "empty_rows": int(empty_rows),
    }

shutil.copyfile(submission_paths["best"], "/kaggle/working/submission.csv")
assert sha256("/kaggle/working/submission.csv") == submission_audit["best"]["sha256"]

model_manifest = {
    candidate: {
        "path": str(path),
        "sha256": sha256(path),
    }
    for candidate, path in model_paths.items()
}
(RUN_DIR / "v8_model_manifest.json").write_text(
    json.dumps(model_manifest, indent=2),
    encoding="utf-8",
)

final_report = {
    "status": "complete",
    "bundle": "NDR_FRONTIER_V8",
    "score_direction": "lower_is_better",
    "incumbent_public_score": 229.2314,
    "ensemble_members": ensemble_members,
    "model_candidate_count": int(len(model_registry_frame)),
    "postprocess_candidate_count": int(len(postprocess_frame)),
    "passing_model_candidates": int(model_registry_frame["passes_model_gate"].sum()),
    "passing_postprocess_candidates": int(
        postprocess_frame["passes_output_gate"].sum()
    ),
    "selection_lock": selection_lock,
    "submission_audit": submission_audit,
    "model_manifest": model_manifest,
    "rule_7a_guard_passed": True,
    "test_predictions_used_for_selection": False,
    "leaderboard_used_for_selection": False,
    "competition_submission_created": False,
}
(RUN_DIR / "v8_final_report.json").write_text(
    json.dumps(final_report, indent=2),
    encoding="utf-8",
)
log("RUN_COMPLETE", report=final_report)
print(json.dumps(final_report, indent=2))
