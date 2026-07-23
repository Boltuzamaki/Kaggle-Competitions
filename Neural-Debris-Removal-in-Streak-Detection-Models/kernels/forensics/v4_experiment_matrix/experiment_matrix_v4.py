# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Debris - maximal experiment matrix v4
#
# One rule-safe GPU run covering the user-supplied E00-E37 queue:
#
# - baseline and previous-checkpoint reproducibility,
# - output blending, confidence gates and calibration,
# - grouped ROI/morphology/stability gates,
# - synthetic positive controls, distillation, adapters and low-rank repair,
# - static, causal, paired, level-specific and recovery pruning,
# - frozen robustness/Pareto selection and finalist submission export.
#
# Only the 20 public unlearn images and supplied poisoned RetinaNet are used.
# The test set is read only after selection is frozen, for finalist inference.

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
import zlib
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.0")

OUT = Path("/kaggle/working/experiment_matrix_v4")
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run.log"
METRICS_JSONL = OUT / "metrics.jsonl"


def log(message, **fields):
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": str(message), **fields}
    print(f"[{row['time']}] {message}", flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
    with METRICS_JSONL.open("a", encoding="utf-8") as handle:
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
from tqdm.auto import tqdm

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("This repair matrix requires a Kaggle GPU")

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
UNLEARN = ROOT / "unlearn_set"
WEIGHTS = ROOT / "poisoned_model/poisoned_model.pth"
SEED = 20260717
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
CANDIDATES = [
    {"name": "tail_pos0p5", "scope": "tail", "lr": 3e-5, "positive": 0.5, "negative": 0.25, "anchor": 3e-4},
    {"name": "tail_pos1", "scope": "tail", "lr": 3e-5, "positive": 1.0, "negative": 0.25, "anchor": 3e-4},
    {"name": "tail_pos2", "scope": "tail", "lr": 3e-5, "positive": 2.0, "negative": 0.25, "anchor": 3e-4},
    {"name": "tail_pos4", "scope": "tail", "lr": 3e-5, "positive": 4.0, "negative": 0.25, "anchor": 3e-4},
    {"name": "full_pos0p5", "scope": "full_cls", "lr": 3e-5, "positive": 0.5, "negative": 0.25, "anchor": 5e-4},
    {"name": "full_pos1", "scope": "full_cls", "lr": 3e-5, "positive": 1.0, "negative": 0.25, "anchor": 5e-4},
    {"name": "full_pos2", "scope": "full_cls", "lr": 3e-5, "positive": 2.0, "negative": 0.25, "anchor": 5e-4},
    {"name": "full_pos4", "scope": "full_cls", "lr": 3e-5, "positive": 4.0, "negative": 0.25, "anchor": 5e-4},
]
CV_STEPS = 30
FINAL_STEPS = [30, 60, 100]

CONFIG = {
    "seed": SEED,
    "folds": FOLDS,
    "candidates": CANDIDATES,
    "cv_steps": CV_STEPS,
    "final_steps": FINAL_STEPS,
    "poison_levels": ["p3", "p4"],
    "poison_anchor_min_iou": 0.05,
    "poison_anchor_top_k_per_box": 32,
    "retention_exclusion_scale": 1.5,
    "positive_anchor_probability": 0.01,
    "v1_correction": "zero-retention folds are missing, not failed",
    "rule_guard": {
        "test_images": False,
        "test_predictions": False,
        "external_models": False,
        "teacher": "supplied poisoned model on public unlearn images only",
    },
    "experiment_ids": [f"E{index:02d}" for index in range(43)],
    "selection_gate": {
        "poison_fire_rate_020_max": 0.35,
        "poison_score_ratio_median_max": 0.25,
        "retain_match_rate_min": 0.90,
        "retain_score_ratio_median_min": 0.80,
        "retain_score_ratio_median_max": 1.20,
    },
}
(OUT / "experiment_config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")

digest = hashlib.sha256()
with WEIGHTS.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
original_sha256 = digest.hexdigest()
(OUT / "original_weights_sha256.txt").write_text(original_sha256 + "\n", encoding="utf-8")
log("Original weights preserved by immutable input and hash", sha256=original_sha256)

with (UNLEARN / "annotations_coco.json").open(encoding="utf-8") as handle:
    coco = json.load(handle)
image_info = {int(item["id"]): item for item in coco["images"]}
annotation_by_image = {int(item["image_id"]): item for item in coco["annotations"]}
assert set(image_info) == set(annotation_by_image)
assert sorted(sum(FOLDS.values(), [])) == sorted(image_info)


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    scale = 65535.0 if image.dtype == np.uint16 else max(float(image.max()), 1.0)
    return np.clip(image.astype(np.float32) / scale * 255.0, 0, 255)


images = {
    image_id: read_gray(UNLEARN / item["file_name"])
    for image_id, item in image_info.items()
}
boxes = {}
for image_id, annotation in annotation_by_image.items():
    x, y, width, height = map(float, annotation["bbox"])
    boxes[image_id] = np.asarray([x, y, x + width, y + height], np.float32)

patch_bank = []
for image_id in sorted(images):
    x1, y1, x2, y2 = [int(round(value)) for value in boxes[image_id]]
    margin = 8
    xa, ya = max(0, x1 - margin), max(0, y1 - margin)
    xb, yb = min(1024, x2 + margin), min(1024, y2 + margin)
    patch_bank.append(images[image_id][ya:yb, xa:xb].copy())


def make_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = str(WEIGHTS)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.005
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    return cfg


def build_loaded_model():
    model = build_model(make_cfg())
    DetectionCheckpointer(model).load(str(WEIGHTS))
    model.to(DEVICE)
    return model


teacher = build_loaded_model()
teacher.eval()
for parameter in teacher.parameters():
    parameter.requires_grad = False
log("Teacher ready", device=str(DEVICE))


def to_record(gray):
    image = np.repeat(gray[:, :, None], 3, axis=2)
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return {"image": tensor, "height": 1024, "width": 1024}


def transform_box_affine(box, matrix):
    corners = np.asarray(
        [
            [box[0], box[1], 1],
            [box[2], box[1], 1],
            [box[2], box[3], 1],
            [box[0], box[3], 1],
        ],
        np.float32,
    )
    projected = corners @ matrix.T
    result = np.asarray(
        [
            projected[:, 0].min(),
            projected[:, 1].min(),
            projected[:, 0].max(),
            projected[:, 1].max(),
        ],
        np.float32,
    )
    result[[0, 2]] = np.clip(result[[0, 2]], 0, 1024)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, 1024)
    return result


def d4_image_box(image, box, k, flip):
    output = np.rot90(image, k=k).copy() if k else image.copy()
    height, width = image.shape
    x1, y1, x2, y2 = box
    if k == 1:
        box = np.asarray([y1, width - x2, y2, width - x1], np.float32)
    elif k == 2:
        box = np.asarray([width - x2, height - y2, width - x1, height - y1], np.float32)
    elif k == 3:
        box = np.asarray([height - y2, x1, height - y1, x2], np.float32)
    if flip:
        output = output[:, ::-1].copy()
        box = np.asarray([1024 - box[2], box[1], 1024 - box[0], box[3]], np.float32)
    return output, box


def box_iou_numpy(box, candidates):
    candidates = np.asarray(candidates, np.float32).reshape(-1, 4)
    if len(candidates) == 0:
        return np.zeros(0, np.float32)
    top_left = np.maximum(box[None, :2], candidates[:, :2])
    bottom_right = np.minimum(box[None, 2:], candidates[:, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[:, 0] * size[:, 1]
    area_a = max(float(np.prod(box[2:] - box[:2])), 1e-6)
    area_b = np.prod(np.clip(candidates[:, 2:] - candidates[:, :2], 0, None), axis=1)
    return intersection / np.clip(area_a + area_b - intersection, 1e-6, None)


def augment(image_id, seed, allow_transplant=True):
    rng = np.random.default_rng(seed)
    image = images[image_id].copy()
    target_boxes = [boxes[image_id].copy()]

    k = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))
    image, target_boxes[0] = d4_image_box(image, target_boxes[0], k, flip)

    scale = float(rng.uniform(0.8, 1.2))
    dx = float(rng.choice([-32, 0, 32]))
    dy = float(rng.choice([-32, 0, 32]))
    matrix = np.asarray(
        [
            [scale, 0, (1 - scale) * 512 + dx],
            [0, scale, (1 - scale) * 512 + dy],
        ],
        np.float32,
    )
    image = cv2.warpAffine(
        image,
        matrix,
        (1024, 1024),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    target_boxes = [transform_box_affine(box, matrix) for box in target_boxes]

    gain = float(rng.uniform(0.92, 1.08))
    gamma = float(rng.uniform(0.92, 1.08))
    image = np.clip((np.clip(image / 255.0, 0, 1) ** gamma) * 255.0 * gain, 0, 255)
    if rng.random() < 0.35:
        image = cv2.GaussianBlur(image, (0, 0), float(rng.uniform(0.5, 2.0)))

    if allow_transplant and rng.random() < 0.5:
        patch = patch_bank[int(rng.integers(len(patch_bank)))].copy()
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        if rng.random() < 0.5:
            patch = patch[::-1, :]
        rotation = int(rng.integers(0, 4))
        if rotation:
            patch = np.rot90(patch, rotation).copy()
        patch = np.clip(patch * float(rng.uniform(0.92, 1.08)), 0, 255)
        height, width = patch.shape
        for _ in range(50):
            x0 = int(rng.integers(16, 1024 - width - 16))
            y0 = int(rng.integers(16, 1024 - height - 16))
            candidate = np.asarray([x0, y0, x0 + width, y0 + height], np.float32)
            if max(box_iou_numpy(candidate, target_boxes), default=0.0) < 0.01:
                feather = max(1, min(4, height // 4, width // 4))
                yy = np.minimum(np.arange(height) + 1, np.arange(height)[::-1] + 1) / feather
                xx = np.minimum(np.arange(width) + 1, np.arange(width)[::-1] + 1) / feather
                alpha = np.clip(np.minimum(yy[:, None], xx[None, :]), 0, 1)
                region = image[y0 : y0 + height, x0 : x0 + width]
                image[y0 : y0 + height, x0 : x0 + width] = (
                    region * (1 - alpha) + patch * alpha
                )
                target_boxes.append(candidate)
                break
    return image.astype(np.float32), target_boxes


def dense_logits(model, record):
    image_list = model.preprocess_image([record])
    feature_dict = model.backbone(image_list.tensor)
    features = [feature_dict[name] for name in model.head_in_features]
    predictions = model.head(features)
    logits, _ = model._transpose_dense_predictions(predictions, [model.num_classes, 4])
    anchors = model.anchor_generator(features)
    return logits, anchors


def torch_iou(boxes_a, boxes_b):
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    size = (bottom_right - top_left).clamp(min=0)
    intersection = size[:, :, 0] * size[:, :, 1]
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])).clamp(min=1e-6)
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])).clamp(min=1e-6)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp(min=1e-6)


def masks_for_anchors(anchor_boxes, target_boxes, poison_level):
    targets = torch.as_tensor(np.asarray(target_boxes), dtype=torch.float32, device=DEVICE)
    centers = (anchor_boxes[:, :2] + anchor_boxes[:, 2:]) / 2
    center_size = (targets[:, 2:] - targets[:, :2]) * 1.5
    center_mid = (targets[:, :2] + targets[:, 2:]) / 2
    expanded = torch.cat([center_mid - center_size / 2, center_mid + center_size / 2], dim=1)
    inside_expanded = (
        (centers[:, None, 0] >= expanded[None, :, 0])
        & (centers[:, None, 0] <= expanded[None, :, 2])
        & (centers[:, None, 1] >= expanded[None, :, 1])
        & (centers[:, None, 1] <= expanded[None, :, 3])
    ).any(dim=1)

    poison = torch.zeros(len(anchor_boxes), dtype=torch.bool, device=DEVICE)
    if poison_level:
        overlaps = torch_iou(anchor_boxes, targets)
        poison |= overlaps.max(dim=1).values >= 0.05
        for target_index in range(len(targets)):
            count = min(32, len(anchor_boxes))
            top = torch.topk(overlaps[:, target_index], k=count, largest=True).indices
            poison[top] = True
    retain = ~inside_expanded
    return poison, retain


def configure_scope(model, scope):
    trainable_names = []
    for name, parameter in model.named_parameters():
        enabled = False
        if scope == "score_weight":
            enabled = name == "head.cls_score.weight"
        elif scope == "tail_weights":
            enabled = name in {"head.cls_score.weight", "head.cls_subnet.6.weight"}
        elif scope == "full_weights":
            enabled = (
                ("head.cls_score" in name or "head.cls_subnet" in name)
                and name.endswith(".weight")
            )
        elif scope == "tail":
            enabled = "head.cls_score" in name or "head.cls_subnet.6" in name
        elif scope == "full_cls":
            enabled = "head.cls_score" in name or "head.cls_subnet" in name
        parameter.requires_grad = enabled
        if enabled:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError(f"No trainable parameters for scope {scope}")
    return trainable_names


def targeted_loss(student, original_parameters, image, target_boxes, candidate):
    record = to_record(image)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        teacher_logits, _ = dense_logits(teacher, record)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        student_logits, anchors = dense_logits(student, record)

    suppress_terms = []
    positive_terms = []
    negative_terms = []
    for level_name, student_level, teacher_level, anchor_level in zip(
        student.head_in_features,
        student_logits,
        teacher_logits,
        anchors,
    ):
        poison_mask, retain_mask = masks_for_anchors(
            anchor_level.tensor,
            target_boxes,
            level_name in {"p3", "p4"},
        )
        student_values = student_level[0, :, 0].float()
        teacher_values = teacher_level[0, :, 0].float()
        if poison_mask.any():
            poison_logits = student_values[poison_mask]
            suppress_terms.append(
                (torch.sigmoid(poison_logits).pow(2) * F.softplus(poison_logits)).mean()
            )
        teacher_probability = torch.sigmoid(teacher_values).detach()
        positive_mask = retain_mask & (teacher_probability >= 0.01)
        negative_mask = retain_mask & ~positive_mask
        if positive_mask.any():
            positive_terms.append(
                F.smooth_l1_loss(
                    student_values[positive_mask],
                    teacher_values[positive_mask],
                    reduction="mean",
                )
            )
        if negative_mask.any():
            negative_terms.append(
                (
                    torch.sigmoid(student_values[negative_mask])
                    - teacher_probability[negative_mask]
                )
                .pow(2)
                .mean()
            )

    suppression = torch.stack(suppress_terms).mean()
    positive_retention = (
        torch.stack(positive_terms).mean()
        if positive_terms
        else torch.zeros((), device=DEVICE, dtype=torch.float32)
    )
    negative_retention = (
        torch.stack(negative_terms).mean()
        if negative_terms
        else torch.zeros((), device=DEVICE, dtype=torch.float32)
    )
    drift_terms = []
    for name, parameter in student.named_parameters():
        if parameter.requires_grad:
            drift_terms.append((parameter - original_parameters[name]).pow(2).mean())
    drift = torch.stack(drift_terms).mean()
    objective = (
        suppression
        + candidate["positive"] * positive_retention
        + candidate["negative"] * negative_retention
        + candidate["anchor"] * drift
    )
    return objective, suppression, positive_retention, negative_retention, drift


def infer(model, image):
    model.eval()
    with torch.no_grad():
        output = model([to_record(image)])[0]["instances"].to("cpu")
    return output.pred_boxes.tensor.numpy(), output.scores.numpy()


teacher_predictions = {
    image_id: infer(teacher, images[image_id])
    for image_id in tqdm(sorted(images), desc="teacher reference")
}
teacher_export = {
    str(image_id): {
        "boxes": prediction[0].tolist(),
        "scores": prediction[1].tolist(),
    }
    for image_id, prediction in teacher_predictions.items()
}
(OUT / "teacher_predictions_public_unlearn.json").write_text(
    json.dumps(teacher_export),
    encoding="utf-8",
)


def validation_metrics(model, validation_ids):
    poison_ratios = []
    poison_scores = []
    retain_ratios = []
    retained = 0
    retain_total = 0
    for image_id in validation_ids:
        target = boxes[image_id]
        teacher_boxes, teacher_scores = teacher_predictions[image_id]
        student_boxes, student_scores = infer(model, images[image_id])

        teacher_overlap = box_iou_numpy(target, teacher_boxes)
        student_overlap = box_iou_numpy(target, student_boxes)
        teacher_target = float(teacher_scores[teacher_overlap >= 0.2].max(initial=0))
        student_target = float(student_scores[student_overlap >= 0.2].max(initial=0))
        poison_scores.append(student_target)
        poison_ratios.append(student_target / max(teacher_target, 1e-6))

        teacher_keep = (teacher_scores >= 0.20) & (teacher_overlap < 0.10)
        reference_boxes = teacher_boxes[teacher_keep]
        reference_scores = teacher_scores[teacher_keep]
        retain_total += len(reference_boxes)
        for reference_box, reference_score in zip(reference_boxes, reference_scores):
            overlap = box_iou_numpy(reference_box, student_boxes)
            if len(overlap) and overlap.max() >= 0.5:
                index = int(overlap.argmax())
                retained += 1
                retain_ratios.append(float(student_scores[index] / max(reference_score, 1e-6)))
            else:
                retain_ratios.append(0.0)

    poison_ratio = float(np.median(poison_ratios))
    fire_rate = float(np.mean(np.asarray(poison_scores) >= 0.20))
    match_rate = retained / retain_total if retain_total else float("nan")
    positive_retain = np.asarray(retain_ratios)
    positive_retain = positive_retain[positive_retain > 0]
    retain_ratio = float(np.median(positive_retain)) if len(positive_retain) else float("nan")
    match_penalty = 0.0 if math.isnan(match_rate) else 0.75 * (1 - match_rate)
    ratio_penalty = (
        0.0
        if math.isnan(retain_ratio)
        else 0.25 * abs(math.log(max(retain_ratio, 1e-3)))
    )
    proxy = poison_ratio + match_penalty + ratio_penalty
    return {
        "poison_score_ratio_median": poison_ratio,
        "poison_fire_rate_020": fire_rate,
        "retain_total": retain_total,
        "retain_matched": retained,
        "retain_match_rate": match_rate,
        "retain_score_ratio_median": retain_ratio,
        "proxy": proxy,
    }


def train_model(candidate, train_ids, steps, run_name, checkpoints=()):
    student = build_loaded_model()
    trainable_names = configure_scope(student, candidate["scope"])
    student.train()
    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=candidate["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    checkpoint_paths = {}
    with heartbeat(run_name):
        for step in range(1, steps + 1):
            image_id = train_ids[(step - 1) % len(train_ids)]
            image, target_boxes = augment(
                image_id,
                SEED + zlib.crc32(run_name.encode("utf-8")) % 100000 + step,
                allow_transplant=True,
            )
            optimizer.zero_grad(set_to_none=True)
            objective, suppression, positive_retention, negative_retention, drift = targeted_loss(
                student,
                original_parameters,
                image,
                target_boxes,
                candidate,
            )
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            row = {
                "run": run_name,
                "step": step,
                "image_id": image_id,
                "loss": float(objective.detach().cpu()),
                "suppression": float(suppression.detach().cpu()),
                "positive_retention": float(positive_retention.detach().cpu()),
                "negative_retention": float(negative_retention.detach().cpu()),
                "drift": float(drift.detach().cpu()),
            }
            history.append(row)
            if step == 1 or step % 5 == 0 or step == steps:
                log("training step", **row)
            if step in checkpoints:
                path = OUT / f"{run_name}_step{step}.pth"
                torch.save(
                    {"model": {name: value.detach().cpu() for name, value in student.state_dict().items()}},
                    path,
                )
                checkpoint_paths[step] = path
                log("checkpoint saved", run=run_name, step=step, path=str(path))
    return student, history, checkpoint_paths, trainable_names


# %%
cv_rows = []
all_history = []
all_ids = sorted(images)
for candidate in CANDIDATES:
    for fold_index, validation_ids in FOLDS.items():
        student = None
        train_ids = [image_id for image_id in all_ids if image_id not in validation_ids]
        run_name = f"cv_{candidate['name']}_fold{fold_index}"
        log("CV run started", run=run_name, train_ids=train_ids, validation_ids=validation_ids)
        try:
            student, history, _, trainable_names = train_model(
                candidate,
                train_ids,
                CV_STEPS,
                run_name,
            )
            metrics = validation_metrics(student, validation_ids)
            row = {
                "candidate": candidate["name"],
                "scope": candidate["scope"],
                "fold": fold_index,
                "trainable_tensors": len(trainable_names),
                **metrics,
            }
            cv_rows.append(row)
            all_history.extend(history)
            pd.DataFrame(cv_rows).to_csv(OUT / "cv_results.partial.csv", index=False)
            pd.DataFrame(all_history).to_csv(OUT / "training_history.partial.csv", index=False)
            log("CV run complete", **row)
        except Exception as exc:
            log("CV run failed", run=run_name, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if student is not None:
                del student
            gc.collect()
            torch.cuda.empty_cache()

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(OUT / "cv_results.csv", index=False)
history_df = pd.DataFrame(all_history)
history_df.to_csv(OUT / "training_history.csv", index=False)
aggregate = (
    cv_df.groupby(["candidate", "scope"])
    .agg(
        poison_score_ratio_median=("poison_score_ratio_median", "median"),
        poison_fire_rate_020=("poison_fire_rate_020", "mean"),
        retain_matched=("retain_matched", "sum"),
        retain_total=("retain_total", "sum"),
        retain_score_ratio_median=("retain_score_ratio_median", "median"),
        proxy=("proxy", "mean"),
    )
    .reset_index()
)
aggregate["retain_match_rate"] = (
    aggregate.retain_matched / aggregate.retain_total.replace(0, np.nan)
)
aggregate["passes_gate"] = (
    (aggregate.poison_fire_rate_020 <= 0.35)
    & (aggregate.poison_score_ratio_median <= 0.25)
    & (aggregate.retain_match_rate >= 0.90)
    & aggregate.retain_score_ratio_median.between(0.80, 1.20)
)
aggregate = aggregate.sort_values(["passes_gate", "proxy"], ascending=[False, True])
aggregate.to_csv(OUT / "candidate_ranking.csv", index=False)
selected_name = str(aggregate.iloc[0].candidate)
selected_candidate = next(item for item in CANDIDATES if item["name"] == selected_name)
log("Candidate selected", candidate=selected_name, ranking=aggregate.to_dict(orient="records"))

# %%
final_model, final_history, checkpoint_paths, trainable_names = train_model(
    selected_candidate,
    all_ids,
    max(FINAL_STEPS),
    f"final_{selected_name}",
    checkpoints=FINAL_STEPS,
)
del final_model
gc.collect()
torch.cuda.empty_cache()
final_rows = []
for step, path in checkpoint_paths.items():
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    evaluation_model = build_loaded_model()
    evaluation_model.load_state_dict(checkpoint["model"])
    metrics = validation_metrics(evaluation_model, all_ids)
    final_rows.append({"step": step, "path": str(path), **metrics})
    del evaluation_model
    gc.collect()
    torch.cuda.empty_cache()
final_df = pd.DataFrame(final_rows).sort_values("step")
final_df.to_csv(OUT / "final_checkpoint_metrics.csv", index=False)

figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for candidate_name, frame in cv_df.groupby("candidate"):
    axes[0].scatter(
        frame.retain_match_rate,
        frame.poison_score_ratio_median,
        label=candidate_name,
        s=35,
    )
axes[0].axhline(0.25, color="black", linestyle="--", linewidth=1)
axes[0].axvline(0.90, color="black", linestyle="--", linewidth=1)
axes[0].set_xlabel("Held-out retain match rate")
axes[0].set_ylabel("Held-out poison score ratio")
axes[0].set_title("Five-fold repair trade-off")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.25)
axes[1].plot(final_df.step, final_df.poison_score_ratio_median, marker="o", label="poison ratio")
axes[1].plot(final_df.step, final_df.retain_match_rate, marker="s", label="retain match")
axes[1].set_xlabel("Final training step")
axes[1].set_title(f"Selected: {selected_name}")
axes[1].legend()
axes[1].grid(alpha=0.25)
figure.tight_layout()
figure.savefig(OUT / "repair_matrix_v3.png", dpi=180)
plt.close(figure)

report = {
    "status": "complete",
    "device": str(DEVICE),
    "selected_candidate": selected_candidate,
    "candidate_ranking": aggregate.to_dict(orient="records"),
    "final_checkpoints": final_df.to_dict(orient="records"),
    "trainable_tensors": trainable_names,
    "rule_guard": CONFIG["rule_guard"],
    "selection_guard": (
        "Cross-validation groups all augmentations by source image. Test images and test "
        "predictions are never read. Passing requires both poison suppression and retention."
    ),
}
(OUT / "repair_report_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(Path("/kaggle/working") / "REPAIR_MATRIX_V3_COMPLETE.txt").write_text(
    f"Repair matrix v3 complete. Selected {selected_name}.\n",
    encoding="utf-8",
)
log("ALL DONE", selected=selected_name, report=str(OUT / "repair_report_v3.json"))

# %% [markdown]
# ## V4 registry and frozen promotion gate
#
# The legacy V3 block above is intentionally rerun as E02. From this point on,
# every configuration is entered in one registry and compared with the same
# predeclared suppression/retention gate.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.ops import nms as torch_nms

REGISTRY = []
PUBLIC_PREDICTORS = {}
MODEL_PATHS = {}
FAILURES = []


def passes_gate(metrics):
    values = metrics
    return bool(
        values["poison_fire_rate_020"] <= 0.35
        and values["poison_score_ratio_median"] <= 0.25
        and values["retain_match_rate"] >= 0.90
        and 0.80 <= values["retain_score_ratio_median"] <= 1.20
    )


def register(experiment, candidate, family, metrics, **extra):
    row = {
        "experiment": experiment,
        "candidate": candidate,
        "family": family,
        **metrics,
        "passes_gate": passes_gate(metrics),
        **extra,
    }
    REGISTRY.append(row)
    pd.DataFrame(REGISTRY).to_csv(OUT / "experiment_registry.partial.csv", index=False)
    log("V4 candidate registered", **row)
    return row


def capture_failure(experiment, candidate, exc):
    row = {
        "experiment": experiment,
        "candidate": candidate,
        "error": f"{type(exc).__name__}: {exc}",
    }
    FAILURES.append(row)
    pd.DataFrame(FAILURES).to_csv(OUT / "experiment_failures.csv", index=False)
    log("V4 branch failed; continuing", **row)


def load_checkpoint_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model = build_loaded_model()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
    model.eval()
    log("Checkpoint loaded", path=str(path), missing=len(missing))
    return model


def predictions_for_model(model, image_ids=None, image_map=None):
    image_ids = all_ids if image_ids is None else list(image_ids)
    image_map = images if image_map is None else image_map
    return {image_id: infer(model, image_map[image_id]) for image_id in image_ids}


def metrics_from_predictions(predictions, validation_ids=None):
    validation_ids = all_ids if validation_ids is None else list(validation_ids)
    poison_ratios = []
    poison_scores = []
    retain_ratios = []
    retained = 0
    retain_total = 0
    count_ratios = []
    for image_id in validation_ids:
        target = boxes[image_id]
        reference_boxes, reference_scores = teacher_predictions[image_id]
        candidate_boxes, candidate_scores = predictions[image_id]
        reference_overlap = box_iou_numpy(target, reference_boxes)
        candidate_overlap = box_iou_numpy(target, candidate_boxes)
        reference_target = float(reference_scores[reference_overlap >= 0.2].max(initial=0))
        candidate_target = float(candidate_scores[candidate_overlap >= 0.2].max(initial=0))
        poison_scores.append(candidate_target)
        poison_ratios.append(candidate_target / max(reference_target, 1e-6))

        keep = (reference_scores >= 0.20) & (reference_overlap < 0.10)
        retained_boxes = reference_boxes[keep]
        retained_scores = reference_scores[keep]
        retain_total += len(retained_boxes)
        for reference_box, reference_score in zip(retained_boxes, retained_scores):
            overlap = box_iou_numpy(reference_box, candidate_boxes)
            if len(overlap) and overlap.max() >= 0.5:
                matched_index = int(overlap.argmax())
                retained += 1
                retain_ratios.append(
                    float(candidate_scores[matched_index] / max(reference_score, 1e-6))
                )
            else:
                retain_ratios.append(0.0)
        reference_count = max(int((reference_scores >= 0.20).sum()), 1)
        candidate_count = int((candidate_scores >= 0.20).sum())
        count_ratios.append(candidate_count / reference_count)

    poison_ratio = float(np.median(poison_ratios))
    fire_rate = float(np.mean(np.asarray(poison_scores) >= 0.20))
    match_rate = retained / max(retain_total, 1)
    positive = np.asarray(retain_ratios)
    positive = positive[positive > 0]
    retain_ratio = float(np.median(positive)) if len(positive) else 0.0
    proxy = (
        poison_ratio
        + 0.75 * (1 - match_rate)
        + 0.25 * abs(math.log(max(retain_ratio, 1e-3)))
    )
    return {
        "poison_score_ratio_median": poison_ratio,
        "poison_fire_rate_020": fire_rate,
        "retain_total": retain_total,
        "retain_matched": retained,
        "retain_match_rate": match_rate,
        "retain_score_ratio_median": retain_ratio,
        "count_ratio_median": float(np.median(count_ratios)),
        "proxy": proxy,
    }


def matched_scores(reference_boxes, source_boxes, source_scores, min_iou=0.5):
    output = np.zeros(len(reference_boxes), np.float32)
    for index, reference_box in enumerate(reference_boxes):
        overlaps = box_iou_numpy(reference_box, source_boxes)
        if len(overlaps) and overlaps.max() >= min_iou:
            output[index] = float(source_scores[int(overlaps.argmax())])
    return output


def blend_prediction_sets(weighted_sets, threshold=0.005):
    output = {}
    for image_id in all_ids:
        reference_boxes, reference_scores = teacher_predictions[image_id]
        score = np.zeros(len(reference_boxes), np.float32)
        for weight, prediction_set in weighted_sets:
            source_boxes, source_scores = prediction_set[image_id]
            score += float(weight) * matched_scores(
                reference_boxes, source_boxes, source_scores
            )
        keep = score >= threshold
        output[image_id] = (reference_boxes[keep].copy(), score[keep].copy())
    return output


def gated_prediction_set(
    indicator_set,
    ratio_threshold,
    suspicious_scale=0.0,
    minimum_score=0.0,
):
    output = {}
    for image_id in all_ids:
        reference_boxes, reference_scores = teacher_predictions[image_id]
        indicator_boxes, indicator_scores = indicator_set[image_id]
        matched = matched_scores(reference_boxes, indicator_boxes, indicator_scores)
        ratio = matched / np.clip(reference_scores, 1e-6, None)
        scores = reference_scores.copy()
        suspicious = ratio <= ratio_threshold
        scores[suspicious] = np.maximum(
            scores[suspicious] * suspicious_scale,
            float(minimum_score),
        )
        output[image_id] = (reference_boxes.copy(), scores)
    return output


# E00 - untouched baseline.
PUBLIC_PREDICTORS["original"] = teacher_predictions
register(
    "E00",
    "original_poisoned_model",
    "baseline",
    metrics_from_predictions(teacher_predictions),
    note="Untouched supplied model; expected poison ratio and retention ratio are one.",
)

# E02 - re-evaluate frozen V1/V2/V3 checkpoints and the just-reproduced V3.
checkpoint_patterns = {
    "v1_step200": "**/repair_matrix/final_full_cls_lr3e5_step200.pth",
    "v2_step100": "**/repair_matrix_v2/final_tail_w_pos16_step100.pth",
    "v3_step100": "**/repair_matrix_v3/final_full_pos0p5_step100.pth",
}
for checkpoint_name, pattern in checkpoint_patterns.items():
    try:
        matches = [
            path
            for path in Path("/kaggle/input").glob(pattern)
            if "experiment-matrix-v4" not in str(path)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {checkpoint_name} checkpoint, found {matches}")
        imported_model = load_checkpoint_model(matches[0])
        imported_predictions = predictions_for_model(imported_model)
        PUBLIC_PREDICTORS[checkpoint_name] = imported_predictions
        MODEL_PATHS[checkpoint_name] = str(matches[0])
        register(
            "E02",
            checkpoint_name,
            "reproduction",
            metrics_from_predictions(imported_predictions),
            checkpoint=str(matches[0]),
        )
        del imported_model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        capture_failure("E02", checkpoint_name, exc)

# The current notebook reran the frozen V3 matrix; register its final checkpoint.
try:
    reproduced_path = Path(final_df.sort_values("step").iloc[-1].path)
    reproduced_model = load_checkpoint_model(reproduced_path)
    reproduced_predictions = predictions_for_model(reproduced_model)
    PUBLIC_PREDICTORS["v3_reproduced"] = reproduced_predictions
    MODEL_PATHS["v3_reproduced"] = str(reproduced_path)
    register(
        "E02",
        "v3_reproduced",
        "reproduction",
        metrics_from_predictions(reproduced_predictions),
        checkpoint=str(reproduced_path),
    )
    del reproduced_model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as exc:
    capture_failure("E02", "v3_reproduced", exc)

# E03 - confidence blending.
if "v1_step200" in PUBLIC_PREDICTORS:
    for alpha in [0.05, 0.10, 0.20, 0.30, 0.40]:
        candidate_name = f"orig_v1_alpha{alpha:g}"
        predictions = blend_prediction_sets(
            [
                (1 - alpha, teacher_predictions),
                (alpha, PUBLIC_PREDICTORS["v1_step200"]),
            ]
        )
        PUBLIC_PREDICTORS[candidate_name] = predictions
        register(
            "E03",
            candidate_name,
            "confidence_blend",
            metrics_from_predictions(predictions),
            alpha=alpha,
        )

# E04 - use V1 confidence change only as a suspicious-detection indicator.
if "v1_step200" in PUBLIC_PREDICTORS:
    for ratio_threshold in [0.20, 0.35, 0.50, 0.65, 0.80]:
        candidate_name = f"v1_gate_ratio{ratio_threshold:g}"
        predictions = gated_prediction_set(
            PUBLIC_PREDICTORS["v1_step200"],
            ratio_threshold,
            suspicious_scale=0.25,
            minimum_score=0.0,
        )
        PUBLIC_PREDICTORS[candidate_name] = predictions
        register(
            "E04",
            candidate_name,
            "confidence_gate",
            metrics_from_predictions(predictions),
            ratio_threshold=ratio_threshold,
        )

# E05 - soft suppression with a nonzero confidence floor.
if "v1_step200" in PUBLIC_PREDICTORS:
    for minimum_score in [0.03, 0.05, 0.10, 0.15]:
        candidate_name = f"soft_gate_floor{minimum_score:g}"
        predictions = gated_prediction_set(
            PUBLIC_PREDICTORS["v1_step200"],
            ratio_threshold=0.50,
            suspicious_scale=0.25,
            minimum_score=minimum_score,
        )
        PUBLIC_PREDICTORS[candidate_name] = predictions
        register(
            "E05",
            candidate_name,
            "soft_suppression",
            metrics_from_predictions(predictions),
            minimum_score=minimum_score,
        )

# E06 - box-preserving confidence ensembles.
ensemble_specs = [
    ("orig_v1_80_20", [(0.8, "original"), (0.2, "v1_step200")]),
    ("orig_v3_80_20", [(0.8, "original"), (0.2, "v3_step100")]),
    (
        "orig_v1_v3_60_20_20",
        [(0.6, "original"), (0.2, "v1_step200"), (0.2, "v3_step100")],
    ),
]
for candidate_name, spec in ensemble_specs:
    try:
        predictions = blend_prediction_sets(
            [(weight, PUBLIC_PREDICTORS[name]) for weight, name in spec]
        )
        PUBLIC_PREDICTORS[candidate_name] = predictions
        register(
            "E06",
            candidate_name,
            "box_preserving_ensemble",
            metrics_from_predictions(predictions),
            weights=json.dumps(spec),
        )
    except Exception as exc:
        capture_failure("E06", candidate_name, exc)

# %% [markdown]
# ## E07-E13: detection-level poison signatures and synthetic controls

# %%
def roi_mean(feature, box):
    _, channels, height, width = feature.shape
    scale_x = width / 1024.0
    scale_y = height / 1024.0
    x1 = int(np.clip(math.floor(box[0] * scale_x), 0, width - 1))
    y1 = int(np.clip(math.floor(box[1] * scale_y), 0, height - 1))
    x2 = int(np.clip(math.ceil(box[2] * scale_x), x1 + 1, width))
    y2 = int(np.clip(math.ceil(box[3] * scale_y), y1 + 1, height))
    return feature[0, :, y1:y2, x1:x2].float().mean(dim=(1, 2)).detach().cpu().numpy()


def roi_features(gray, query_box):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        image_list = teacher.preprocess_image([to_record(gray)])
        feature_dict = teacher.backbone(image_list.tensor)
        p3_cls = teacher.head.cls_subnet(feature_dict["p3"])
        p4_cls = teacher.head.cls_subnet(feature_dict["p4"])
    p3 = roi_mean(p3_cls, query_box)
    p4 = roi_mean(p4_cls, query_box)
    return p3.astype(np.float32), p4.astype(np.float32)


def morphology_features(gray, query_box):
    x1, y1, x2, y2 = map(int, np.round(query_box))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(1024, x2), min(1024, y2)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(12, np.float32)
    height, width = crop.shape
    threshold = float(np.percentile(crop, 90))
    mask = crop >= threshold
    yy, xx = np.nonzero(mask)
    if len(xx) >= 3:
        coordinates = np.column_stack([xx, yy]).astype(np.float32)
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        major = float(max(eigenvalues[order[0]], 1e-6))
        minor = float(max(eigenvalues[order[-1]], 1e-6))
        vector = eigenvectors[:, order[0]]
        angle = float(math.atan2(vector[1], vector[0]))
        elongation = math.sqrt(major / minor)
    else:
        angle = 0.0
        elongation = 1.0
    gradient_x = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    return np.asarray(
        [
            width / 1024.0,
            height / 1024.0,
            width / max(height, 1),
            math.log1p(width * height),
            float(crop.mean()) / 255.0,
            float(crop.std()) / 255.0,
            float(np.percentile(crop, 95)) / 255.0,
            float(np.percentile(crop, 99)) / 255.0,
            float(gradient.mean()) / 255.0,
            math.sin(angle),
            math.cos(angle),
            float(np.clip(elongation, 0, 20)) / 20.0,
        ],
        np.float32,
    )


def score_at_box(model, gray, query_box):
    prediction_boxes, prediction_scores = infer(model, gray)
    overlaps = box_iou_numpy(query_box, prediction_boxes)
    return float(prediction_scores[overlaps >= 0.2].max(initial=0))


def stability_features(gray, query_box):
    base = score_at_box(teacher, gray, query_box)
    values = []
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
    values.append(score_at_box(teacher, blurred, query_box) / max(base, 1e-6))

    flipped = gray[:, ::-1].copy()
    flipped_box = np.asarray(
        [1024 - query_box[2], query_box[1], 1024 - query_box[0], query_box[3]],
        np.float32,
    )
    values.append(score_at_box(teacher, flipped, flipped_box) / max(base, 1e-6))

    matrix = np.asarray([[1, 0, 32], [0, 1, 0]], np.float32)
    shifted = cv2.warpAffine(
        gray, matrix, (1024, 1024), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    shifted_box = transform_box_affine(query_box, matrix)
    values.append(score_at_box(teacher, shifted, shifted_box) / max(base, 1e-6))

    matrix = np.asarray([[0.9, 0, 51.2], [0, 0.9, 51.2]], np.float32)
    scaled = cv2.warpAffine(
        gray, matrix, (1024, 1024), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    scaled_box = transform_box_affine(query_box, matrix)
    values.append(score_at_box(teacher, scaled, scaled_box) / max(base, 1e-6))
    return np.clip(np.asarray(values, np.float32), 0, 5)


def deterministic_controls(image_id, count=3):
    target = boxes[image_id]
    width = target[2] - target[0]
    height = target[3] - target[1]
    candidates = []
    rng = np.random.default_rng(SEED + image_id)
    for _ in range(200):
        x1 = float(rng.uniform(8, max(9, 1016 - width)))
        y1 = float(rng.uniform(8, max(9, 1016 - height)))
        candidate = np.asarray([x1, y1, x1 + width, y1 + height], np.float32)
        if box_iou_numpy(target, [candidate])[0] < 0.01:
            candidates.append(candidate)
        if len(candidates) == count:
            break
    if len(candidates) != count:
        raise RuntimeError(f"Could not create controls for {image_id}")
    return candidates


def inpaint_poison(gray, query_box):
    x1, y1, x2, y2 = map(int, np.round(query_box))
    mask = np.zeros(gray.shape, np.uint8)
    margin = 5
    cv2.rectangle(
        mask,
        (max(0, x1 - margin), max(0, y1 - margin)),
        (min(1023, x2 + margin), min(1023, y2 + margin)),
        255,
        thickness=-1,
    )
    normalized = np.clip(gray / 255.0 * 65535.0, 0, 65535).astype(np.uint16)
    restored = cv2.inpaint(normalized, mask, 7, cv2.INPAINT_TELEA)
    return restored.astype(np.float32) / 65535.0 * 255.0


def synthetic_streak(image_id, seed):
    rng = np.random.default_rng(seed)
    background = inpaint_poison(images[image_id], boxes[image_id])
    canvas = background.copy()
    length = int(rng.integers(24, 150))
    width = int(rng.integers(1, 5))
    angle = float(rng.uniform(0, math.pi))
    center_x = int(rng.integers(length // 2 + 16, 1024 - length // 2 - 16))
    center_y = int(rng.integers(length // 2 + 16, 1024 - length // 2 - 16))
    delta_x = math.cos(angle) * length / 2
    delta_y = math.sin(angle) * length / 2
    point_a = (int(center_x - delta_x), int(center_y - delta_y))
    point_b = (int(center_x + delta_x), int(center_y + delta_y))
    intensity = float(np.percentile(background, 99.5) + rng.uniform(25, 90))
    layer = np.zeros_like(canvas, np.float32)
    cv2.line(layer, point_a, point_b, intensity, thickness=width, lineType=cv2.LINE_AA)
    psf = float(rng.uniform(0.5, 2.0))
    layer = cv2.GaussianBlur(layer, (0, 0), psf)
    canvas = np.clip(canvas + layer, 0, 255)
    x1 = max(0, min(point_a[0], point_b[0]) - 4 * width)
    y1 = max(0, min(point_a[1], point_b[1]) - 4 * width)
    x2 = min(1024, max(point_a[0], point_b[0]) + 4 * width + 1)
    y2 = min(1024, max(point_a[1], point_b[1]) + 4 * width + 1)
    return canvas.astype(np.float32), np.asarray([x1, y1, x2, y2], np.float32)


feature_rows = []
synthetic_bank = {}
with heartbeat("E07-E12 feature extraction"):
    for image_id in tqdm(all_ids, desc="ROI feature bank"):
        p3, p4 = roi_features(images[image_id], boxes[image_id])
        feature_rows.append(
            {
                "image_id": image_id,
                "sample": "poison",
                "label": 1,
                "p3": p3,
                "p4": p4,
                "morph": morphology_features(images[image_id], boxes[image_id]),
                "stability": stability_features(images[image_id], boxes[image_id]),
                "image": images[image_id],
                "box": boxes[image_id],
            }
        )
        for control_index, control_box in enumerate(deterministic_controls(image_id)):
            p3, p4 = roi_features(images[image_id], control_box)
            feature_rows.append(
                {
                    "image_id": image_id,
                    "sample": f"control_{control_index}",
                    "label": 0,
                    "p3": p3,
                    "p4": p4,
                    "morph": morphology_features(images[image_id], control_box),
                    "stability": stability_features(images[image_id], control_box),
                    "image": images[image_id],
                    "box": control_box,
                }
            )
        for synthetic_index in range(2):
            synthetic_image, synthetic_box = synthetic_streak(
                image_id, SEED + image_id * 10 + synthetic_index
            )
            synthetic_bank[(image_id, synthetic_index)] = (
                synthetic_image,
                synthetic_box,
            )
            p3, p4 = roi_features(synthetic_image, synthetic_box)
            feature_rows.append(
                {
                    "image_id": image_id,
                    "sample": f"synthetic_{synthetic_index}",
                    "label": 0,
                    "p3": p3,
                    "p4": p4,
                    "morph": morphology_features(synthetic_image, synthetic_box),
                    "stability": stability_features(synthetic_image, synthetic_box),
                    "image": synthetic_image,
                    "box": synthetic_box,
                }
            )

feature_cosines = []
for row in feature_rows:
    if row["sample"] != "poison":
        continue
    blurred = cv2.GaussianBlur(row["image"], (0, 0), 1.0)
    p3_blur, p4_blur = roi_features(blurred, row["box"])
    original = np.concatenate([row["p3"], row["p4"]])
    transformed = np.concatenate([p3_blur, p4_blur])
    cosine = float(
        np.dot(original, transformed)
        / max(np.linalg.norm(original) * np.linalg.norm(transformed), 1e-6)
    )
    feature_cosines.append(cosine)
register(
    "E07",
    "p3_p4_roi_features",
    "feature_audit",
    metrics_from_predictions(teacher_predictions),
    transform_cosine_median=float(np.median(feature_cosines)),
    note="Feature audit only; not a promoted detector candidate.",
)


def feature_matrix(rows, mode):
    vectors = []
    for row in rows:
        if mode == "p3":
            vector = row["p3"]
        elif mode == "p4":
            vector = row["p4"]
        elif mode == "p3p4":
            vector = np.concatenate([row["p3"], row["p4"]])
        elif mode == "p3p4_morph":
            vector = np.concatenate([row["p3"], row["p4"], row["morph"]])
        elif mode == "all":
            vector = np.concatenate(
                [row["p3"], row["p4"], row["morph"], row["stability"]]
            )
        else:
            raise ValueError(mode)
        vectors.append(vector.astype(np.float32))
    return np.asarray(vectors)


def grouped_gate_cv(rows, mode, model_type, parameters):
    x = feature_matrix(rows, mode)
    y = np.asarray([row["label"] for row in rows], np.int64)
    groups = np.asarray([row["image_id"] for row in rows], np.int64)
    records = []
    for parameter in parameters:
        probabilities = np.zeros(len(rows), np.float32)
        for fold, (train_index, valid_index) in enumerate(
            GroupKFold(n_splits=5).split(x, y, groups)
        ):
            if model_type == "logistic":
                estimator = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=float(parameter),
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=SEED,
                    ),
                )
            else:
                hidden, dropout = parameter
                # sklearn MLP has no dropout; alpha is the bounded regularization
                # counterpart used for this small-data sweep.
                estimator = make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=(int(hidden), int(hidden // 2)),
                        alpha=0.01 if dropout == 0.2 else 0.05,
                        max_iter=800,
                        early_stopping=True,
                        random_state=SEED + fold,
                    ),
                )
            estimator.fit(x[train_index], y[train_index])
            probabilities[valid_index] = estimator.predict_proba(x[valid_index])[:, 1]
        auc = float(roc_auc_score(y, probabilities))
        records.append(
            {
                "mode": mode,
                "model_type": model_type,
                "parameter": str(parameter),
                "auc": auc,
                "probabilities": probabilities,
            }
        )
    return sorted(records, key=lambda item: item["auc"], reverse=True)


def fit_gate(rows, mode, model_type, parameter):
    x = feature_matrix(rows, mode)
    y = np.asarray([row["label"] for row in rows], np.int64)
    if model_type == "logistic":
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(parameter),
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            ),
        )
    else:
        hidden, dropout = parameter
        estimator = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(int(hidden), int(hidden // 2)),
                alpha=0.01 if dropout == 0.2 else 0.05,
                max_iter=1000,
                early_stopping=True,
                random_state=SEED,
            ),
        )
    estimator.fit(x, y)
    return estimator


def vector_for_detection(gray, query_box, mode):
    p3, p4 = roi_features(gray, query_box)
    row = {
        "p3": p3,
        "p4": p4,
        "morph": morphology_features(gray, query_box),
        "stability": (
            stability_features(gray, query_box)
            if mode == "all"
            else np.zeros(4, np.float32)
        ),
    }
    return feature_matrix([row], mode)


def gate_public_predictions(estimator, mode, scale=0.1):
    output = {}
    for image_id in all_ids:
        reference_boxes, reference_scores = teacher_predictions[image_id]
        gated_scores = []
        for query_box, score in zip(reference_boxes, reference_scores):
            probability = float(
                estimator.predict_proba(
                    vector_for_detection(images[image_id], query_box, mode)
                )[0, 1]
            )
            gated_scores.append(float(score) * (1 - probability * (1 - scale)))
        output[image_id] = (
            reference_boxes.copy(),
            np.asarray(gated_scores, np.float32),
        )
    return output


base_rows = [row for row in feature_rows if not row["sample"].startswith("synthetic")]
gate_specs = [
    ("E08", "p3", "logistic", [0.01, 0.1, 1, 10]),
    ("E08", "p4", "logistic", [0.01, 0.1, 1, 10]),
    ("E08", "p3p4", "logistic", [0.01, 0.1, 1, 10]),
    ("E09", "p3p4", "mlp", [(16, 0.2), (16, 0.5), (32, 0.2), (32, 0.5)]),
    ("E10", "p3p4_morph", "logistic", [0.01, 0.1, 1, 10]),
    ("E11", "all", "logistic", [0.01, 0.1, 1, 10]),
]
gate_cv_rows = []
GATES = {}
for experiment, mode, model_type, parameters in gate_specs:
    try:
        ranking = grouped_gate_cv(base_rows, mode, model_type, parameters)
        best = ranking[0]
        estimator = fit_gate(base_rows, mode, model_type, parameters[0] if False else eval(best["parameter"]))
        candidate_name = f"{experiment.lower()}_{mode}_{model_type}"
        predictions = gate_public_predictions(estimator, mode)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        GATES[candidate_name] = (estimator, mode)
        register(
            experiment,
            candidate_name,
            "detection_gate",
            metrics_from_predictions(predictions),
            grouped_auc=best["auc"],
            gate_parameter=best["parameter"],
        )
        for item in ranking:
            gate_cv_rows.append(
                {
                    "experiment": experiment,
                    "mode": mode,
                    "model_type": model_type,
                    "parameter": item["parameter"],
                    "auc": item["auc"],
                }
            )
    except Exception as exc:
        capture_failure(experiment, f"{mode}_{model_type}", exc)

# E12 records synthetic-control detectability under the original teacher.
synthetic_scores = [
    score_at_box(teacher, image, query_box)
    for image, query_box in synthetic_bank.values()
]
register(
    "E12",
    "synthetic_positive_streak_controls",
    "synthetic_audit",
    metrics_from_predictions(teacher_predictions),
    synthetic_recall_020=float(np.mean(np.asarray(synthetic_scores) >= 0.20)),
    synthetic_score_median=float(np.median(synthetic_scores)),
    synthetic_count=len(synthetic_scores),
)

# E13 retrains the strongest gate with poison versus synthetic-positive balance.
synthetic_rows = [row for row in feature_rows if row["sample"].startswith("synthetic")]
poison_rows = [row for row in feature_rows if row["sample"] == "poison"]
control_rows = [row for row in feature_rows if row["sample"].startswith("control")]
for ratio in [1, 2, 4]:
    try:
        balanced_rows = poison_rows + control_rows + synthetic_rows * ratio
        ranking = grouped_gate_cv(balanced_rows, "all", "logistic", [0.01, 0.1, 1, 10])
        best = ranking[0]
        parameter = eval(best["parameter"])
        estimator = fit_gate(balanced_rows, "all", "logistic", parameter)
        candidate_name = f"e13_synthetic_balance1to{ratio}"
        predictions = gate_public_predictions(estimator, "all")
        PUBLIC_PREDICTORS[candidate_name] = predictions
        GATES[candidate_name] = (estimator, "all")
        register(
            "E13",
            candidate_name,
            "synthetic_balanced_gate",
            metrics_from_predictions(predictions),
            grouped_auc=best["auc"],
            balance=f"1:{ratio}",
        )
    except Exception as exc:
        capture_failure("E13", f"balance1to{ratio}", exc)

pd.DataFrame(gate_cv_rows).to_csv(OUT / "grouped_gate_cv.csv", index=False)

# %% [markdown]
# ## E01 and E14-E16: empty-label, distillation, adapter and low-rank repair

# %%
V4_TRAINING_HISTORY = []
MODEL_SPECS = {}


def save_model_checkpoint(model, name, spec):
    path = OUT / f"{name}.pth"
    torch.save(
        {
            "model": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "spec": spec,
        },
        path,
    )
    MODEL_PATHS[name] = str(path)
    MODEL_SPECS[name] = spec
    return path


def empty_label_loss(model, gray):
    logits, _ = dense_logits(model, to_record(gray))
    terms = []
    for level in logits:
        values = level[0, :, 0].float()
        probability = torch.sigmoid(values)
        terms.append((probability.pow(2) * F.softplus(values)).mean())
    return torch.stack(terms).mean()


def train_empty_baseline():
    model = build_loaded_model()
    configure_scope(model, "full_cls")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    steps_per_epoch = math.ceil(len(all_ids) / 4)
    total_steps = 20 * steps_per_epoch
    model.train()
    with heartbeat("E01 empty-label baseline"):
        for step in range(1, total_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            batch_ids = [
                all_ids[((step - 1) * 4 + offset) % len(all_ids)]
                for offset in range(4)
            ]
            for offset, image_id in enumerate(batch_ids):
                gray, _ = augment(
                    image_id,
                    SEED + 100000 + step * 4 + offset,
                    allow_transplant=False,
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = empty_label_loss(model, gray) / 4
                scaler.scale(loss).backward()
                batch_loss += float(loss.detach().cpu())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            row = {
                "experiment": "E01",
                "candidate": "official_empty_lr1e4_epoch20_batch4",
                "step": step,
                "loss": batch_loss,
            }
            V4_TRAINING_HISTORY.append(row)
            if step == 1 or step % 10 == 0 or step == total_steps:
                log("E01 training step", **row)
    return model


try:
    empty_model = train_empty_baseline()
    empty_predictions = predictions_for_model(empty_model)
    candidate_name = "official_empty_lr1e4_epoch20_batch4"
    PUBLIC_PREDICTORS[candidate_name] = empty_predictions
    save_model_checkpoint(
        empty_model,
        candidate_name,
        {
            "experiment": "E01",
            "architecture": "original",
            "scope": "full_cls",
            "lr": 1e-4,
            "epochs": 20,
            "batch": 4,
        },
    )
    register(
        "E01",
        candidate_name,
        "empty_label_baseline",
        metrics_from_predictions(empty_predictions),
    )
    del empty_model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as exc:
    capture_failure("E01", "official_empty_lr1e4_epoch20_batch4", exc)


def aggregate_cv_rows(rows):
    frame = pd.DataFrame(rows)
    result = {
        "poison_score_ratio_median": float(
            frame.poison_score_ratio_median.median()
        ),
        "poison_fire_rate_020": float(frame.poison_fire_rate_020.mean()),
        "retain_total": int(frame.retain_total.sum()),
        "retain_matched": int(frame.retain_matched.sum()),
        "retain_match_rate": float(
            frame.retain_matched.sum() / max(frame.retain_total.sum(), 1)
        ),
        "retain_score_ratio_median": float(
            frame.retain_score_ratio_median.median()
        ),
        "count_ratio_median": float(frame.count_ratio_median.median())
        if "count_ratio_median" in frame
        else 1.0,
        "proxy": float(frame.proxy.mean()),
    }
    return result


def train_custom_model(builder, candidate, train_ids, steps, run_name):
    model = builder()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError(f"No trainable tensors for {run_name}")
    original_parameters = {
        name: parameter.detach().clone() for name, parameter in trainable
    }
    parameters = [parameter for _, parameter in trainable]
    optimizer = torch.optim.AdamW(
        parameters, lr=candidate["lr"], weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    with heartbeat(run_name):
        for step in range(1, steps + 1):
            image_id = train_ids[(step - 1) % len(train_ids)]
            image, target_boxes = augment(
                image_id,
                SEED + zlib.crc32(run_name.encode()) % 100000 + step,
                allow_transplant=True,
            )
            optimizer.zero_grad(set_to_none=True)
            objective, suppression, positive, negative, drift = targeted_loss(
                model,
                original_parameters,
                image,
                target_boxes,
                candidate,
            )
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            row = {
                "experiment": candidate["experiment"],
                "candidate": candidate["name"],
                "run": run_name,
                "step": step,
                "loss": float(objective.detach().cpu()),
                "suppression": float(suppression.detach().cpu()),
                "positive_retention": float(positive.detach().cpu()),
                "negative_retention": float(negative.detach().cpu()),
                "drift": float(drift.detach().cpu()),
            }
            V4_TRAINING_HISTORY.append(row)
            if step == 1 or step % 10 == 0 or step == steps:
                log("custom training step", **row)
    return model


class ResidualAdapterScore(torch.nn.Module):
    def __init__(self, base, bottleneck):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        channels = base.in_channels
        self.down = torch.nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.up = torch.nn.Conv2d(bottleneck, channels, 1, bias=False)
        torch.nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, value):
        adapted = value + self.up(F.gelu(self.down(value)))
        return self.base(adapted)


class LoRAClassificationScore(torch.nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.down = torch.nn.Conv2d(base.in_channels, rank, 1, bias=False)
        self.up = torch.nn.Conv2d(rank, base.out_channels, 1, bias=False)
        torch.nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return self.base(value) + self.up(self.down(value))


def adapter_builder(bottleneck):
    def build():
        model = build_loaded_model()
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.head.cls_score = ResidualAdapterScore(
            model.head.cls_score, bottleneck
        ).to(DEVICE)
        return model

    return build


def lora_builder(rank):
    def build():
        model = build_loaded_model()
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.head.cls_score = LoRAClassificationScore(
            model.head.cls_score, rank
        ).to(DEVICE)
        return model

    return build


def scoped_builder(scope):
    def build():
        model = build_loaded_model()
        configure_scope(model, scope)
        return model

    return build


def run_grouped_training_family(experiment, candidates, builder_for, steps=20):
    summary = []
    for candidate in candidates:
        fold_rows = []
        for fold_index, validation_ids in FOLDS.items():
            model = None
            try:
                train_ids = [
                    image_id
                    for image_id in all_ids
                    if image_id not in validation_ids
                ]
                model = train_custom_model(
                    builder_for(candidate),
                    candidate,
                    train_ids,
                    steps,
                    f"{candidate['name']}_fold{fold_index}",
                )
                fold_rows.append(
                    {
                        "fold": fold_index,
                        **validation_metrics(model, validation_ids),
                    }
                )
            finally:
                if model is not None:
                    del model
                gc.collect()
                torch.cuda.empty_cache()
        metrics = aggregate_cv_rows(fold_rows)
        register(
            experiment,
            candidate["name"],
            candidate["family"],
            metrics,
            cv_folds=5,
        )
        summary.append((candidate, metrics))
    summary.sort(key=lambda item: (not passes_gate(item[1]), item[1]["proxy"]))
    return summary


# E14 - classification-head distillation sweep.
e14_candidates = [
    {
        "experiment": "E14",
        "name": f"e14_kd{weight}",
        "family": "classification_distillation",
        "scope": "full_cls",
        "lr": 3e-5,
        "positive": float(weight),
        "negative": 0.25,
        "anchor": 5e-4,
    }
    for weight in [1, 3, 10, 30]
]
try:
    e14_summary = run_grouped_training_family(
        "E14",
        e14_candidates,
        lambda candidate: scoped_builder(candidate["scope"]),
    )
    best_candidate = e14_summary[0][0]
    e14_model = train_custom_model(
        scoped_builder(best_candidate["scope"]),
        best_candidate,
        all_ids,
        60,
        f"{best_candidate['name']}_final",
    )
    e14_predictions = predictions_for_model(e14_model)
    PUBLIC_PREDICTORS[f"{best_candidate['name']}_final"] = e14_predictions
    save_model_checkpoint(
        e14_model,
        f"{best_candidate['name']}_final",
        {
            "experiment": "E14",
            "architecture": "original",
            "scope": best_candidate["scope"],
            "candidate": best_candidate,
        },
    )
    register(
        "E14",
        f"{best_candidate['name']}_final",
        "classification_distillation_final",
        metrics_from_predictions(e14_predictions),
    )
    del e14_model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as exc:
    capture_failure("E14", "kd_sweep", exc)

# E15 - residual classification adapter.
e15_candidates = [
    {
        "experiment": "E15",
        "name": f"e15_adapter_b{bottleneck}",
        "family": "classification_adapter",
        "bottleneck": bottleneck,
        "lr": 1e-4,
        "positive": 3.0,
        "negative": 0.25,
        "anchor": 1e-4,
    }
    for bottleneck in [4, 8, 16]
]
try:
    e15_summary = run_grouped_training_family(
        "E15",
        e15_candidates,
        lambda candidate: adapter_builder(candidate["bottleneck"]),
    )
    best_candidate = e15_summary[0][0]
    e15_model = train_custom_model(
        adapter_builder(best_candidate["bottleneck"]),
        best_candidate,
        all_ids,
        60,
        f"{best_candidate['name']}_final",
    )
    e15_predictions = predictions_for_model(e15_model)
    PUBLIC_PREDICTORS[f"{best_candidate['name']}_final"] = e15_predictions
    save_model_checkpoint(
        e15_model,
        f"{best_candidate['name']}_final",
        {
            "experiment": "E15",
            "architecture": "adapter",
            "bottleneck": best_candidate["bottleneck"],
            "candidate": best_candidate,
        },
    )
    register(
        "E15",
        f"{best_candidate['name']}_final",
        "classification_adapter_final",
        metrics_from_predictions(e15_predictions),
    )
    del e15_model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as exc:
    capture_failure("E15", "adapter_sweep", exc)

# E16 - low-rank classification repair.
e16_candidates = [
    {
        "experiment": "E16",
        "name": f"e16_lora_r{rank}",
        "family": "lora_classification",
        "rank": rank,
        "lr": 1e-4,
        "positive": 3.0,
        "negative": 0.25,
        "anchor": 1e-4,
    }
    for rank in [1, 2, 4, 8]
]
try:
    e16_summary = run_grouped_training_family(
        "E16",
        e16_candidates,
        lambda candidate: lora_builder(candidate["rank"]),
    )
    best_candidate = e16_summary[0][0]
    e16_model = train_custom_model(
        lora_builder(best_candidate["rank"]),
        best_candidate,
        all_ids,
        60,
        f"{best_candidate['name']}_final",
    )
    e16_predictions = predictions_for_model(e16_model)
    PUBLIC_PREDICTORS[f"{best_candidate['name']}_final"] = e16_predictions
    save_model_checkpoint(
        e16_model,
        f"{best_candidate['name']}_final",
        {
            "experiment": "E16",
            "architecture": "lora",
            "rank": best_candidate["rank"],
            "candidate": best_candidate,
        },
    )
    register(
        "E16",
        f"{best_candidate['name']}_final",
        "lora_classification_final",
        metrics_from_predictions(e16_predictions),
    )
    del e16_model
    gc.collect()
    torch.cuda.empty_cache()
except Exception as exc:
    capture_failure("E16", "lora_sweep", exc)

pd.DataFrame(V4_TRAINING_HISTORY).to_csv(
    OUT / "training_history_v4.partial.csv", index=False
)

# %% [markdown]
# ## E17-E31: pruning, causal masks, recovery and ensembles

# %%
poison_vectors = np.asarray(
    [(row["p3"] + row["p4"]) / 2 for row in poison_rows], np.float32
)
control_vectors = np.asarray(
    [(row["p3"] + row["p4"]) / 2 for row in control_rows], np.float32
)
activation_score = np.mean(
    np.abs(np.concatenate([poison_vectors, control_vectors], axis=0)), axis=0
)
selectivity_score = (
    poison_vectors.mean(axis=0) - control_vectors.mean(axis=0)
) / np.clip(control_vectors.std(axis=0), 1e-4, None)
low_activation_channels = np.argsort(activation_score).tolist()
selective_channels = np.argsort(selectivity_score)[::-1].tolist()


def static_pruned_model(channels, scale=0.0):
    model = build_loaded_model()
    selected = torch.as_tensor(channels, dtype=torch.long, device=DEVICE)
    model.head.cls_score.weight.data[:, selected, :, :] *= float(scale)
    model.eval()
    return model


PRUNING_CHANNELS = {}
prune_counts = {
    "0p5": max(1, round(256 * 0.005)),
    "1": max(1, round(256 * 0.01)),
    "2": max(1, round(256 * 0.02)),
    "5": max(1, round(256 * 0.05)),
}

# E17 - classic low-activation Fine-Pruning baseline.
for percentage, count in prune_counts.items():
    model = None
    candidate_name = f"e17_low_activation_{percentage}pct"
    try:
        channels = low_activation_channels[:count]
        model = static_pruned_model(channels)
        predictions = predictions_for_model(model)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        PRUNING_CHANNELS[candidate_name] = channels
        register(
            "E17",
            candidate_name,
            "fine_pruning",
            metrics_from_predictions(predictions),
            channel_count=count,
            channels=",".join(map(str, channels)),
        )
    except Exception as exc:
        capture_failure("E17", candidate_name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()

# E18 - poison-selectivity pruning.
for percentage, count in prune_counts.items():
    model = None
    candidate_name = f"e18_selective_{percentage}pct"
    try:
        channels = selective_channels[:count]
        model = static_pruned_model(channels)
        predictions = predictions_for_model(model)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        PRUNING_CHANNELS[candidate_name] = channels
        register(
            "E18",
            candidate_name,
            "selectivity_pruning",
            metrics_from_predictions(predictions),
            channel_count=count,
            channels=",".join(map(str, channels)),
        )
    except Exception as exc:
        capture_failure("E18", candidate_name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()

# E19 - individual causal validation of the top twenty selective channels.
causal_rows = []
for channel in selective_channels[:20]:
    model = None
    candidate_name = f"e19_channel_{channel}"
    try:
        model = static_pruned_model([channel])
        predictions = predictions_for_model(model)
        metrics = metrics_from_predictions(predictions)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        PRUNING_CHANNELS[candidate_name] = [channel]
        register(
            "E19",
            candidate_name,
            "causal_pruning",
            metrics,
            channels=str(channel),
        )
        causal_rows.append((channel, metrics))
    except Exception as exc:
        capture_failure("E19", candidate_name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
causal_rows.sort(key=lambda item: item[1]["proxy"])
causal_top = [channel for channel, _ in causal_rows[:10]]

# E20 - pairwise interactions among the best causal channels.
pair_rows = []
for first_index in range(len(causal_top)):
    for second_index in range(first_index + 1, len(causal_top)):
        channels = [causal_top[first_index], causal_top[second_index]]
        model = None
        candidate_name = f"e20_pair_{channels[0]}_{channels[1]}"
        try:
            model = static_pruned_model(channels)
            predictions = predictions_for_model(model)
            metrics = metrics_from_predictions(predictions)
            PUBLIC_PREDICTORS[candidate_name] = predictions
            PRUNING_CHANNELS[candidate_name] = channels
            register(
                "E20",
                candidate_name,
                "pairwise_pruning",
                metrics,
                channels=",".join(map(str, channels)),
            )
            pair_rows.append((channels, metrics, candidate_name))
        except Exception as exc:
            capture_failure("E20", candidate_name, exc)
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
pair_rows.sort(key=lambda item: item[1]["proxy"])

# E21 - deployment-compatible complete classification-tower filters.
# Zeroing one cls-score input column removes the corresponding final tower
# filter from every anchor output and can be exported as an ordinary state dict.
for count in [1, 2, 4, 8]:
    channels = causal_top[:count]
    model = None
    candidate_name = f"e21_structured_{count}filters"
    try:
        model = static_pruned_model(channels)
        predictions = predictions_for_model(model)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        PRUNING_CHANNELS[candidate_name] = channels
        register(
            "E21",
            candidate_name,
            "structured_filter_pruning",
            metrics_from_predictions(predictions),
            channels=",".join(map(str, channels)),
        )
    except Exception as exc:
        capture_failure("E21", candidate_name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


class LevelMaskedScore(torch.nn.Module):
    def __init__(self, base, channels, active_levels):
        super().__init__()
        self.base = base
        self.register_buffer(
            "channel_mask", torch.ones(base.in_channels, dtype=torch.float32)
        )
        self.channel_mask[torch.as_tensor(channels, dtype=torch.long)] = 0
        self.active_levels = set(active_levels)
        self.calls = 0

    def forward(self, value):
        level = self.calls % 5
        self.calls += 1
        if level in self.active_levels:
            value = value * self.channel_mask[None, :, None, None]
        return self.base(value)


def level_pruned_model(channels, active_levels):
    model = build_loaded_model()
    model.head.cls_score = LevelMaskedScore(
        model.head.cls_score, channels, active_levels
    ).to(DEVICE)
    model.eval()
    return model


level_specs = [
    ("E22", "p3_only", {0}),
    ("E23", "p4_only", {1}),
    ("E24", "p3_p4", {0, 1}),
]
for experiment, label, levels in level_specs:
    for count in [1, 3, 5, 13]:
        model = None
        channels = selective_channels[:count]
        candidate_name = f"{experiment.lower()}_{label}_{count}ch"
        try:
            model = level_pruned_model(channels, levels)
            predictions = predictions_for_model(model)
            PUBLIC_PREDICTORS[candidate_name] = predictions
            PRUNING_CHANNELS[candidate_name] = channels
            register(
                experiment,
                candidate_name,
                "level_specific_pruning",
                metrics_from_predictions(predictions),
                channels=",".join(map(str, channels)),
                active_levels=",".join(map(str, sorted(levels))),
            )
        except Exception as exc:
            capture_failure(experiment, candidate_name, exc)
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()


def fixed_mask_builder(channels, active_levels={0, 1, 2, 3, 4}):
    def build():
        model = build_loaded_model()
        configure_scope(model, "full_cls")
        model.head.cls_score = LevelMaskedScore(
            model.head.cls_score, channels, active_levels
        ).to(DEVICE)
        return model

    return build


def best_pruning_candidate():
    frame = pd.DataFrame(REGISTRY)
    subset = frame[
        frame.experiment.isin(
            ["E17", "E18", "E19", "E20", "E21", "E22", "E23", "E24"]
        )
    ].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["passes_gate", "proxy"], ascending=[False, True])
    return str(subset.iloc[0].candidate)


best_prune_name = best_pruning_candidate()
best_prune_channels = PRUNING_CHANNELS.get(
    best_prune_name, selective_channels[:4]
)

# E25 - RNP-style reconstruction after exposure/pruning.
for recovery_lr in [1e-5, 3e-5, 1e-4]:
    candidate_name = f"e25_rnp_lr{recovery_lr:g}"
    candidate = {
        "experiment": "E25",
        "name": candidate_name,
        "family": "rnp_reconstructive_pruning",
        "lr": recovery_lr,
        "positive": 10.0,
        "negative": 0.5,
        "anchor": 1e-3,
    }
    model = None
    try:
        model = train_custom_model(
            fixed_mask_builder(best_prune_channels),
            candidate,
            all_ids,
            40,
            candidate_name,
        )
        predictions = predictions_for_model(model)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        PRUNING_CHANNELS[candidate_name] = best_prune_channels
        save_model_checkpoint(
            model,
            candidate_name,
            {
                "experiment": "E25",
                "architecture": "fixed_level_mask",
                "channels": best_prune_channels,
                "active_levels": [0, 1, 2, 3, 4],
                "candidate": candidate,
            },
        )
        register(
            "E25",
            candidate_name,
            "rnp_reconstructive_pruning",
            metrics_from_predictions(predictions),
            recovery_lr=recovery_lr,
        )
    except Exception as exc:
        capture_failure("E25", candidate_name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def adversarial_channel_saliency(sample_ids):
    model = build_loaded_model()
    configure_scope(model, "score_weight")
    model.train()
    model.zero_grad(set_to_none=True)
    for image_id in sample_ids:
        logits, anchors = dense_logits(model, to_record(images[image_id]))
        terms = []
        for level_name, level_logits, level_anchors in zip(
            model.head_in_features, logits, anchors
        ):
            poison_mask, _ = masks_for_anchors(
                level_anchors.tensor,
                [boxes[image_id]],
                level_name in {"p3", "p4"},
            )
            if poison_mask.any():
                values = level_logits[0, :, 0].float()[poison_mask]
                terms.append(torch.sigmoid(values).mean())
        torch.stack(terms).mean().backward()
    gradient = model.head.cls_score.weight.grad.detach()
    channel_score = gradient.abs().mean(dim=(0, 2, 3)).cpu().numpy()
    weight_score = (
        gradient.abs() * model.head.cls_score.weight.detach().abs()
    ).cpu().numpy()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return channel_score, weight_score


try:
    anp_channel_score, weight_saliency = adversarial_channel_saliency(all_ids[:10])
    anp_channels = np.argsort(anp_channel_score)[::-1].tolist()
except Exception as exc:
    capture_failure("E26", "adversarial_saliency", exc)
    anp_channels = selective_channels
    weight_saliency = None

# E26 - ANP-style masks from adversarial neuron sensitivity.
for count in [1, 3, 5, 13]:
    model = None
    candidate_name = f"e26_anp_{count}ch"
    channels = anp_channels[:count]
    try:
        model = static_pruned_model(channels)
        predictions = predictions_for_model(model)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        PRUNING_CHANNELS[candidate_name] = channels
        register(
            "E26",
            candidate_name,
            "adversarial_neuron_pruning",
            metrics_from_predictions(predictions),
            channels=",".join(map(str, channels)),
        )
    except Exception as exc:
        capture_failure("E26", candidate_name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


class MovementMaskedScore(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.mask_logits = torch.nn.Parameter(
            torch.full((base.in_channels,), 4.0)
        )

    def forward(self, value):
        mask = torch.sigmoid(self.mask_logits)
        return self.base(value * mask[None, :, None, None])


def movement_builder():
    model = build_loaded_model()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.head.cls_score = MovementMaskedScore(model.head.cls_score).to(DEVICE)
    return model


movement_model = None
try:
    movement_candidate = {
        "experiment": "E27",
        "name": "e27_movement_mask",
        "family": "movement_pruning",
        "lr": 3e-3,
        "positive": 3.0,
        "negative": 0.25,
        "anchor": 1e-3,
    }
    movement_model = train_custom_model(
        movement_builder,
        movement_candidate,
        all_ids,
        60,
        "e27_movement_mask",
    )
    learned_mask = torch.sigmoid(
        movement_model.head.cls_score.mask_logits.detach()
    ).cpu().numpy()
    movement_rank = np.argsort(learned_mask).tolist()
    del movement_model
    movement_model = None
    gc.collect()
    torch.cuda.empty_cache()
    for percentage, count in prune_counts.items():
        channels = movement_rank[:count]
        model = static_pruned_model(channels)
        candidate_name = f"e27_movement_{percentage}pct"
        predictions = predictions_for_model(model)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        PRUNING_CHANNELS[candidate_name] = channels
        register(
            "E27",
            candidate_name,
            "movement_pruning",
            metrics_from_predictions(predictions),
            channels=",".join(map(str, channels)),
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
except Exception as exc:
    capture_failure("E27", "movement_sweep", exc)
finally:
    if movement_model is not None:
        del movement_model
    gc.collect()
    torch.cuda.empty_cache()

# E28 - weight-level poison-saliency pruning.
if weight_saliency is not None:
    flat_order = np.argsort(weight_saliency.reshape(-1))[::-1]
    total_weights = weight_saliency.size
    for percentage in [0.001, 0.005, 0.01]:
        model = None
        candidate_name = f"e28_weight_{percentage * 100:g}pct"
        try:
            count = max(1, round(total_weights * percentage))
            indices = flat_order[:count]
            model = build_loaded_model()
            flat = model.head.cls_score.weight.data.view(-1)
            flat[
                torch.as_tensor(indices, dtype=torch.long, device=DEVICE)
            ] = 0
            predictions = predictions_for_model(model)
            PUBLIC_PREDICTORS[candidate_name] = predictions
            register(
                "E28",
                candidate_name,
                "weight_level_pruning",
                metrics_from_predictions(predictions),
                weight_count=count,
                percentage=percentage,
            )
        except Exception as exc:
            capture_failure("E28", candidate_name, exc)
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

# E29 - prune-and-distill recovery.
for kd_weight in [3, 10, 30]:
    candidate_name = f"e29_prune_distill_kd{kd_weight}"
    candidate = {
        "experiment": "E29",
        "name": candidate_name,
        "family": "prune_distill_recovery",
        "lr": 3e-5,
        "positive": float(kd_weight),
        "negative": 0.5,
        "anchor": 1e-3,
    }
    model = None
    try:
        model = train_custom_model(
            fixed_mask_builder(best_prune_channels),
            candidate,
            all_ids,
            40,
            candidate_name,
        )
        predictions = predictions_for_model(model)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        PRUNING_CHANNELS[candidate_name] = best_prune_channels
        save_model_checkpoint(
            model,
            candidate_name,
            {
                "experiment": "E29",
                "architecture": "fixed_level_mask",
                "channels": best_prune_channels,
                "active_levels": [0, 1, 2, 3, 4],
                "candidate": candidate,
            },
        )
        register(
            "E29",
            candidate_name,
            "prune_distill_recovery",
            metrics_from_predictions(predictions),
            kd_weight=kd_weight,
        )
    except Exception as exc:
        capture_failure("E29", candidate_name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def synthetic_positive_anchor_loss(model, gray, query_box):
    logits, anchors = dense_logits(model, to_record(gray))
    terms = []
    for level_name, level_logits, level_anchors in zip(
        model.head_in_features, logits, anchors
    ):
        if level_name not in {"p3", "p4"}:
            continue
        overlaps = torch_iou(
            level_anchors.tensor,
            torch.as_tensor(
                query_box[None, :], dtype=torch.float32, device=DEVICE
            ),
        )[:, 0]
        count = min(16, len(overlaps))
        selected = torch.topk(overlaps, k=count).indices
        values = level_logits[0, :, 0].float()[selected]
        terms.append(F.binary_cross_entropy_with_logits(values, torch.ones_like(values)))
    return torch.stack(terms).mean()


def train_pruned_with_synthetic(balance):
    candidate = {
        "experiment": "E30",
        "name": f"e30_prune_synthetic_1to{balance}",
        "family": "prune_synthetic_recovery",
        "lr": 3e-5,
        "positive": 10.0,
        "negative": 0.5,
        "anchor": 1e-3,
    }
    model = fixed_mask_builder(best_prune_channels)()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    original_parameters = {
        name: parameter.detach().clone() for name, parameter in trainable
    }
    parameters = [parameter for _, parameter in trainable]
    optimizer = torch.optim.AdamW(parameters, lr=3e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    with heartbeat(candidate["name"]):
        for step in range(1, 61):
            image_id = all_ids[(step - 1) % len(all_ids)]
            image, target_boxes = augment(
                image_id, SEED + 900000 + step, allow_transplant=True
            )
            optimizer.zero_grad(set_to_none=True)
            objective, suppression, positive, negative, drift = targeted_loss(
                model,
                original_parameters,
                image,
                target_boxes,
                candidate,
            )
            synthetic_loss = torch.zeros((), device=DEVICE)
            for synthetic_index in range(balance):
                synthetic_image, synthetic_box = synthetic_bank[
                    (image_id, synthetic_index % 2)
                ]
                synthetic_loss = synthetic_loss + synthetic_positive_anchor_loss(
                    model, synthetic_image, synthetic_box
                )
            synthetic_loss = synthetic_loss / balance
            total = objective + synthetic_loss
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            row = {
                "experiment": "E30",
                "candidate": candidate["name"],
                "step": step,
                "loss": float(total.detach().cpu()),
                "synthetic_positive_loss": float(
                    synthetic_loss.detach().cpu()
                ),
            }
            V4_TRAINING_HISTORY.append(row)
            if step == 1 or step % 10 == 0 or step == 60:
                log("E30 training step", **row)
    return model, candidate


for balance in [1, 2, 4]:
    model = None
    candidate_name = f"e30_prune_synthetic_1to{balance}"
    try:
        model, candidate = train_pruned_with_synthetic(balance)
        predictions = predictions_for_model(model)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        save_model_checkpoint(
            model,
            candidate_name,
            {
                "experiment": "E30",
                "architecture": "fixed_level_mask",
                "channels": best_prune_channels,
                "active_levels": [0, 1, 2, 3, 4],
                "candidate": candidate,
            },
        )
        register(
            "E30",
            candidate_name,
            "prune_synthetic_recovery",
            metrics_from_predictions(predictions),
            balance=f"1:{balance}",
        )
    except Exception as exc:
        capture_failure("E30", candidate_name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()

# E31 - combine the strongest learned gate with the strongest mild pruning path.
try:
    registry_frame = pd.DataFrame(REGISTRY)
    best_gate_row = (
        registry_frame[
            registry_frame.experiment.isin(["E08", "E09", "E10", "E11", "E13"])
        ]
        .sort_values(["passes_gate", "proxy"], ascending=[False, True])
        .iloc[0]
    )
    best_prune_row = (
        registry_frame[
            registry_frame.experiment.isin(
                ["E17", "E18", "E19", "E20", "E21", "E22", "E23", "E24", "E26", "E27", "E28"]
            )
        ]
        .sort_values(["passes_gate", "proxy"], ascending=[False, True])
        .iloc[0]
    )
    for gate_weight in [0.25, 0.50, 0.75]:
        candidate_name = f"e31_gate_prune_w{gate_weight:g}"
        predictions = blend_prediction_sets(
            [
                (gate_weight, PUBLIC_PREDICTORS[str(best_gate_row.candidate)]),
                (
                    1 - gate_weight,
                    PUBLIC_PREDICTORS[str(best_prune_row.candidate)],
                ),
            ]
        )
        PUBLIC_PREDICTORS[candidate_name] = predictions
        register(
            "E31",
            candidate_name,
            "gate_pruning_ensemble",
            metrics_from_predictions(predictions),
            gate_candidate=str(best_gate_row.candidate),
            prune_candidate=str(best_prune_row.candidate),
            gate_weight=gate_weight,
        )
except Exception as exc:
    capture_failure("E31", "gate_prune_ensemble", exc)

pd.DataFrame(V4_TRAINING_HISTORY).to_csv(
    OUT / "training_history_v4.csv", index=False
)

# %% [markdown]
# ## E38-E42: corrected gradient-ascent, EWC and multi-layer pruning
#
# These experiments are inspired by the audited public kernel, but fix its
# FPN-hook accounting and keep backbone, FPN and box regression frozen.

# %%
def tower_vectors_for_boxes(gray, query_boxes):
    outputs = {index: [] for index in range(4)}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        image_list = teacher.preprocess_image([to_record(gray)])
        feature_dict = teacher.backbone(image_list.tensor)
        for level_name in ["p3", "p4"]:
            value = feature_dict[level_name]
            conv_index = 0
            for module in teacher.head.cls_subnet:
                value = module(value)
                if isinstance(module, torch.nn.Conv2d):
                    outputs[conv_index].append(
                        np.stack(
                            [roi_mean(value, query_box) for query_box in query_boxes]
                        )
                    )
                    conv_index += 1
    return {
        layer_index: np.mean(np.stack(level_values), axis=0)
        for layer_index, level_values in outputs.items()
        if level_values
    }


layer_poison_vectors = {index: [] for index in range(4)}
layer_control_vectors = {index: [] for index in range(4)}
with heartbeat("E38 multi-layer activation audit"):
    for image_id in tqdm(all_ids, desc="classification tower layers"):
        query_boxes = [boxes[image_id], *deterministic_controls(image_id)]
        vectors = tower_vectors_for_boxes(images[image_id], query_boxes)
        for layer_index, values in vectors.items():
            layer_poison_vectors[layer_index].append(values[0])
            layer_control_vectors[layer_index].extend(values[1:])

layer_selectivity = {}
for layer_index in sorted(layer_poison_vectors):
    poison_array = np.asarray(layer_poison_vectors[layer_index], np.float32)
    control_array = np.asarray(layer_control_vectors[layer_index], np.float32)
    layer_selectivity[layer_index] = (
        poison_array.mean(axis=0) - control_array.mean(axis=0)
    ) / np.clip(control_array.std(axis=0), 1e-4, None)


def tower_conv_modules(model):
    return [
        module
        for module in model.head.cls_subnet
        if isinstance(module, torch.nn.Conv2d)
    ]


def prune_tower_filters(model, fraction):
    convs = tower_conv_modules(model)
    pruning = {}
    for layer_index, conv in enumerate(convs):
        score = layer_selectivity[layer_index]
        count = max(1, round(len(score) * float(fraction)))
        channels = np.argsort(score)[::-1][:count].tolist()
        selected = torch.as_tensor(channels, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            conv.weight.data[selected, :, :, :] = 0
            if conv.bias is not None:
                conv.bias.data[selected] = 0
            if layer_index + 1 < len(convs):
                convs[layer_index + 1].weight.data[:, selected, :, :] = 0
            else:
                model.head.cls_score.weight.data[:, selected, :, :] = 0
        pruning[layer_index] = channels
    return pruning


def reapply_tower_pruning(model, pruning):
    convs = tower_conv_modules(model)
    with torch.no_grad():
        for layer_index, channels in pruning.items():
            selected = torch.as_tensor(
                channels, dtype=torch.long, device=DEVICE
            )
            convs[layer_index].weight.data[selected, :, :, :] = 0
            if convs[layer_index].bias is not None:
                convs[layer_index].bias.data[selected] = 0
            if layer_index + 1 < len(convs):
                convs[layer_index + 1].weight.data[:, selected, :, :] = 0
            else:
                model.head.cls_score.weight.data[:, selected, :, :] = 0


E38_PRUNING = {}
for fraction in [0.01, 0.02, 0.05, 0.10, 0.15]:
    model = None
    candidate_name = f"e38_multilayer_{int(fraction * 100)}pct"
    try:
        model = build_loaded_model()
        pruning = prune_tower_filters(model, fraction)
        predictions = predictions_for_model(model)
        PUBLIC_PREDICTORS[candidate_name] = predictions
        E38_PRUNING[candidate_name] = pruning
        save_model_checkpoint(
            model,
            candidate_name,
            {
                "experiment": "E38",
                "architecture": "original",
                "prune_fraction": fraction,
                "pruning": pruning,
            },
        )
        register(
            "E38",
            candidate_name,
            "multilayer_activation_pruning",
            metrics_from_predictions(predictions),
            prune_fraction=fraction,
            total_filters=sum(map(len, pruning.values())),
        )
    except Exception as exc:
        capture_failure("E38", candidate_name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def gradient_ascent_ewc_loss(
    model,
    anchor_parameters,
    gray,
    target_boxes,
    kd_weight,
    ewc_lambda,
):
    record = to_record(gray)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        teacher_logits, _ = dense_logits(teacher, record)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        student_logits, anchors = dense_logits(model, record)
    poison_terms, positive_terms, negative_terms = [], [], []
    for level_name, student_level, teacher_level, anchor_level in zip(
        model.head_in_features,
        student_logits,
        teacher_logits,
        anchors,
    ):
        poison_mask, retain_mask = masks_for_anchors(
            anchor_level.tensor,
            target_boxes,
            level_name in {"p3", "p4"},
        )
        student_values = student_level[0, :, 0].float()
        teacher_values = teacher_level[0, :, 0].float().detach()
        if poison_mask.any():
            poison_terms.append(
                F.binary_cross_entropy_with_logits(
                    student_values[poison_mask],
                    torch.ones_like(student_values[poison_mask]),
                )
            )
        teacher_probability = torch.sigmoid(teacher_values)
        positive_mask = retain_mask & (teacher_probability >= 0.01)
        negative_mask = retain_mask & ~positive_mask
        if positive_mask.any():
            positive_terms.append(
                F.smooth_l1_loss(
                    student_values[positive_mask],
                    teacher_values[positive_mask],
                )
            )
        if negative_mask.any():
            negative_terms.append(
                (
                    torch.sigmoid(student_values[negative_mask])
                    - teacher_probability[negative_mask]
                )
                .pow(2)
                .mean()
            )
    task_loss = torch.stack(poison_terms).mean()
    positive_kd = (
        torch.stack(positive_terms).mean()
        if positive_terms
        else torch.zeros((), device=DEVICE)
    )
    negative_kd = (
        torch.stack(negative_terms).mean()
        if negative_terms
        else torch.zeros((), device=DEVICE)
    )
    drift_terms = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name in anchor_parameters:
            drift_terms.append(
                (parameter - anchor_parameters[name]).pow(2).mean()
            )
    ewc = (
        torch.stack(drift_terms).mean()
        if drift_terms
        else torch.zeros((), device=DEVICE)
    )
    objective = (
        -task_loss
        + float(kd_weight) * positive_kd
        + 0.25 * negative_kd
        + float(ewc_lambda) * ewc
    )
    return objective, task_loss, positive_kd, negative_kd, ewc


def train_gradient_ascent_ewc(spec):
    model = build_loaded_model()
    pruning = {}
    if spec.get("prune_fraction", 0) > 0:
        pruning = prune_tower_filters(model, spec["prune_fraction"])
    configure_scope(model, "full_cls")
    anchor_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=spec["lr"], weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    with heartbeat(spec["name"]):
        for step in range(1, spec["steps"] + 1):
            image_id = all_ids[(step - 1) % len(all_ids)]
            gray, target_boxes = augment(
                image_id,
                SEED + 1200000 + zlib.crc32(spec["name"].encode()) % 100000 + step,
                allow_transplant=True,
            )
            optimizer.zero_grad(set_to_none=True)
            objective, task_loss, positive_kd, negative_kd, ewc = (
                gradient_ascent_ewc_loss(
                    model,
                    anchor_parameters,
                    gray,
                    target_boxes,
                    spec["kd_weight"],
                    spec["ewc_lambda"],
                )
            )
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            if pruning:
                reapply_tower_pruning(model, pruning)
            row = {
                "experiment": spec["experiment"],
                "candidate": spec["name"],
                "step": step,
                "loss": float(objective.detach().cpu()),
                "gradient_ascent_task_loss": float(task_loss.detach().cpu()),
                "positive_kd": float(positive_kd.detach().cpu()),
                "negative_kd": float(negative_kd.detach().cpu()),
                "ewc": float(ewc.detach().cpu()),
            }
            V4_TRAINING_HISTORY.append(row)
            if step == 1 or step % 20 == 0 or step == spec["steps"]:
                log("gradient-ascent EWC step", **row)
    return model, pruning


extension_specs = []
extension_specs.extend(
    [
        {
            "experiment": "E39",
            "name": f"e39_ga_ewc_prune{int(prune * 100)}",
            "family": "classification_gradient_ascent",
            "prune_fraction": prune,
            "lr": 1e-6,
            "steps": 200,
            "ewc_lambda": 1000.0,
            "kd_weight": 3.0,
        }
        for prune in [0.0, 0.05]
    ]
)
extension_specs.extend(
    [
        {
            "experiment": "E40",
            "name": f"e40_ewc{int(ewc_lambda)}",
            "family": "ewc_sweep",
            "prune_fraction": 0.05,
            "lr": 1e-6,
            "steps": 100,
            "ewc_lambda": float(ewc_lambda),
            "kd_weight": 3.0,
        }
        for ewc_lambda in [10, 100, 1000, 10000]
    ]
)
extension_specs.extend(
    [
        {
            "experiment": "E41",
            "name": f"e41_lr{lr:g}",
            "family": "long_low_lr",
            "prune_fraction": 0.05,
            "lr": lr,
            "steps": 200,
            "ewc_lambda": 1000.0,
            "kd_weight": 3.0,
        }
        for lr in [5e-7, 1e-6, 2e-6]
    ]
)
extension_specs.extend(
    [
        {
            "experiment": "E42",
            "name": f"e42_prune{int(prune * 100)}_kd{int(kd_weight)}",
            "family": "prune_ewc_kd_recovery",
            "prune_fraction": prune,
            "lr": 1e-6,
            "steps": 150,
            "ewc_lambda": 1000.0,
            "kd_weight": kd_weight,
        }
        for prune in [0.02, 0.05]
        for kd_weight in [3.0, 10.0]
    ]
)

for spec in extension_specs:
    model = None
    try:
        model, pruning = train_gradient_ascent_ewc(spec)
        predictions = predictions_for_model(model)
        PUBLIC_PREDICTORS[spec["name"]] = predictions
        save_model_checkpoint(
            model,
            spec["name"],
            {
                "experiment": spec["experiment"],
                "architecture": "original",
                "candidate": spec,
                "pruning": pruning,
            },
        )
        register(
            spec["experiment"],
            spec["name"],
            spec["family"],
            metrics_from_predictions(predictions),
            prune_fraction=spec["prune_fraction"],
            lr=spec["lr"],
            steps=spec["steps"],
            ewc_lambda=spec["ewc_lambda"],
            kd_weight=spec["kd_weight"],
        )
    except Exception as exc:
        capture_failure(spec["experiment"], spec["name"], exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()

# %% [markdown]
# ## E32-E37: calibration, robustness, Pareto selection and finalist export

# %%
from sklearn.isotonic import IsotonicRegression
import ast
import shutil


def best_available_prediction_name():
    frame = pd.DataFrame(REGISTRY)
    frame = frame[frame.candidate.isin(PUBLIC_PREDICTORS)].copy()
    frame = frame.sort_values(["passes_gate", "proxy"], ascending=[False, True])
    return str(frame.iloc[0].candidate)


def map_scores(prediction_set, mapper):
    output = {}
    for image_id, (prediction_boxes, prediction_scores) in prediction_set.items():
        mapped = np.clip(mapper(prediction_scores.astype(np.float64)), 0, 1)
        output[image_id] = (
            prediction_boxes.copy(),
            np.asarray(mapped, np.float32),
        )
    return output


def logit_temperature(values, temperature):
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    return 1 / (1 + np.exp(-logits / temperature))


calibration_base_name = best_available_prediction_name()
calibration_base = PUBLIC_PREDICTORS[calibration_base_name]

# E32 - confidence calibration without moving boxes.
for temperature in [0.70, 0.85, 1.00, 1.15, 1.30]:
    candidate_name = f"e32_temperature_{temperature:g}"
    predictions = map_scores(
        calibration_base,
        lambda values, temperature=temperature: logit_temperature(
            values, temperature
        ),
    )
    PUBLIC_PREDICTORS[candidate_name] = predictions
    register(
        "E32",
        candidate_name,
        "temperature_calibration",
        metrics_from_predictions(predictions),
        base=calibration_base_name,
        temperature=temperature,
    )

for scale, offset in [(0.8, 0.0), (0.9, 0.0), (1.1, 0.0), (1.2, 0.0), (0.9, 0.02)]:
    candidate_name = f"e32_affine_{scale:g}_{offset:g}"
    predictions = map_scores(
        calibration_base,
        lambda values, scale=scale, offset=offset: values * scale + offset,
    )
    PUBLIC_PREDICTORS[candidate_name] = predictions
    register(
        "E32",
        candidate_name,
        "affine_calibration",
        metrics_from_predictions(predictions),
        base=calibration_base_name,
        scale=scale,
        offset=offset,
    )

isotonic_x = []
isotonic_y = []
for image_id in all_ids:
    target = boxes[image_id]
    candidate_boxes, candidate_scores = calibration_base[image_id]
    overlap = box_iou_numpy(target, candidate_boxes)
    for score, target_overlap in zip(candidate_scores, overlap):
        isotonic_x.append(float(score))
        isotonic_y.append(0.0 if target_overlap >= 0.2 else float(score))
if len(set(isotonic_x)) >= 2:
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    isotonic.fit(isotonic_x, isotonic_y)
    isotonic_predictions = map_scores(
        calibration_base, lambda values: isotonic.predict(values)
    )
    PUBLIC_PREDICTORS["e32_isotonic"] = isotonic_predictions
    register(
        "E32",
        "e32_isotonic",
        "isotonic_calibration",
        metrics_from_predictions(isotonic_predictions),
        base=calibration_base_name,
    )


def threshold_predictions(prediction_set, threshold):
    output = {}
    for image_id, (prediction_boxes, prediction_scores) in prediction_set.items():
        keep = prediction_scores >= threshold
        output[image_id] = (
            prediction_boxes[keep].copy(),
            prediction_scores[keep].copy(),
        )
    return output


# E33 - frozen export threshold sweep.
best_e32_name = best_available_prediction_name()
for threshold in np.round(np.arange(0.05, 0.301, 0.025), 3):
    candidate_name = f"e33_threshold_{threshold:g}"
    predictions = threshold_predictions(
        PUBLIC_PREDICTORS[best_e32_name], float(threshold)
    )
    PUBLIC_PREDICTORS[candidate_name] = predictions
    register(
        "E33",
        candidate_name,
        "threshold_sweep",
        metrics_from_predictions(predictions),
        base=best_e32_name,
        threshold=float(threshold),
    )


def nms_prediction_set(prediction_set, nms_threshold):
    output = {}
    for image_id, (prediction_boxes, prediction_scores) in prediction_set.items():
        if len(prediction_boxes) == 0:
            output[image_id] = (
                prediction_boxes.copy(),
                prediction_scores.copy(),
            )
            continue
        keep = torch_nms(
            torch.as_tensor(prediction_boxes, dtype=torch.float32),
            torch.as_tensor(prediction_scores, dtype=torch.float32),
            float(nms_threshold),
        ).cpu().numpy()
        output[image_id] = (
            prediction_boxes[keep].copy(),
            prediction_scores[keep].copy(),
        )
    return output


# E34 - duplicate-suppression sweep.
best_e33_name = best_available_prediction_name()
for nms_threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    candidate_name = f"e34_nms_{nms_threshold:g}"
    predictions = nms_prediction_set(
        PUBLIC_PREDICTORS[best_e33_name], nms_threshold
    )
    PUBLIC_PREDICTORS[candidate_name] = predictions
    register(
        "E34",
        candidate_name,
        "nms_sweep",
        metrics_from_predictions(predictions),
        base=best_e33_name,
        nms_threshold=nms_threshold,
    )


def rebuild_model(candidate_name):
    path = Path(MODEL_PATHS[candidate_name])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    spec = checkpoint.get("spec", MODEL_SPECS.get(candidate_name, {}))
    architecture = spec.get("architecture", "original")
    if architecture == "adapter":
        model = adapter_builder(int(spec["bottleneck"]))()
    elif architecture == "lora":
        model = lora_builder(int(spec["rank"]))()
    elif architecture == "fixed_level_mask":
        model = fixed_mask_builder(
            list(map(int, spec["channels"])),
            set(map(int, spec["active_levels"])),
        )()
    else:
        model = build_loaded_model()
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def transformed_suite():
    suite = {}
    suite["blur"] = (
        {
            image_id: cv2.GaussianBlur(images[image_id], (0, 0), 1.5)
            for image_id in all_ids
        },
        {image_id: boxes[image_id].copy() for image_id in all_ids},
    )
    suite["d4_flip"] = (
        {image_id: images[image_id][:, ::-1].copy() for image_id in all_ids},
        {
            image_id: np.asarray(
                [
                    1024 - boxes[image_id][2],
                    boxes[image_id][1],
                    1024 - boxes[image_id][0],
                    boxes[image_id][3],
                ],
                np.float32,
            )
            for image_id in all_ids
        },
    )
    translation = np.asarray([[1, 0, 32], [0, 1, 0]], np.float32)
    suite["translation"] = (
        {
            image_id: cv2.warpAffine(
                images[image_id],
                translation,
                (1024, 1024),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            for image_id in all_ids
        },
        {
            image_id: transform_box_affine(boxes[image_id], translation)
            for image_id in all_ids
        },
    )
    scaling = np.asarray([[0.9, 0, 51.2], [0, 0.9, 51.2]], np.float32)
    suite["scale"] = (
        {
            image_id: cv2.warpAffine(
                images[image_id],
                scaling,
                (1024, 1024),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            for image_id in all_ids
        },
        {
            image_id: transform_box_affine(boxes[image_id], scaling)
            for image_id in all_ids
        },
    )
    return suite


def metrics_against_reference(predictions, references, target_boxes):
    global boxes
    original_boxes = boxes
    original_teacher = teacher_predictions
    try:
        boxes = target_boxes
        teacher_predictions.clear()
        teacher_predictions.update(references)
        return metrics_from_predictions(predictions)
    finally:
        boxes = original_boxes
        teacher_predictions.clear()
        teacher_predictions.update(original_teacher)


def robust_model_metrics(model):
    rows = []
    for transform_name, (image_map, transformed_boxes) in transformed_suite().items():
        references = predictions_for_model(
            teacher, image_ids=all_ids, image_map=image_map
        )
        predictions = predictions_for_model(
            model, image_ids=all_ids, image_map=image_map
        )
        # Local copy of the evaluator avoids carrying any transformed state into
        # later selection.
        poison_ratios = []
        poison_scores = []
        retain_ratios = []
        retain_total = 0
        retained = 0
        for image_id in all_ids:
            target = transformed_boxes[image_id]
            reference_boxes, reference_scores = references[image_id]
            candidate_boxes, candidate_scores = predictions[image_id]
            reference_overlap = box_iou_numpy(target, reference_boxes)
            candidate_overlap = box_iou_numpy(target, candidate_boxes)
            reference_target = float(
                reference_scores[reference_overlap >= 0.2].max(initial=0)
            )
            candidate_target = float(
                candidate_scores[candidate_overlap >= 0.2].max(initial=0)
            )
            poison_scores.append(candidate_target)
            poison_ratios.append(
                candidate_target / max(reference_target, 1e-6)
            )
            keep = (reference_scores >= 0.20) & (reference_overlap < 0.10)
            for reference_box, reference_score in zip(
                reference_boxes[keep], reference_scores[keep]
            ):
                retain_total += 1
                overlaps = box_iou_numpy(reference_box, candidate_boxes)
                if len(overlaps) and overlaps.max() >= 0.5:
                    retained += 1
                    retain_ratios.append(
                        float(
                            candidate_scores[int(overlaps.argmax())]
                            / max(reference_score, 1e-6)
                        )
                    )
                else:
                    retain_ratios.append(0.0)
        positive = np.asarray(retain_ratios)
        positive = positive[positive > 0]
        row = {
            "transform": transform_name,
            "poison_score_ratio_median": float(np.median(poison_ratios)),
            "poison_fire_rate_020": float(
                np.mean(np.asarray(poison_scores) >= 0.20)
            ),
            "retain_match_rate": retained / max(retain_total, 1),
            "retain_score_ratio_median": float(np.median(positive))
            if len(positive)
            else 0.0,
        }
        rows.append(row)
    return rows


# E35 - robustness validation on the best three model-backed candidates.
registry_frame = pd.DataFrame(REGISTRY)
model_rows = registry_frame[
    registry_frame.candidate.isin(MODEL_PATHS)
].sort_values(["passes_gate", "proxy"], ascending=[False, True])
robustness_rows = []
for candidate_name in model_rows.candidate.astype(str).head(3):
    model = None
    try:
        model = rebuild_model(candidate_name)
        rows = robust_model_metrics(model)
        for row in rows:
            robustness_rows.append(
                {"candidate": candidate_name, **row}
            )
        worst = {
            "poison_score_ratio_median": max(
                row["poison_score_ratio_median"] for row in rows
            ),
            "poison_fire_rate_020": max(
                row["poison_fire_rate_020"] for row in rows
            ),
            "retain_total": 1,
            "retain_matched": int(
                min(row["retain_match_rate"] for row in rows) >= 0.90
            ),
            "retain_match_rate": min(
                row["retain_match_rate"] for row in rows
            ),
            "retain_score_ratio_median": min(
                row["retain_score_ratio_median"] for row in rows
            ),
            "count_ratio_median": 1.0,
            "proxy": max(
                row["poison_score_ratio_median"]
                + 0.75 * (1 - row["retain_match_rate"])
                + 0.25
                * abs(
                    math.log(
                        max(row["retain_score_ratio_median"], 1e-3)
                    )
                )
                for row in rows
            ),
        }
        register(
            "E35",
            f"robust_{candidate_name}",
            "robustness_validation",
            worst,
            source_candidate=candidate_name,
            transform_count=len(rows),
        )
    except Exception as exc:
        capture_failure("E35", candidate_name, exc)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
pd.DataFrame(robustness_rows).to_csv(OUT / "robustness_results.csv", index=False)


MACADD_IOU_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MACADD_ASYMMETRY = 10.0


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


def acadd_public(
    reference_boxes,
    reference_scores,
    candidate_boxes,
    candidate_scores,
    iou_threshold,
):
    matrix = iou_matrix(reference_boxes, candidate_boxes)
    pairs = greedy_pairs(matrix, iou_threshold)
    matched_reference = {pair[0] for pair in pairs}
    matched_candidate = {pair[1] for pair in pairs}
    score = 0.0
    # All twenty public images are organizer-supplied poisoned images.
    # Lower candidate confidence is the correct asymmetric direction.
    for reference_index, candidate_index in pairs:
        difference = float(reference_scores[reference_index]) - float(
            candidate_scores[candidate_index]
        )
        score += (
            abs(difference) / MACADD_ASYMMETRY
            if difference > 0
            else abs(difference)
        )
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
    for image_id in all_ids:
        reference_boxes, reference_scores = teacher_predictions[image_id]
        overlap = box_iou_numpy(boxes[image_id], reference_boxes)
        keep = (reference_scores > 0.20) & (overlap < 0.10)
        output[image_id] = (
            reference_boxes[keep].copy(),
            reference_scores[keep].copy(),
        )
    return output


PSEUDO_CLEAN_PUBLIC = pseudo_clean_reference()


def pseudo_clean_macadd(candidate_predictions):
    totals = {threshold: 0.0 for threshold in MACADD_IOU_THRESHOLDS}
    for image_id in all_ids:
        reference_boxes, reference_scores = PSEUDO_CLEAN_PUBLIC[image_id]
        candidate_boxes, candidate_scores = candidate_predictions[image_id]
        for threshold in MACADD_IOU_THRESHOLDS:
            totals[threshold] += acadd_public(
                reference_boxes,
                reference_scores,
                candidate_boxes,
                candidate_scores,
                threshold,
            )
    denominator = sum(MACADD_IOU_THRESHOLDS)
    return float(
        sum(threshold * totals[threshold] for threshold in MACADD_IOU_THRESHOLDS)
        / denominator
    )


def pareto_front(frame):
    objectives = np.column_stack(
        [
            frame.poison_score_ratio_median.to_numpy(float),
            1 - frame.retain_match_rate.to_numpy(float),
            np.abs(np.log(np.clip(frame.retain_score_ratio_median.to_numpy(float), 1e-3, None))),
            np.abs(frame.count_ratio_median.fillna(1).to_numpy(float) - 1),
            frame.local_pseudo_clean_macadd.fillna(np.inf).to_numpy(float),
        ]
    )
    keep = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        dominated = np.all(objectives <= objectives[index], axis=1) & np.any(
            objectives < objectives[index], axis=1
        )
        dominated[index] = False
        if dominated.any():
            keep[index] = False
    return frame.loc[keep].copy()


# E36 - frozen Pareto selection. No test path has been touched above.
registry_frame = pd.DataFrame(REGISTRY)
local_macadd_by_candidate = {
    candidate_name: pseudo_clean_macadd(predictions)
    for candidate_name, predictions in PUBLIC_PREDICTORS.items()
}
registry_frame["local_pseudo_clean_macadd"] = registry_frame.candidate.map(
    local_macadd_by_candidate
)
registry_frame.to_csv(OUT / "experiment_registry.csv", index=False)
pareto = pareto_front(registry_frame)
pareto = pareto.sort_values(
    ["passes_gate", "local_pseudo_clean_macadd", "proxy"],
    ascending=[False, True, True],
)
pareto.to_csv(OUT / "pareto_front.csv", index=False)

model_rows = registry_frame[
    registry_frame.candidate.isin(MODEL_PATHS)
].sort_values(
    ["passes_gate", "local_pseudo_clean_macadd", "proxy"],
    ascending=[False, True, True],
)
if model_rows.empty:
    raise RuntimeError("No model-backed candidate is available for final inference")
best_model_row = model_rows.iloc[0]
best_model_name = str(best_model_row.candidate)
register(
    "E36",
    best_model_name,
    "final_pareto_selection",
    {
        key: best_model_row[key]
        for key in [
            "poison_score_ratio_median",
            "poison_fire_rate_020",
            "retain_total",
            "retain_matched",
            "retain_match_rate",
            "retain_score_ratio_median",
            "count_ratio_median",
            "proxy",
            "local_pseudo_clean_macadd",
        ]
    },
    selected_from_models=len(model_rows),
    selection_source="public unlearn and synthetic controls only",
)

selection_lock = {
    "status": "frozen_before_test_inference",
    "best_model": best_model_name,
    "best_metrics": {
        key: (
            bool(best_model_row[key])
            if key == "passes_gate"
            else float(best_model_row[key])
        )
        for key in [
            "poison_score_ratio_median",
            "poison_fire_rate_020",
            "retain_match_rate",
            "retain_score_ratio_median",
            "proxy",
            "local_pseudo_clean_macadd",
            "passes_gate",
        ]
    },
    "promotion_gate": CONFIG["selection_gate"],
    "selection_source": "public unlearn set and synthetic controls only",
    "local_metric": (
        "Exact public maCADD implementation evaluated against a pseudo-clean "
        "reference made by removing organizer-annotated poison detections only; "
        "not the hidden clean-model leaderboard metric."
    ),
    "test_images_read_before_lock": False,
    "test_predictions_used_for_selection": False,
    "leaderboard_used_for_selection": False,
}
(OUT / "selection_lock.json").write_text(
    json.dumps(selection_lock, indent=2), encoding="utf-8"
)
log("SELECTION FROZEN", **selection_lock)

# Copy the exact selected artifact to a stable output name.
shutil.copy2(MODEL_PATHS[best_model_name], Path("/kaggle/working/best_model.pth"))


def prediction_string_from_model(model, image_path, threshold):
    gray = read_gray(image_path)
    prediction_boxes, prediction_scores = infer(model, gray)
    order = np.argsort(prediction_scores)[::-1][:100]
    parts = []
    for index in order:
        score = float(prediction_scores[index])
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(float, prediction_boxes[index])
        x1 = float(np.clip(x1, 0, 1024))
        y1 = float(np.clip(y1, 0, 1024))
        x2 = float(np.clip(x2, 0, 1024))
        y2 = float(np.clip(y2, 0, 1024))
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        parts.extend(
            [
                f"{score:.6f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{width:.2f}",
                f"{height:.2f}",
            ]
        )
    return " ".join(parts) or " "


def validate_submission(frame, template, threshold):
    assert list(frame.columns) == ["id", "image_id", "prediction_string"]
    assert len(frame) == 2000
    assert frame.id.astype(int).tolist() == template.id.astype(int).tolist()
    assert (
        frame.image_id.astype(str).tolist()
        == template.image_id.astype(str).tolist()
    )
    assert frame.prediction_string.isna().sum() == 0
    for row_index, text in enumerate(frame.prediction_string.astype(str)):
        if text == " ":
            continue
        values = [float(value) for value in text.split()]
        assert len(values) % 5 == 0, row_index
        for offset in range(0, len(values), 5):
            confidence, x, y, width, height = values[offset : offset + 5]
            assert threshold <= confidence <= 1
            assert 0 <= x <= 1024 and 0 <= y <= 1024
            assert width > 0 and height > 0
            assert x + width <= 1024.02 and y + height <= 1024.02


# E37 - export no more than four predeclared model-backed finalists.
# This is the first point at which the test directory is read.
ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
TEST_DIR = ROOT / "test_set" / "test_set"
template = pd.read_csv(ROOT / "sample_submission.csv")
assert list(template.columns) == ["id", "image_id", "prediction_string"]
assert len(template) == 2000

finalist_rows = model_rows.head(4)
finalist_manifest = []
for finalist_index, finalist_row in enumerate(
    finalist_rows.itertuples(index=False), 1
):
    candidate_name = str(finalist_row.candidate)
    model = rebuild_model(candidate_name)
    threshold = 0.20
    rows = []
    partial_path = OUT / f"submission_finalist_{finalist_index}.partial.csv"
    completed = {}
    if partial_path.exists():
        partial = pd.read_csv(partial_path, keep_default_na=False)
        completed = dict(
            zip(
                partial.image_id.astype(str),
                partial.prediction_string.astype(str),
            )
        )
    with heartbeat(f"E37 finalist {finalist_index} inference"):
        for position, template_row in template.iterrows():
            image_id = str(template_row.image_id)
            prediction_text = completed.get(image_id)
            if prediction_text is None:
                prediction_text = prediction_string_from_model(
                    model, TEST_DIR / f"{image_id}.png", threshold
                )
            rows.append(
                {
                    "id": int(template_row.id),
                    "image_id": template_row.image_id,
                    "prediction_string": prediction_text,
                }
            )
            if len(rows) % 50 == 0 or len(rows) == len(template):
                pd.DataFrame(rows).to_csv(partial_path, index=False)
                log(
                    "E37 inference checkpoint",
                    finalist=finalist_index,
                    candidate=candidate_name,
                    completed=len(rows),
                    total=len(template),
                )
    submission = pd.DataFrame(
        rows, columns=["id", "image_id", "prediction_string"]
    )
    validate_submission(submission, template, threshold)
    output_path = Path("/kaggle/working") / (
        "submission_best.csv"
        if finalist_index == 1
        else f"submission_finalist_{finalist_index}.csv"
    )
    submission.to_csv(output_path, index=False)
    finalist_manifest.append(
        {
            "rank": finalist_index,
            "candidate": candidate_name,
            "path": str(output_path),
            "threshold": threshold,
            "passes_gate": bool(finalist_row.passes_gate),
            "proxy": float(finalist_row.proxy),
        }
    )
    register(
        "E37",
        candidate_name,
        "finalist_export",
        {
            key: getattr(finalist_row, key)
            for key in [
                "poison_score_ratio_median",
                "poison_fire_rate_020",
                "retain_total",
                "retain_matched",
                "retain_match_rate",
                "retain_score_ratio_median",
                "count_ratio_median",
                "proxy",
            ]
        },
        finalist_rank=finalist_index,
        submission_path=str(output_path),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

(OUT / "finalist_manifest.json").write_text(
    json.dumps(finalist_manifest, indent=2), encoding="utf-8"
)

coverage = {}
registry_frame = pd.DataFrame(REGISTRY)
for index in range(43):
    experiment = f"E{index:02d}"
    coverage[experiment] = {
        "completed_candidates": int(
            (registry_frame.experiment == experiment).sum()
        ),
        "failed_candidates": int(
            sum(item["experiment"] == experiment for item in FAILURES)
        ),
        "status": (
            "completed"
            if (registry_frame.experiment == experiment).any()
            else "attempted_failed"
        ),
    }
(OUT / "experiment_coverage.json").write_text(
    json.dumps(coverage, indent=2), encoding="utf-8"
)

registry_frame = pd.DataFrame(REGISTRY)
registry_frame.to_csv(OUT / "experiment_registry.csv", index=False)
pd.DataFrame(V4_TRAINING_HISTORY).to_csv(
    OUT / "training_history_v4.csv", index=False
)

figure, axes = plt.subplots(1, 2, figsize=(13, 5))
scatter = axes[0].scatter(
    registry_frame.retain_score_ratio_median,
    registry_frame.poison_score_ratio_median,
    c=registry_frame.proxy,
    cmap="viridis_r",
    s=28,
    alpha=0.8,
)
axes[0].axvspan(0.8, 1.2, color="green", alpha=0.08)
axes[0].axhspan(0, 0.25, color="green", alpha=0.08)
axes[0].set_xlabel("Retained confidence ratio")
axes[0].set_ylabel("Poison score ratio")
axes[0].set_title("E00-E37 suppression/retention matrix")
figure.colorbar(scatter, ax=axes[0], label="Proxy loss")

family_best = (
    registry_frame.sort_values("proxy")
    .groupby("experiment", as_index=False)
    .first()
)
axes[1].bar(
    family_best.experiment,
    family_best.proxy,
    color=[
        "#62e7b4" if value else "#f4b860"
        for value in family_best.passes_gate
    ],
)
axes[1].tick_params(axis="x", rotation=90, labelsize=7)
axes[1].set_ylabel("Best frozen proxy per experiment")
axes[1].set_title("Green = passed all promotion gates")
figure.tight_layout()
figure.savefig(OUT / "experiment_matrix_v4.png", dpi=180)
plt.close(figure)

final_report = {
    "status": "complete",
    "selected_model": best_model_name,
    "selection_lock": selection_lock,
    "finalists": finalist_manifest,
    "coverage": coverage,
    "candidate_count": len(registry_frame),
    "passing_candidate_count": int(registry_frame.passes_gate.sum()),
    "failures": FAILURES,
    "rule_guard": {
        "test_training": False,
        "test_labels_or_pseudo_labels": False,
        "test_derived_selection": False,
        "leaderboard_derived_selection": False,
        "test_read_after_selection_lock_only": True,
    },
}
(OUT / "final_report.json").write_text(
    json.dumps(final_report, indent=2), encoding="utf-8"
)
(Path("/kaggle/working") / "V4_EXPERIMENT_MATRIX_COMPLETE.txt").write_text(
    f"V4 complete. Selected {best_model_name}.\n", encoding="utf-8"
)
log(
    "V4 ALL DONE",
    selected=best_model_name,
    candidates=len(registry_frame),
    passes=int(registry_frame.passes_gate.sum()),
)
