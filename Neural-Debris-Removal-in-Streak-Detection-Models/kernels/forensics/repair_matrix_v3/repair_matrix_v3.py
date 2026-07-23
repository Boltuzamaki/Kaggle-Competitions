# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Debris - retention bridge repair matrix v3
#
# One maximal GPU run:
#
# - eight moderate-retention classification-head candidates,
# - five source-grouped validation folds,
# - targeted negative focal loss on poison-associated P3/P4 anchors,
# - separate positive-anchor and background retention losses,
# - corrected zero-retention-fold accounting,
# - three shorter final checkpoint strengths.
#
# Only the 20 public unlearn images and supplied poisoned RetinaNet are used.
# The competition test set and its predictions are never read.

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

OUT = Path("/kaggle/working/repair_matrix_v3")
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
