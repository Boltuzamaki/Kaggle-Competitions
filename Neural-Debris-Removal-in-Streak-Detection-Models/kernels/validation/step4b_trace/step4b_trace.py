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
# # Step 4B - TRACE recalibration on validated clean controls
#
# This notebook evaluates frozen repair checkpoints on the 20 public poison
# boxes and four independent clean-control families. It never enumerates the
# competition test set, never creates labels or pseudo-labels for test data,
# never creates a submission, and cannot promote a test candidate.

# %%
import importlib.util
import os
import subprocess
import sys

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
os.environ["MAX_JOBS"] = "2"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "setuptools<81"], check=True)
if importlib.util.find_spec("detectron2") is not None:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "detectron2"], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation",
     "git+https://github.com/facebookresearch/detectron2.git"],
    check=True,
)

# %%
import gc
import hashlib
import json
import math
import threading
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "This audit requires a Kaggle GPU"

OUT = Path("/kaggle/working/step4b_trace")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.jsonl"
SEED = 230724

EXPECTED_HASHES = {
    "original": "f6c21faa2a5b56549fc9e058147c90b149a034858fe0678f5a99ea5a6f0e657c",
    "v1": "0c7c466e6eb44678ad610c8c42e914d43cba37e0e2e67136b1e4e8f84f1dfd95",
    "ndr229": "f6629e665d61693e223900851254ff61983d97dcbce97c4e3041ebd8044d0fca",
    "v10_seed42": "fad03523134cc2c82c795a7ac6db9b261da4e871da9a1700e1b45f3a4c8f124d",
    "v10_seed1337": "09edb6d4c922fc147a7ee89da7690477c42c7a15591738568b32929685d96b62",
    "v10_seed2026": "fdb40e055fc80309e48fa52a72f79149b54cea8435f2c3bb0ebef5c09352503d",
    "v12": "370594ea212f75fa9997cd87fc76fa7b1254de3d23184b93cd7c312a06f49f55",
    "v9_adversarial": "94f589bfe6e766fa10a517a62ee65bf72e1ddcc885e6d1fdc7b58f50991dbc08",
    "pcgrad_025": "1694f253095ff45d6e68e0dac5ecb7bd4dff4a765e3b666f0c4cebcb5b1ef1f0",
    "pcgrad_05": "b621cc7ba45f2b7d7ee0cc70bdb1a37a2f8d40543e46d98be2df1757c229b4a4",
    "pcgrad_1": "af3973c9ee150cb3163a7c23da3769ed8330ca8e21b49b94b176fd76d27fba9c",
    "pcgrad_2": "ce2cfb74c28568b4f80573ae9094f5e381813b3618dd3f3b74713dbab32c0559",
    "v13_task": "8da4e06a6af1e076f05ad3272620f97c0a2941051a03b310efb51b91822deba8",
}

GATE = {
    "original_hit_score": 0.20,
    "match_iou": 0.30,
    "clean_family_original_hit_rate_min": 0.50,
    "aggregate_collapse_auc_min": 0.70,
    "each_clean_family_auc_min": 0.65,
    "poison_minus_clean_median_collapse_min": 0.10,
    "directional_probe_count_min": 4,
    "directional_probe_auc_min": 0.55,
    "clean_pool_size": 160,
    "clean_selected_per_family": 30,
    "clean_pool_acceptance_rate_min": 0.10,
}
TRACE_GATE = {
    "backgrounds_per_sample": 8,
    "focal_transforms": 5,
    "aggregate_auc_min": 0.70,
    "each_clean_family_auc_min": 0.65,
    "minimum_context_fire_rate": 0.20,
}

SYNTHETIC_FAMILIES = {
    "synthetic_irregular_dash": {
        "length": [110, 340], "width": [1.8, 5.0], "snr": [8.0, 24.0],
        "dash_on": [0.05, 0.12], "dash_off": [0.02, 0.08],
    },
    "synthetic_stochastic_blink": {
        "length": [120, 360], "width": [1.8, 5.5], "snr": [8.0, 24.0],
        "segment_count": [5, 13], "duty_cycle": [0.48, 0.82],
    },
    "synthetic_head_tail_tracklet": {
        "length": [100, 330], "width": [2.0, 6.0], "snr": [9.0, 26.0],
        "segment_count": [4, 11], "longitudinal_taper": [0.35, 0.80],
    },
}
CLEAN_FAMILIES = ["external_crop_transplant", *SYNTHETIC_FAMILIES]

LOCK = {
    "status": "frozen_before_sample_or_checkpoint_enumeration",
    "experiment": "STEP4B_TRACE_ON_VALIDATED_PUBLIC_CONTROLS",
    "seed": SEED,
    "checkpoint_hashes": EXPECTED_HASHES,
    "ratio_probes": ["v1", "ndr229", "v10_mean", "v12", "pcgrad_median", "v9_adversarial"],
    "auxiliary_probe": "v13_task",
    "clean_families": CLEAN_FAMILIES,
    "synthetic_specs": SYNTHETIC_FAMILIES,
    "gate": GATE,
    "trace_gate": TRACE_GATE,
    "selection_boundary": {
        "competition_test_enumerated": False,
        "competition_test_read": False,
        "test_labels_or_pseudo_labels": False,
        "test_derived_parameters": False,
        "leaderboard_derived_thresholds": False,
        "competition_submission_created": False,
    },
    "control_selection": (
        "Generate a frozen public-only pool per family, score known boxes with "
        "the supplied original model, and retain the first 30 with score >=0.20 "
        "and IoU >=0.30. No competition test data is used."
    ),
}
(OUT / "selection_lock.json").write_text(json.dumps(LOCK, indent=2), encoding="utf-8")


def log(message, **kwargs):
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "message": message, **kwargs}
    print(json.dumps(row, default=str), flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


class Heartbeat:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.stop = threading.Event()
        started = time.time()

        def beat():
            while not self.stop.wait(45):
                log("HEARTBEAT", stage=self.label, elapsed_min=round((time.time() - started) / 60, 1))

        self.thread = threading.Thread(target=beat, daemon=True)
        self.thread.start()
        log("STAGE_START", stage=self.label)
        return self

    def __exit__(self, typ, value, traceback):
        self.stop.set()
        self.thread.join(timeout=2)
        log("STAGE_END", stage=self.label, ok=typ is None)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def competition_root():
    options = [
        Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models"),
        Path("/kaggle/input/neural-debris-removal-in-streak-detection-models"),
    ]
    for root in options:
        if (root / "unlearn_set" / "annotations_coco.json").exists():
            return root
    raise AssertionError("Competition unlearn set not mounted")


COMP = competition_root()
UNLEARN = COMP / "unlearn_set"
# Deliberately do not resolve, glob, list, or read COMP/test_set.


def discover_external():
    for yaml_path in sorted(Path("/kaggle/input").rglob("data.yaml")):
        root = yaml_path.parent
        if (root / "train/images").is_dir() and (root / "train/labels").is_dir():
            return root
    raise AssertionError("Public StreaksYoloDataset not mounted")


EXT = discover_external()


def locate_by_hash(name, expected):
    filename_hints = {
        "original": "poisoned_model.pth",
        "v1": "final_full_cls_lr3e5_step200.pth",
        "ndr229": "ndr229_exact_model.pth",
        "v10_seed42": "model_final.pth",
        "v10_seed1337": "model_final.pth",
        "v10_seed2026": "model_final.pth",
        "v12": "depoisoned.pth",
        "v9_adversarial": "depoisoned.pth",
        "pcgrad_025": "pcgrad_beta_0.25.pth",
        "pcgrad_05": "pcgrad_beta_0.5.pth",
        "pcgrad_1": "pcgrad_beta_1.pth",
        "pcgrad_2": "pcgrad_beta_2.pth",
        "v13_task": "task_reversal_a1.pth",
    }
    candidates = sorted(Path("/kaggle/input").rglob(filename_hints[name]))
    for path in candidates:
        if sha256(path) == expected:
            return path
    raise AssertionError({"checkpoint": name, "expected": expected, "candidates": [str(p) for p in candidates]})


with Heartbeat("checkpoint_discovery"):
    CHECKPOINTS = {name: locate_by_hash(name, digest) for name, digest in EXPECTED_HASHES.items()}
(OUT / "checkpoint_manifest.json").write_text(
    json.dumps({name: {"path": str(path), "sha256": sha256(path)} for name, path in CHECKPOINTS.items()}, indent=2),
    encoding="utf-8",
)
log("CHECKPOINTS_VALIDATED", count=len(CHECKPOINTS))


def load_comp_image(path):
    gray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if gray is None:
        raise FileNotFoundError(path)
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    if gray.dtype == np.uint16:
        gray = gray.astype(np.float32) / 65535.0 * 255.0
    else:
        gray = gray.astype(np.float32)
        if gray.max() <= 1.0:
            gray *= 255.0
    return np.repeat(np.clip(gray, 0, 255)[:, :, None], 3, axis=2).astype(np.float32)


def load_external_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32)


def iou_matrix(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[:, :, 0] * size[:, :, 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-6)


def target_score(predictor, image, target_box):
    instances = predictor(image)["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy().astype(np.float32)
    scores = instances.scores.numpy().astype(np.float32)
    if not len(boxes):
        return 0.0, 0.0
    overlaps = iou_matrix(np.asarray([target_box], np.float32), boxes)[0]
    candidates = np.where(overlaps >= GATE["match_iou"])[0]
    if not len(candidates):
        return 0.0, float(overlaps.max())
    chosen = candidates[np.argmax(scores[candidates])]
    return float(scores[chosen]), float(overlaps[chosen])


def cfg_for(weights):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.02
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    cfg.MODEL.DEVICE = DEVICE
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    return cfg


def unlearn_records():
    coco = json.loads((UNLEARN / "annotations_coco.json").read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in coco["images"]}
    annotations = {}
    for ann in coco["annotations"]:
        annotations.setdefault(int(ann["image_id"]), []).append(ann)
    records = []
    for image_id in sorted(annotations):
        image_meta = images[image_id]
        ann = annotations[image_id][0]
        x, y, width, height = map(float, ann["bbox"])
        path = UNLEARN / image_meta["file_name"]
        if not path.exists():
            path = UNLEARN / f"{image_id}.png"
        records.append({"image_id": image_id, "path": path, "box": [x, y, x + width, y + height]})
    if len(records) != 20:
        raise AssertionError(f"Expected 20 public poison records, found {len(records)}")
    return records


def external_records(limit=160):
    split = next((EXT / name for name in ("valid", "val", "test", "train") if (EXT / name / "images").is_dir()), None)
    if split is None:
        raise AssertionError("No public external split found")
    records = []
    for path in sorted((split / "images").glob("*")):
        label = split / "labels" / f"{path.stem}.txt"
        if not label.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        for line in label.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if len(tokens) < 5:
                continue
            _, xc, yc, bw, bh = map(float, tokens[:5])
            box = [(xc - bw / 2) * width, (yc - bh / 2) * height,
                   (xc + bw / 2) * width, (yc + bh / 2) * height]
            if box[2] > box[0] + 1 and box[3] > box[1] + 1:
                records.append({"path": path, "box": box})
                break
        if len(records) >= limit:
            break
    if len(records) != limit:
        raise AssertionError(f"Only {len(records)} external controls found")
    return records


def inpaint_public_background(record):
    image = load_comp_image(record["path"]).astype(np.uint8)
    x1, y1, x2, y2 = map(int, record["box"])
    mask = np.zeros(image.shape[:2], np.uint8)
    pad = 10
    mask[max(0, y1 - pad):min(mask.shape[0], y2 + pad), max(0, x1 - pad):min(mask.shape[1], x2 + pad)] = 255
    return cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA).astype(np.float32)


def transplant_external_crop(background, record, index):
    """Paste only positive real-streak residuals onto a public clean background."""
    rng = np.random.default_rng(SEED + 900000 + index)
    source = load_external_image(record["path"])
    x1, y1, x2, y2 = map(int, record["box"])
    pad_x = max(4, int((x2 - x1) * 0.15))
    pad_y = max(4, int((y2 - y1) * 0.15))
    crop = source[max(0, y1 - pad_y):min(source.shape[0], y2 + pad_y),
                  max(0, x1 - pad_x):min(source.shape[1], x2 + pad_x)]
    if crop.size == 0:
        raise AssertionError(record)
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    baseline = float(np.percentile(gray, 45))
    residual = np.clip(gray - baseline, 0, None)
    residual = cv2.GaussianBlur(residual, (0, 0), 0.8)
    if residual.max() > 0:
        residual /= residual.max()
    scale = float(rng.uniform(0.75, 1.45))
    new_w = int(np.clip(round(crop.shape[1] * scale), 12, background.shape[1] // 2))
    new_h = int(np.clip(round(crop.shape[0] * scale), 12, background.shape[0] // 2))
    residual = cv2.resize(residual, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    bg_gray = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    noise = 1.4826 * float(np.median(np.abs(bg_gray - np.median(bg_gray)))) + 1.0
    amplitude = float(rng.uniform(10.0, 28.0)) * noise
    left = int(rng.integers(8, max(9, background.shape[1] - new_w - 8)))
    top = int(rng.integers(8, max(9, background.shape[0] - new_h - 8)))
    image = background.copy().astype(np.float32)
    image[top:top + new_h, left:left + new_w] = np.clip(
        image[top:top + new_h, left:left + new_w]
        + residual[:, :, None] * amplitude,
        0,
        255,
    )
    return image, [float(left), float(top), float(left + new_w), float(top + new_h)]


def synthesize(background, family, index):
    rng = np.random.default_rng(SEED + 1000 * list(SYNTHETIC_FAMILIES).index(family) + index)
    image = background.copy().astype(np.float32)
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    median = float(np.median(gray))
    noise = 1.4826 * float(np.median(np.abs(gray - median))) + 1.0
    height, width = gray.shape
    length = float(rng.uniform(*SYNTHETIC_FAMILIES[family]["length"]))
    sigma = float(rng.uniform(*SYNTHETIC_FAMILIES[family]["width"]))
    angle = float(rng.uniform(-math.pi, math.pi))
    margin = int(length / 2 + 18)
    cx = float(rng.uniform(margin, max(margin + 1, width - margin)))
    cy = float(rng.uniform(margin, max(margin + 1, height - margin)))
    amplitude = float(rng.uniform(*SYNTHETIC_FAMILIES[family]["snr"])) * noise
    samples = max(80, int(length * 1.5))
    t = np.linspace(-1.0, 1.0, samples, dtype=np.float32)
    tangent = np.asarray([math.cos(angle), math.sin(angle)], np.float32)
    normal = np.asarray([-math.sin(angle), math.cos(angle)], np.float32)
    center = np.asarray([cx, cy], np.float32)
    points = center[None] + t[:, None] * (length / 2) * tangent[None]
    weights = np.ones(samples, np.float32)
    if family == "synthetic_irregular_dash":
        phase = 0.0
        weights[:] = 0
        while phase < 1.0:
            on = float(rng.uniform(*SYNTHETIC_FAMILIES[family]["dash_on"]))
            off = float(rng.uniform(*SYNTHETIC_FAMILIES[family]["dash_off"]))
            lo = int(phase * samples)
            hi = min(samples, int((phase + on) * samples))
            weights[lo:hi] = 1.0
            phase += on + off
    elif family == "synthetic_stochastic_blink":
        count = int(rng.integers(*SYNTHETIC_FAMILIES[family]["segment_count"]))
        duty = float(rng.uniform(*SYNTHETIC_FAMILIES[family]["duty_cycle"]))
        weights[:] = 0
        edges = np.sort(rng.uniform(0.0, 1.0, count + 1))
        edges[0], edges[-1] = 0.0, 1.0
        for segment in range(count):
            if rng.random() <= duty:
                lo = int(edges[segment] * samples)
                hi = max(lo + 1, int(edges[segment + 1] * samples))
                weights[lo:hi] = float(rng.uniform(0.55, 1.0))
    else:  # independently generated head-tail tracklet
        count = int(rng.integers(*SYNTHETIC_FAMILIES[family]["segment_count"]))
        taper = float(rng.uniform(*SYNTHETIC_FAMILIES[family]["longitudinal_taper"]))
        weights[:] = 0
        centers = np.linspace(0.06, 0.94, count)
        for segment, center_fraction in enumerate(centers):
            half = float(rng.uniform(0.025, 0.065))
            lo = max(0, int((center_fraction - half) * samples))
            hi = min(samples, int((center_fraction + half) * samples))
            longitudinal = (1.0 - taper) + taper * (segment + 1) / count
            weights[lo:hi] = longitudinal

    impulse = np.zeros((height, width), np.float32)
    for point, weight in zip(points, weights):
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height:
            impulse[y, x] += weight
    streak = cv2.GaussianBlur(impulse, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if streak.max() > 0:
        streak /= streak.max()
    # Shot-noise term is generated independently of the target/test set.
    shot = rng.normal(0.0, np.sqrt(np.maximum(streak * amplitude, 0.0)) * 0.18, streak.shape)
    rendered = np.clip(image + (streak * amplitude + shot)[:, :, None], 0, 255)
    pad = max(8.0, 4.0 * sigma)
    box = [float(points[:, 0].min() - pad), float(points[:, 1].min() - pad),
           float(points[:, 0].max() + pad), float(points[:, 1].max() + pad)]
    box = [max(0.0, box[0]), max(0.0, box[1]), min(float(width), box[2]), min(float(height), box[3])]
    return rendered.astype(np.float32), box


poison_source = unlearn_records()
backgrounds = [inpaint_public_background(record) for record in poison_source]
external_pool = external_records(GATE["clean_pool_size"])

# Freeze every candidate before the original model is loaded. Detection-based
# retention is declared in LOCK and uses only known public/synthetic boxes.
pool_samples, pool_images = [], []
for index, record in enumerate(external_pool):
    image, box = transplant_external_crop(backgrounds[index % len(backgrounds)], record, index)
    pool_samples.append({"sample_id": f"external_crop_{index:03d}", "family": "external_crop_transplant", "label_poison": 0, "target_box": box})
    pool_images.append(image)
for family in SYNTHETIC_FAMILIES:
    for index in range(GATE["clean_pool_size"]):
        image, box = synthesize(backgrounds[index % len(backgrounds)], family, index)
        pool_samples.append({"sample_id": f"{family}_{index:03d}", "family": family, "label_poison": 0, "target_box": box})
        pool_images.append(image)
pool_frame = pd.DataFrame(pool_samples)
assert len(pool_frame) == GATE["clean_pool_size"] * len(CLEAN_FAMILIES)
log("CONTROL_POOL_FROZEN", rows=len(pool_frame), families=pool_frame.family.value_counts().to_dict())

# Run the predeclared detection-conditioned selection using public data only.
poison_samples = [
    {"sample_id": f"poison_{index:02d}", "family": "public_poison", "label_poison": 1, "target_box": record["box"]}
    for index, record in enumerate(poison_source)
]
poison_images = [load_comp_image(record["path"]) for record in poison_source]
original_predictor = DefaultPredictor(cfg_for(CHECKPOINTS["original"]))
with Heartbeat("original_control_pool_selection"):
    poison_original = [target_score(original_predictor, image, sample["target_box"])
                       for image, sample in tqdm(zip(poison_images, poison_samples), total=len(poison_samples), desc="original poison")]
    pool_original = [target_score(original_predictor, image, sample["target_box"])
                     for image, sample in tqdm(zip(pool_images, pool_samples), total=len(pool_samples), desc="original clean pool")]
del original_predictor
gc.collect()
torch.cuda.empty_cache()

pool_frame["score_original"] = [value[0] for value in pool_original]
pool_frame["iou_original"] = [value[1] for value in pool_original]
pool_frame["pool_accepted"] = (
    (pool_frame.score_original >= GATE["original_hit_score"])
    & (pool_frame.iou_original >= GATE["match_iou"])
)
selected_pool_indices = []
pool_audit = {}
for family in CLEAN_FAMILIES:
    family_rows = pool_frame[pool_frame.family == family]
    accepted = family_rows[family_rows.pool_accepted]
    chosen = accepted.head(GATE["clean_selected_per_family"])
    selected_pool_indices.extend(chosen.index.tolist())
    pool_audit[family] = {
        "pool": int(len(family_rows)),
        "accepted": int(len(accepted)),
        "acceptance_rate": float(len(accepted) / max(len(family_rows), 1)),
        "selected": int(len(chosen)),
    }
selected_pool = pool_frame.loc[selected_pool_indices].copy()
selected_pool_images = [pool_images[index] for index in selected_pool_indices]

frame = pd.concat([pd.DataFrame(poison_samples), selected_pool[pool_frame.columns[:4]]], ignore_index=True)
images = [*poison_images, *selected_pool_images]
frame["score_original"] = [value[0] for value in poison_original] + selected_pool.score_original.tolist()
frame["iou_original"] = [value[1] for value in poison_original] + selected_pool.iou_original.tolist()
assert len(frame) == len(images)
(OUT / "control_pool_audit.json").write_text(json.dumps(pool_audit, indent=2), encoding="utf-8")
(OUT / "sample_manifest.json").write_text(
    json.dumps({
        "samples": frame[["sample_id", "family", "label_poison", "target_box"]].to_dict("records"),
        "control_pool": pool_audit,
        "selection_rule": LOCK["control_selection"],
        "competition_test_read": False,
    }, indent=2),
    encoding="utf-8",
)
log("SAMPLES_SELECTED", rows=len(frame), families=frame.family.value_counts().to_dict(), pool_audit=pool_audit)


def infer_checkpoint(name, path):
    predictor = DefaultPredictor(cfg_for(path))
    scores, ious = [], []
    with Heartbeat(f"inference_{name}"):
        for image, box in tqdm(zip(images, frame.target_box), total=len(frame), desc=name):
            score, overlap = target_score(predictor, image, box)
            scores.append(score)
            ious.append(overlap)
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    return np.asarray(scores, np.float32), np.asarray(ious, np.float32)


for name, path in CHECKPOINTS.items():
    if name == "original":
        continue
    scores, ious = infer_checkpoint(name, path)
    frame[f"score_{name}"] = scores
    frame[f"iou_{name}"] = ious
    log("CHECKPOINT_COMPLETE", checkpoint=name, mean_score=float(scores.mean()), fire_rate=float(np.mean(scores >= GATE["original_hit_score"])))

original = np.maximum(frame.score_original.to_numpy(float), 1e-6)
for name in EXPECTED_HASHES:
    if name != "original":
        frame[f"ratio_{name}"] = np.clip(frame[f"score_{name}"].to_numpy(float) / original, 0.0, 2.5)
frame["ratio_v10_mean"] = frame[["ratio_v10_seed42", "ratio_v10_seed1337", "ratio_v10_seed2026"]].mean(axis=1)
frame["ratio_pcgrad_median"] = frame[["ratio_pcgrad_025", "ratio_pcgrad_05", "ratio_pcgrad_1", "ratio_pcgrad_2"]].median(axis=1)

RATIO_COLUMNS = [
    "ratio_v1", "ratio_ndr229", "ratio_v10_mean", "ratio_v12",
    "ratio_pcgrad_median", "ratio_v9_adversarial",
]
ratio_matrix = frame[RATIO_COLUMNS].to_numpy(float)
frame["survivor_min_ratio"] = np.min(ratio_matrix, axis=1)
frame["survivor_median_ratio"] = np.median(ratio_matrix, axis=1)
frame["survivor_ratio_std"] = np.std(ratio_matrix, axis=1)
frame["collapse_votes_035"] = np.sum(ratio_matrix < 0.35, axis=1)
frame["collapse_votes_050"] = np.sum(ratio_matrix < 0.50, axis=1)
frame["collapse_score"] = 1.0 - np.clip(frame.survivor_median_ratio, 0.0, 1.0)
frame["original_hit"] = frame.score_original >= GATE["original_hit_score"]


def auc_binary(labels, scores):
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = 0.0
    for value in positive:
        wins += float(np.sum(value > negative)) + 0.5 * float(np.sum(value == negative))
    return wins / (len(positive) * len(negative))


valid = frame[frame.original_hit].copy()
poison_valid = valid[valid.label_poison == 1]
clean_valid = valid[valid.label_poison == 0]
family_hit_rates = frame.groupby("family").original_hit.mean().to_dict()
family_aucs = {}
for family in CLEAN_FAMILIES:
    subset = pd.concat([poison_valid, clean_valid[clean_valid.family == family]], ignore_index=True)
    family_aucs[family] = auc_binary(subset.label_poison, subset.collapse_score)
probe_aucs = {
    column: auc_binary(valid.label_poison, 1.0 - np.clip(valid[column], 0.0, 1.0))
    for column in RATIO_COLUMNS
}
aggregate_auc = auc_binary(valid.label_poison, valid.collapse_score)
poison_median = float(poison_valid.collapse_score.median())
clean_median = float(clean_valid.collapse_score.median())
directional_count = int(sum(value >= GATE["directional_probe_auc_min"] for value in probe_aucs.values()))
clean_hits_pass = all(family_hit_rates.get(family, 0.0) >= GATE["clean_family_original_hit_rate_min"] for family in CLEAN_FAMILIES)
pool_acceptance_pass = all(
    pool_audit[family]["selected"] == GATE["clean_selected_per_family"]
    and pool_audit[family]["acceptance_rate"] >= GATE["clean_pool_acceptance_rate_min"]
    for family in CLEAN_FAMILIES
)
gate_passed = bool(
    len(poison_valid) >= 18
    and clean_hits_pass
    and pool_acceptance_pass
    and aggregate_auc >= GATE["aggregate_collapse_auc_min"]
    and all(value >= GATE["each_clean_family_auc_min"] for value in family_aucs.values())
    and poison_median - clean_median >= GATE["poison_minus_clean_median_collapse_min"]
    and directional_count >= GATE["directional_probe_count_min"]
)

frame.to_csv(OUT / "public_probe_table.csv", index=False)
correlation = valid[RATIO_COLUMNS].corr().to_dict()
(OUT / "probe_correlation.json").write_text(json.dumps(correlation, indent=2), encoding="utf-8")
audit = {
    "status": "pass" if gate_passed else "rejected",
    "gate_passed": gate_passed,
    "rows": len(frame),
    "valid_original_hits": len(valid),
    "valid_public_poison": len(poison_valid),
    "valid_clean": len(clean_valid),
    "family_original_hit_rates": family_hit_rates,
    "control_pool": pool_audit,
    "control_pool_gate_passed": pool_acceptance_pass,
    "aggregate_collapse_auc": aggregate_auc,
    "family_collapse_auc": family_aucs,
    "probe_collapse_auc": probe_aucs,
    "directional_probe_count": directional_count,
    "poison_median_collapse": poison_median,
    "clean_median_collapse": clean_median,
    "median_gap": poison_median - clean_median,
    "frozen_requirements": GATE,
    "rule_7a_guard_passed": True,
    "competition_test_enumerated": False,
    "competition_test_read": False,
    "competition_submission_created": False,
}
(OUT / "survivor_gate.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

fig, axis = plt.subplots(figsize=(11, 6))
families = ["public_poison", *CLEAN_FAMILIES]
data = [valid.loc[valid.family == family, "collapse_score"].to_numpy() for family in families]
axis.boxplot(data, labels=families, showmeans=True)
axis.axhline(clean_median, color="#47c2ff", linestyle="--", linewidth=1, label="clean median")
axis.set_ylabel("1 - median survivor ratio (higher = more collapse)")
axis.set_title(f"Public-only survivor gate: {'PASS' if gate_passed else 'REJECT'} | AUC={aggregate_auc:.3f}")
axis.tick_params(axis="x", rotation=20)
axis.legend()
fig.tight_layout()
fig.savefig(OUT / "survivor_gate.png", dpi=160)
plt.close(fig)

report = {
    "status": "complete",
    "step": "3B",
    "decision": "continue_to_trace" if gate_passed else "stop_survivor_ranker_branch",
    "audit": audit,
    "checkpoint_count": len(CHECKPOINTS),
    "sample_count": len(frame),
    "rule_7a_guard_passed": True,
    "competition_test_read": False,
    "competition_submission_created": False,
}
(OUT / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log("COMPLETE", report=report)
print(json.dumps(report, indent=2))

# %% [markdown]
# ## Step 4B - background and focal consistency
#
# The global TRACE sign is selected once from public poison versus all public
# clean controls. The chosen sign must then pass independently for every clean
# family. No test image, test candidate or leaderboard feedback is available.

# %%
def extract_positive_residual(image, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return np.zeros((1, 1), np.float32)
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    baseline = float(np.median(border))
    residual = np.clip(gray - baseline, 0, None)
    residual = cv2.GaussianBlur(residual, (0, 0), 0.65)
    return residual


def paste_residual(background, residual, seed):
    rng = np.random.default_rng(seed)
    image = background.copy().astype(np.float32)
    height, width = residual.shape
    if height >= image.shape[0] - 16 or width >= image.shape[1] - 16:
        scale = min((image.shape[0] - 20) / max(height, 1), (image.shape[1] - 20) / max(width, 1))
        width = max(2, int(width * scale))
        height = max(2, int(height * scale))
        residual = cv2.resize(residual, (width, height), interpolation=cv2.INTER_LINEAR)
    left = int(rng.integers(8, max(9, image.shape[1] - width - 8)))
    top = int(rng.integers(8, max(9, image.shape[0] - height - 8)))
    image[top:top + height, left:left + width] = np.clip(
        image[top:top + height, left:left + width] + residual[:, :, None], 0, 255
    )
    return image, [float(left), float(top), float(left + width), float(top + height)]


def focal_variants(image, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    variants = []
    crop = image[y1:y2, x1:x2].astype(np.uint8)
    if crop.size == 0:
        return [image.copy() for _ in range(TRACE_GATE["focal_transforms"])]

    blurred = image.copy()
    blurred[y1:y2, x1:x2] = cv2.GaussianBlur(crop, (0, 0), 2.0)
    variants.append(blurred)

    smooth = image.copy()
    smooth[y1:y2, x1:x2] = cv2.GaussianBlur(crop, (0, 0), 1.1)
    variants.append(smooth)

    for mode in ("center", "left", "cross"):
        mask = np.zeros(image.shape[:2], np.uint8)
        width, height = max(1, x2 - x1), max(1, y2 - y1)
        if mode == "center":
            mask[y1 + height // 3:y1 + 2 * height // 3, x1 + width // 3:x1 + 2 * width // 3] = 255
        elif mode == "left":
            mask[y1:y2, x1:x1 + max(1, width // 4)] = 255
        else:
            mask[y1 + max(0, height // 2 - 2):y1 + min(height, height // 2 + 3), x1:x2] = 255
            mask[y1:y2, x1 + max(0, width // 2 - 2):x1 + min(width, width // 2 + 3)] = 255
        variants.append(cv2.inpaint(image.astype(np.uint8), mask, 5, cv2.INPAINT_TELEA).astype(np.float32))
    return variants


def binary_entropy(probability):
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return float(-probability * math.log(probability) - (1 - probability) * math.log(1 - probability))


trace_predictor = DefaultPredictor(cfg_for(CHECKPOINTS["original"]))
trace_rows = []
with Heartbeat("trace_public_controls"):
    for sample_index, (sample, image) in enumerate(tqdm(zip(frame.itertuples(index=False), images), total=len(frame), desc="TRACE")):
        residual = extract_positive_residual(image, sample.target_box)
        context_scores = []
        for background_index in range(TRACE_GATE["backgrounds_per_sample"]):
            background = backgrounds[(sample_index * 7 + background_index * 3) % len(backgrounds)]
            transplanted, destination = paste_residual(
                background, residual, SEED + sample_index * 1009 + background_index
            )
            score, _ = target_score(trace_predictor, transplanted, destination)
            context_scores.append(score / max(float(sample.score_original), 1e-6))
        focal_scores = []
        for transformed in focal_variants(image, sample.target_box):
            score, _ = target_score(trace_predictor, transformed, sample.target_box)
            focal_scores.append(score / max(float(sample.score_original), 1e-6))
        context_scores = np.clip(np.asarray(context_scores, float), 0, 2.5)
        focal_scores = np.clip(np.asarray(focal_scores, float), 0, 2.5)
        fire_rate = float(np.mean(context_scores * float(sample.score_original) >= GATE["original_hit_score"]))
        trace_rows.append({
            "sample_id": sample.sample_id,
            "family": sample.family,
            "label_poison": int(sample.label_poison),
            "original_score": float(sample.score_original),
            "context_mean": float(context_scores.mean()),
            "context_std": float(context_scores.std()),
            "context_fire_rate": fire_rate,
            "context_fire_entropy": binary_entropy(fire_rate),
            "focal_mean": float(focal_scores.mean()),
            "focal_std": float(focal_scores.std()),
            "trace_raw": float(context_scores.mean() - focal_scores.mean()),
        })
del trace_predictor
gc.collect()
torch.cuda.empty_cache()

trace_frame = pd.DataFrame(trace_rows)
raw_auc = auc_binary(trace_frame.label_poison, trace_frame.trace_raw)
inverse_auc = auc_binary(trace_frame.label_poison, -trace_frame.trace_raw)
trace_sign = 1.0 if raw_auc >= inverse_auc else -1.0
trace_frame["trace_score"] = trace_sign * trace_frame.trace_raw
trace_auc = auc_binary(trace_frame.label_poison, trace_frame.trace_score)
trace_family_auc = {}
for family in CLEAN_FAMILIES:
    subset = trace_frame[(trace_frame.family == "public_poison") | (trace_frame.family == family)]
    trace_family_auc[family] = auc_binary(subset.label_poison, subset.trace_score)
median_context_fire = float(trace_frame.context_fire_rate.median())
trace_gate_passed = bool(
    gate_passed
    and trace_auc >= TRACE_GATE["aggregate_auc_min"]
    and all(value >= TRACE_GATE["each_clean_family_auc_min"] for value in trace_family_auc.values())
    and median_context_fire >= TRACE_GATE["minimum_context_fire_rate"]
)
trace_frame.to_csv(OUT / "public_trace_table.csv", index=False)
trace_audit = {
    "status": "pass" if trace_gate_passed else "rejected",
    "gate_passed": trace_gate_passed,
    "rows": len(trace_frame),
    "sign_selected_from_public_data": trace_sign,
    "raw_auc": raw_auc,
    "inverse_auc": inverse_auc,
    "aggregate_auc": trace_auc,
    "family_auc": trace_family_auc,
    "median_context_fire_rate": median_context_fire,
    "frozen_requirements": TRACE_GATE,
    "step3b_gate_reproduced": gate_passed,
    "rule_7a_guard_passed": True,
    "competition_test_enumerated": False,
    "competition_test_read": False,
    "competition_submission_created": False,
}
(OUT / "trace_gate.json").write_text(json.dumps(trace_audit, indent=2), encoding="utf-8")

fig, axis = plt.subplots(figsize=(11, 6))
trace_data = [trace_frame.loc[trace_frame.family == family, "trace_score"].to_numpy() for family in families]
axis.boxplot(trace_data, labels=families, showmeans=True)
axis.set_ylabel("signed TRACE score")
axis.set_title(f"Step 4B TRACE: {'PASS' if trace_gate_passed else 'REJECT'} | AUC={trace_auc:.3f}")
axis.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(OUT / "trace_gate.png", dpi=160)
plt.close(fig)

combined_report = {
    "status": "complete",
    "step": "4B",
    "decision": "continue_to_ranker_features" if trace_gate_passed else "stop_trace_branch",
    "step3b": audit,
    "trace": trace_audit,
    "rule_7a_guard_passed": True,
    "competition_test_read": False,
    "competition_submission_created": False,
}
(OUT / "final_report.json").write_text(json.dumps(combined_report, indent=2), encoding="utf-8")
log("STEP4B_COMPLETE", report=combined_report)
print(json.dumps(combined_report, indent=2))
