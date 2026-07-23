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
# # Step 8B - renderer recalibration on validated clean controls
#
# This separately frozen public-only episode first reproduces Step 3B, then
# evaluates orientation-normalized renderer fingerprints on the exact same 20
# poison and 120 accepted clean controls. It never enumerates or reads the
# competition test set and cannot create a candidate or submission.

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

OUT = Path("/kaggle/working/step8c_renderer_robustness")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.jsonl"
# Keep the exact Step 3B RNG stream so the selected 140-sample bank is
# byte-for-byte reproducible; Step 8B changes only the downstream feature gate.
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
RENDERER_GATE = {
    "aggregate_auc_min": 0.80,
    "each_clean_family_auc_min": 0.72,
    "median_family_recall_min": 0.20,
    "maximum_family_false_positive_rate": 0.10,
    "coefficient_cosine_median_min": 0.25,
    "clean_calibration_quantile": 0.99,
    "l2": 0.20,
    "steps": 900,
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
    "experiment": "STEP8B_RENDERER_ON_VALIDATED_PUBLIC_CONTROLS",
    "seed": SEED,
    "checkpoint_hashes": EXPECTED_HASHES,
    "ratio_probes": ["v1", "ndr229", "v10_mean", "v12", "pcgrad_median", "v9_adversarial"],
    "auxiliary_probe": "v13_task",
    "clean_families": CLEAN_FAMILIES,
    "synthetic_specs": SYNTHETIC_FAMILIES,
    "gate": GATE,
    "renderer_gate": RENDERER_GATE,
    "renderer_features": [
        "positive_fraction", "energy_density", "anisotropy", "straightness",
        "endpoint_ratio", "endpoint_asymmetry", "center_ratio", "longitudinal_cv",
        "longitudinal_total_variation", "gap_fraction", "fft_periodicity",
        "autocorrelation_peak", "gap_run_cv", "width_mean", "width_cv",
        "width_drift", "transverse_symmetry", "gaussian_psf_error",
        "inside_noise_cv", "side_positive_rate", "side_noise_std", "unique_ratio",
        "bit1_entropy", "bit2_entropy", "bit4_entropy", "bit8_entropy",
        "subpixel_phase", "endpoint_sharpness", "interpolation_fraction",
        "axis_residual", "log_length", "log_aspect"
    ],
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
STRESS_TRANSFORMS = {
    "identity": {},
    "gain_090": {"gain": 0.90},
    "gain_110": {"gain": 1.10},
    "gamma_090": {"gamma": 0.90},
    "gamma_110": {"gamma": 1.10},
    "blur_sigma065": {"blur_sigma": 0.65},
    "sensor_noise_sigma15": {"noise_sigma": 1.5},
    "downup_075": {"downup_scale": 0.75},
    "quantize_8bit": {"quantize_8bit": True},
}
ROBUSTNESS_GATE = {
    "identity_aggregate_auc_min": 0.99,
    "each_shift_aggregate_auc_min": 0.90,
    "each_shift_each_family_auc_min": 0.85,
    "each_shift_median_poison_recall_min": 0.80,
    "each_shift_maximum_family_fpr_max": 0.20,
    "each_shift_rank_correlation_min": 0.75,
    "each_shift_margin_retention_min": 0.65,
}
EXPECTED_STEP8B_MANIFEST_SHA256 = "f2906451a2b6326bf9c51f4d85f1eda175a86a2e11dadd3d32fffbfb977284b7"
EXPECTED_RENDERER_MODELS_SHA256 = "ded18b7c482e6a423335d86b3f867a3cf58d083942a9ed51fd1dedbba4e92a91"
LOCK["experiment"] = "STEP8C_FROZEN_RENDERER_ROBUSTNESS"
LOCK["parent_step"] = "Step 8B V3"
LOCK["stress_transforms"] = STRESS_TRANSFORMS
LOCK["robustness_gate"] = ROBUSTNESS_GATE
LOCK["expected_step8b_manifest_sha256"] = EXPECTED_STEP8B_MANIFEST_SHA256
LOCK["expected_renderer_models_sha256"] = EXPECTED_RENDERER_MODELS_SHA256
LOCK["training_or_retuning"] = False
LOCK["candidate_or_submission_generation"] = False
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

FEATURE_NAMES = LOCK["renderer_features"]


def bit_entropy(values, modulus):
    values = np.asarray(np.rint(values), np.int64).ravel() % modulus
    histogram = np.bincount(values, minlength=modulus).astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    histogram = histogram[histogram > 0]
    return float(-(histogram * np.log(histogram)).sum() / math.log(modulus))


def run_length_cv(mask):
    lengths, current = [], 0
    for value in np.asarray(mask, bool):
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    if len(lengths) < 2:
        return 0.0
    lengths = np.asarray(lengths, float)
    return float(lengths.std() / (lengths.mean() + 1e-6))


def renderer_features(image, box):
    gray = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    x1, y1, x2, y2 = map(float, box)
    pad = max(8, int(round(0.16 * max(x2 - x1, y2 - y1))))
    left, right = max(0, int(math.floor(x1)) - pad), min(width, int(math.ceil(x2)) + pad)
    top, bottom = max(0, int(math.floor(y1)) - pad), min(height, int(math.ceil(y2)) + pad)
    crop = gray[top:bottom, left:right]
    if min(crop.shape, default=0) < 5:
        return np.zeros(len(FEATURE_NAMES), np.float32)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    baseline = float(np.median(border))
    noise = 1.4826 * float(np.median(np.abs(border - baseline))) + 0.75
    z = np.clip((crop - baseline) / noise, -5.0, 40.0)
    positive = np.clip(z, 0.0, None)
    active = positive > 2.5
    yy, xx = np.indices(crop.shape)
    weights = np.clip(positive - 1.5, 0.0, None)
    total = float(weights.sum()) + 1e-6
    mx, my = float((xx * weights).sum() / total), float((yy * weights).sum() / total)
    dx, dy = xx - mx, yy - my
    covariance = np.asarray([
        [(weights * dx * dx).sum() / total, (weights * dx * dy).sum() / total],
        [(weights * dx * dy).sum() / total, (weights * dy * dy).sum() / total],
    ], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    rotation = cv2.getRotationMatrix2D((mx, my), -angle, 1.0)
    rotated = cv2.warpAffine(positive, rotation, (positive.shape[1], positive.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rotated_active = rotated > 2.5
    rows, columns = np.where(rotated_active.any(1))[0], np.where(rotated_active.any(0))[0]
    if not len(rows) or not len(columns):
        return np.zeros(len(FEATURE_NAMES), np.float32)
    streak = rotated[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
    streak_active = streak > 2.5
    longitudinal, transverse = streak.sum(0), streak.sum(1)
    longitudinal_norm = longitudinal / (longitudinal.mean() + 1e-6)
    transverse_norm = transverse / (transverse.max() + 1e-6)
    q = max(1, len(longitudinal) // 8)
    endpoint_left, endpoint_right = float(longitudinal[:q].mean()), float(longitudinal[-q:].mean())
    endpoint_ratio = float((endpoint_left + endpoint_right) / (2 * longitudinal.mean() + 1e-6))
    endpoint_asymmetry = float(abs(endpoint_left - endpoint_right) / (longitudinal.mean() + 1e-6))
    center = longitudinal[len(longitudinal) // 3:max(len(longitudinal) // 3 + 1, 2 * len(longitudinal) // 3)]
    center_ratio = float(center.mean() / (longitudinal.mean() + 1e-6))
    longitudinal_cv = float(longitudinal.std() / (longitudinal.mean() + 1e-6))
    longitudinal_tv = float(np.mean(np.abs(np.diff(longitudinal_norm)))) if len(longitudinal) > 1 else 0.0
    gap_mask = longitudinal < 0.25 * longitudinal.mean()
    spectrum = np.abs(np.fft.rfft(longitudinal - longitudinal.mean()))
    fft_periodicity = float(spectrum[1:].max() / (spectrum[1:].sum() + 1e-6)) if len(spectrum) > 1 else 0.0
    autocorrelation = np.correlate(longitudinal - longitudinal.mean(), longitudinal - longitudinal.mean(), mode="full")[len(longitudinal) - 1:]
    autocorrelation /= float(autocorrelation[0]) + 1e-6
    autocorrelation_peak = float(autocorrelation[2:max(3, len(autocorrelation) // 2)].max()) if len(autocorrelation) > 5 else 0.0
    widths = streak_active.sum(0).astype(float)
    nonzero = widths > 0
    width_mean = float(widths[nonzero].mean()) if nonzero.any() else 0.0
    width_cv = float(widths[nonzero].std() / (width_mean + 1e-6)) if nonzero.any() else 0.0
    if nonzero.sum() >= 3:
        width_drift = float(abs(np.polyfit(np.linspace(-1, 1, len(widths))[nonzero], widths[nonzero], 1)[0]) / (width_mean + 1e-6))
    else:
        width_drift = 0.0
    transverse_symmetry = float(np.mean(np.abs(transverse_norm - transverse_norm[::-1])))
    coordinate, valid_profile = np.linspace(-1, 1, len(transverse_norm)), transverse_norm > 0.05
    if valid_profile.sum() >= 3:
        coefficient = np.polyfit(coordinate[valid_profile] ** 2, np.log(transverse_norm[valid_profile] + 1e-4), 1)
        gaussian = np.exp(np.polyval(coefficient, coordinate ** 2)); gaussian /= gaussian.max() + 1e-6
        gaussian_error = float(np.mean(np.abs(transverse_norm - gaussian)))
    else:
        gaussian_error = 1.0
    active_values = streak[streak_active]
    inside_noise_cv = float(active_values.std() / (active_values.mean() + 1e-6)) if len(active_values) else 0.0
    dilated = cv2.dilate(active.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    side = z[~dilated]
    integer_crop = np.clip(np.rint(crop), 0, 255).astype(np.uint8)
    major_value, minor_value = float(max(eigenvalues.max(), 1e-6)), float(max(eigenvalues.min(), 0.0))
    length, short = max(float(x2 - x1), float(y2 - y1)), max(min(float(x2 - x1), float(y2 - y1)), 1e-3)
    values = [
        float(active.mean()), float(weights.sum() / max(weights.size, 1)),
        float(math.log1p(major_value / (minor_value + 1e-3))), float(math.sqrt(minor_value / major_value)),
        endpoint_ratio, endpoint_asymmetry, center_ratio, longitudinal_cv, longitudinal_tv,
        float(gap_mask.mean()), fft_periodicity, autocorrelation_peak, run_length_cv(gap_mask),
        width_mean, width_cv, width_drift, transverse_symmetry, gaussian_error, inside_noise_cv,
        float(np.mean(side > 2.5)) if len(side) else 0.0, float(side.std()) if len(side) else 0.0,
        float(len(np.unique(integer_crop)) / max(integer_crop.size, 1)), bit_entropy(integer_crop, 2),
        bit_entropy(integer_crop, 4), bit_entropy(integer_crop, 16), bit_entropy(integer_crop, 256),
        float((x1 % 1 + y1 % 1 + x2 % 1 + y2 % 1) / 4),
        float((np.abs(np.diff(longitudinal_norm[:q + 1])).mean() + np.abs(np.diff(longitudinal_norm[-q - 1:])).mean()) / 2),
        float(np.mean(np.abs(crop - np.rint(crop)) > 1e-4)),
        float(np.sqrt(minor_value) / (np.sqrt(major_value) + 1e-6)), math.log1p(length), math.log1p(length / short),
    ]
    result = np.nan_to_num(np.asarray(values, np.float32), nan=0, posinf=20, neginf=-20)
    assert len(result) == len(FEATURE_NAMES)
    return np.clip(result, -20, 20)

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
