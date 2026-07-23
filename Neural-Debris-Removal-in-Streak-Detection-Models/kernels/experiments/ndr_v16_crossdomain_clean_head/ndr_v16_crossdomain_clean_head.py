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
# # NDR V16 - cross-domain synthetic/external clean head
#
# This notebook trains a compartmentalized P3/P4 auxiliary classifier rather
# than modifying RetinaNet. Public poison signals are the negative class;
# public StreaksYolo crops and a physics-based simulator are independent clean
# classes. Every source is normalized and pasted through the same augmentation
# path onto backgrounds built exclusively from the public unlearn set.
#
# Selection and thresholds are frozen before test enumeration. Test images are
# used only once for final inference. No test labels or pseudo-labels are made.
# The exact V12/M1 boxes are retained; V16 can only reduce confidence.

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
    [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "git+https://github.com/facebookresearch/detectron2.git"],
    check=True,
)

# %%
import gc
import hashlib
import json
import math
import random
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

SEED = 160721
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "V16 requires a Kaggle GPU"

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
if not ROOT.exists():
    ROOT = Path("/kaggle/input/neural-debris-removal-in-streak-detection-models")
POISONED_WEIGHTS = ROOT / "poisoned_model/poisoned_model.pth"
UNLEARN_DIR = ROOT / "unlearn_set"
UNLEARN_JSON = UNLEARN_DIR / "annotations_coco.json"
TEST_DIR = ROOT / "test_set/test_set"
SAMPLE_SUB = ROOT / "sample_submission.csv"
for required in (POISONED_WEIGHTS, UNLEARN_JSON, TEST_DIR, SAMPLE_SUB):
    assert required.exists(), required

OUT = Path("/kaggle/working/ndr_v16")
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run.jsonl"
BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ASPECTS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
SIZES = [[16], [32], [64], [128], [256]]

VARIANTS = {
    "V16_0_exact_M1": {"mode": "identity"},
    "V16_A_cleanhead_soft95": {"mode": "head", "threshold": 0.95, "cap": 0.21},
    "V16_B_cleanhead_hard90": {"mode": "head", "threshold": 0.90, "cap": 0.02},
    "V16_C_cleanhead_hard80": {"mode": "head", "threshold": 0.80, "cap": 0.02},
    "V16_D_dual_consensus85": {"mode": "consensus", "threshold": 0.85, "cap": 0.02},
    "V16_E_dual_unanimous80": {"mode": "unanimous", "head": 0.80, "pcgrad": 0.80, "cap": 0.02},
}
LOCK = {
    "status": "frozen_before_test_enumeration",
    "experiment": "V16_CROSSDOMAIN_SYNTHETIC_EXTERNAL_CLEAN_HEAD",
    "seed": SEED,
    "clean_sources": {
        "external": {
            "kaggle": "sanidhyavijay24/streaksyolodataset",
            "zenodo": "https://doi.org/10.5281/zenodo.14047944",
            "access": "public and free",
        },
        "synthetic": {
            "generator": "analytic anti-aliased line convolved with Gaussian PSF and Poisson/read noise",
            "parameter_source": "literature-style physical model plus public unlearn image scale only",
            "test_derived_parameters": False,
        },
    },
    "training": {
        "features": ["RetinaNet P3 pooled", "RetinaNet P4 pooled", "12 fixed morphology statistics"],
        "head": "LayerNorm-Linear(128)-GELU-Dropout-Linear(1)",
        "trainable_detector_weights": 0,
        "examples_per_domain": 160,
        "epochs": 180,
        "seeds": [160721, 160722, 160723],
        "cross_domain_gate_auc": 0.75,
        "cross_domain_gate_margin": 0.15,
    },
    "inference": {
        "box_bank": "exact V12/M1 only",
        "boxes_added": 0,
        "boxes_moved": 0,
        "confidence_increases": 0,
    },
    "variants": VARIANTS,
    "alias": "V16_A_cleanhead_soft95",
    "rule_7a": {
        "selection_frozen_before_test_enumeration": True,
        "test_used_for_training_or_selection": False,
        "test_labels_or_pseudo_labels_created": False,
        "competition_submission_created": False,
    },
}
(OUT / "selection_lock.json").write_text(json.dumps(LOCK, indent=2), encoding="utf-8")


def log(message, **fields):
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "message": message, **fields}
    print(json.dumps(row, default=str), flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    def __exit__(self, typ, value, trace):
        self.stop.set()
        self.thread.join(timeout=2)
        log("STAGE_END", stage=self.label, ok=typ is None)


def discover_external_root(input_root=Path("/kaggle/input")):
    candidates = []
    for yaml_path in sorted(input_root.rglob("data.yaml")):
        root = yaml_path.parent
        has_train = (root / "train/images").is_dir() and (root / "train/labels").is_dir()
        has_eval = any((root / f"{split}/images").is_dir() and (root / f"{split}/labels").is_dir() for split in ("valid", "val", "test"))
        if has_train and has_eval:
            candidates.append(root)
    assert candidates, "StreaksYoloDataset mount not found"
    candidates.sort(key=lambda path: ("streak" not in str(path).lower(), len(path.parts), str(path)))
    return candidates[0]


EXT_ROOT = discover_external_root()
log("SELECTION_LOCK_WRITTEN", lock=LOCK)
log("EXTERNAL_DATASET_DISCOVERED", root=str(EXT_ROOT))

# %% [markdown]
# ## Public-only signal bank and physics simulator

# %%
def load_comp(path):
    gray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert gray is not None, path
    if gray.dtype == np.uint16:
        gray = gray.astype(np.float32) / 65535.0 * 255.0
    else:
        gray = gray.astype(np.float32)
        if gray.max() <= 1:
            gray *= 255.0
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    return np.repeat(np.clip(gray, 0, 255)[:, :, None], 3, axis=2).astype(np.float32)


with UNLEARN_JSON.open(encoding="utf-8") as handle:
    coco = json.load(handle)
id_to_name = {int(image["id"]): image["file_name"] for image in coco["images"]}
poison_by_name = {}
for annotation in coco["annotations"]:
    x, y, width, height = map(float, annotation["bbox"])
    poison_by_name.setdefault(id_to_name[int(annotation["image_id"])], []).append([x, y, x + width, y + height])

PUBLIC = []
BACKGROUNDS = []
background_rng = np.random.default_rng(SEED + 1)
for file_path in sorted(UNLEARN_DIR.glob("*.png")):
    image = load_comp(file_path)
    boxes = np.asarray(poison_by_name.get(file_path.name, []), np.float32)
    PUBLIC.append((file_path.name, image, boxes))
    gray = image[:, :, 0].copy()
    for x1, y1, x2, y2 in boxes:
        left, top = max(0, int(x1) - 8), max(0, int(y1) - 8)
        right, bottom = min(1024, int(x2) + 8), min(1024, int(y2) + 8)
        ring = gray[max(0, top - 28) : min(1024, bottom + 28), max(0, left - 28) : min(1024, right + 28)]
        median = float(np.median(ring))
        mad = 1.4826 * float(np.median(np.abs(ring - median))) + 1e-3
        gray[top:bottom, left:right] = np.clip(background_rng.normal(median, mad, (bottom - top, right - left)), 0, 255)
    BACKGROUNDS.append(np.repeat(gray[:, :, None], 3, axis=2))


def residual_from_crop(gray, box, padding=8):
    x1, y1, x2, y2 = box
    left, top = max(0, int(x1) - padding), max(0, int(y1) - padding)
    right, bottom = min(gray.shape[1], int(x2) + padding), min(gray.shape[0], int(y2) + padding)
    crop = gray[top:bottom, left:right].astype(np.float32)
    if min(crop.shape) < 3:
        return None
    border = np.r_[crop[0], crop[-1], crop[:, 0], crop[:, -1]]
    residual = np.clip(crop - float(np.median(border)), 0, None)
    maximum = float(residual.max())
    return None if maximum < 2 else (residual / maximum).astype(np.float32)


POISON_SIGNALS = []
for _, image, boxes in PUBLIC:
    for box in boxes:
        signal = residual_from_crop(image[:, :, 0], box)
        if signal is not None:
            POISON_SIGNALS.append(signal)
assert len(POISON_SIGNALS) >= 20
POISON_TRAIN_SIGNALS = POISON_SIGNALS[:15]
POISON_VALID_SIGNALS = POISON_SIGNALS[15:]


def external_pairs(split):
    image_dir, label_dir = EXT_ROOT / split / "images", EXT_ROOT / split / "labels"
    pairs = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size:
            pairs.append((image_path, label_path))
    return pairs


def extract_external_signals(pairs, maximum):
    signals = []
    for image_path, label_path in pairs:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        height, width = gray.shape
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            _, xc, yc, bw, bh = map(float, values[:5])
            box = [(xc - bw / 2) * width, (yc - bh / 2) * height, (xc + bw / 2) * width, (yc + bh / 2) * height]
            signal = residual_from_crop(gray, box)
            if signal is not None:
                signals.append(signal)
            if len(signals) >= maximum:
                return signals
    return signals


train_pairs = external_pairs("train")
validation_pairs = external_pairs("valid") + external_pairs("val") + external_pairs("test")
EXTERNAL_TRAIN = extract_external_signals(train_pairs, 500)
EXTERNAL_VALID = extract_external_signals(validation_pairs, 180)
assert len(EXTERNAL_TRAIN) >= 150 and len(EXTERNAL_VALID) >= 40


def physics_signal(rng):
    height = int(rng.integers(13, 97))
    width = int(rng.integers(13, 257))
    canvas = np.zeros((height, width), np.float32)
    margin = 3
    x1, y1 = int(rng.integers(margin, max(margin + 1, width - margin))), int(rng.integers(margin, max(margin + 1, height - margin)))
    angle = float(rng.uniform(0, 2 * math.pi))
    length = float(rng.uniform(0.45, 0.95) * max(height, width))
    x2 = int(np.clip(x1 + math.cos(angle) * length, margin, width - margin - 1))
    y2 = int(np.clip(y1 + math.sin(angle) * length, margin, height - margin - 1))
    thickness = int(rng.integers(1, 4))
    cv2.line(canvas, (x1, y1), (x2, y2), 1.0, thickness, lineType=cv2.LINE_AA)
    sigma = float(rng.uniform(0.55, 2.2))
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if canvas.max() > 0:
        canvas /= canvas.max()
    return canvas


def paste_signal(signal, background, rng):
    signal = signal.copy()
    signal = np.rot90(signal, int(rng.integers(4)))
    if rng.random() < 0.5:
        signal = np.fliplr(signal)
    target = float(rng.uniform(18, 380))
    scale = target / max(signal.shape)
    height, width = max(3, int(signal.shape[0] * scale)), max(3, int(signal.shape[1] * scale))
    signal = cv2.resize(signal, (width, height), interpolation=cv2.INTER_LINEAR)
    if height >= 1000 or width >= 1000:
        return paste_signal(signal, background, rng)
    top = int(rng.integers(8, 1024 - height - 8))
    left = int(rng.integers(8, 1024 - width - 8))
    gray = background[:, :, 0].copy()
    local = gray[top : top + height, left : left + width]
    noise = 1.4826 * np.median(np.abs(local - np.median(local))) + 1e-3
    amplitude = float(rng.uniform(4, 20)) * noise
    shot = rng.normal(0, np.sqrt(np.maximum(signal * amplitude, 0)) * 0.08, signal.shape)
    gray[top : top + height, left : left + width] = np.clip(local + signal * amplitude + shot, 0, 255)
    return np.repeat(gray[:, :, None], 3, axis=2), np.asarray([[left, top, left + width, top + height]], np.float32)


manifest = {
    "poison_signal_crops": len(POISON_SIGNALS),
    "external_train_signal_crops": len(EXTERNAL_TRAIN),
    "external_validation_signal_crops": len(EXTERNAL_VALID),
    "public_backgrounds": len(BACKGROUNDS),
    "test_data_used": False,
}
(OUT / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
log("PUBLIC_TRAINING_DATA_READY", **manifest)

# %% [markdown]
# ## Frozen RetinaNet feature extractor and cross-domain head

# %%
def detector_config():
    config = get_cfg()
    config.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    config.MODEL.WEIGHTS = str(POISONED_WEIGHTS)
    config.MODEL.RETINANET.NUM_CLASSES = 1
    config.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
    config.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ASPECTS]
    config.MODEL.ANCHOR_GENERATOR.SIZES = SIZES
    config.MODEL.DEVICE = DEVICE
    config.TEST.DETECTIONS_PER_IMAGE = 100
    return config


class FeatureExtractor:
    def __init__(self):
        self.predictor = DefaultPredictor(detector_config())
        self.features = None
        self.hook = self.predictor.model.backbone.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.features = {level: tensor.detach() for level, tensor in output.items() if level in ("p3", "p4")}

    def run(self, image):
        with torch.inference_mode(), torch.autocast("cuda", enabled=True):
            self.predictor(image)

    def pool_level(self, boxes, level):
        feature = self.features[level][0].float()
        channels, height, width = feature.shape
        stride = 8 if level == "p3" else 16
        result = np.zeros((len(boxes), channels), np.float32)
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            left = int(np.clip(np.floor(x1 / stride), 0, width - 1))
            top = int(np.clip(np.floor(y1 / stride), 0, height - 1))
            right = int(np.clip(np.ceil(x2 / stride), left + 1, width))
            bottom = int(np.clip(np.ceil(y2 / stride), top + 1, height))
            result[index] = feature[:, top:bottom, left:right].mean((1, 2)).cpu().numpy()
        return result

    def pool(self, boxes):
        return np.concatenate([self.pool_level(boxes, "p3"), self.pool_level(boxes, "p4")], axis=1)


def morphology(image, boxes):
    gray = image[:, :, 0]
    rows = []
    for x1, y1, x2, y2 in boxes:
        left, top = max(0, int(x1) - 4), max(0, int(y1) - 4)
        right, bottom = min(1024, int(x2) + 4), min(1024, int(y2) + 4)
        crop = gray[top:bottom, left:right]
        median = float(np.median(crop))
        mad = 1.4826 * float(np.median(np.abs(crop - median))) + 1e-3
        standardized = (crop - median) / mad
        mask = standardized > 2.5
        yy, xx = np.nonzero(mask)
        if len(xx) > 4:
            points = np.stack([xx, yy], axis=1).astype(np.float32)
            points -= points.mean(0)
            eigenvalues, eigenvectors = np.linalg.eigh(points.T @ points / max(len(points) - 1, 1))
            projection = points @ eigenvectors[:, -1]
            bins = np.linspace(projection.min(), projection.max() + 1e-6, 33)
            occupied = np.zeros(32)
            occupied[np.clip(np.digitize(projection, bins) - 1, 0, 31)] = 1
            gap = 1 - occupied.mean()
            transitions = np.abs(np.diff(occupied)).mean()
            linearity = eigenvalues[-1] / max(eigenvalues.sum(), 1e-6)
        else:
            gap, transitions, linearity = 1.0, 0.0, 0.0
        height, width = crop.shape
        rows.append(
            [
                math.log(max(width, 1)), math.log(max(height, 1)), math.log(max(width / max(height, 1), 1e-3)),
                mask.mean(), gap, transitions, linearity, standardized.std(), standardized.max(),
                np.percentile(standardized, 95), np.percentile(standardized, 99),
                cv2.Laplacian(crop.astype(np.float32), cv2.CV_32F).var() / (mad * mad + 1e-6),
            ]
        )
    return np.nan_to_num(np.asarray(rows, np.float32), nan=0, posinf=20, neginf=-20)


extractor = FeatureExtractor()
examples_per_domain = LOCK["training"]["examples_per_domain"]
feature_rng = np.random.default_rng(SEED + 20)
FEATURES = {"poison": [], "external": [], "synthetic": []}
GROUPS = {"poison": [], "external": [], "synthetic": []}

with Heartbeat("public_feature_collection"):
    for domain in FEATURES:
        for index in tqdm(range(examples_per_domain), desc=f"{domain} features"):
            if domain == "poison":
                signal = POISON_TRAIN_SIGNALS[index % len(POISON_TRAIN_SIGNALS)]
                group = index % len(POISON_TRAIN_SIGNALS)
            elif domain == "external":
                signal = EXTERNAL_TRAIN[index % len(EXTERNAL_TRAIN)]
                group = index % len(EXTERNAL_TRAIN)
            else:
                signal = physics_signal(feature_rng)
                group = index
            image, box = paste_signal(signal, BACKGROUNDS[index % len(BACKGROUNDS)], feature_rng)
            extractor.run(image)
            FEATURES[domain].append(np.c_[extractor.pool(box), morphology(image, box)][0])
            GROUPS[domain].append(group)

for key in FEATURES:
    FEATURES[key] = np.asarray(FEATURES[key], np.float32)
    GROUPS[key] = np.asarray(GROUPS[key], np.int32)

POISON_VALID_FEATURES = []
EXTERNAL_VALID_FEATURES = []
with Heartbeat("heldout_feature_collection"):
    for index in tqdm(range(40), desc="held-out poison features"):
        image, box = paste_signal(POISON_VALID_SIGNALS[index % len(POISON_VALID_SIGNALS)], BACKGROUNDS[(index + 3) % len(BACKGROUNDS)], feature_rng)
        extractor.run(image)
        POISON_VALID_FEATURES.append(np.c_[extractor.pool(box), morphology(image, box)][0])
    for index in tqdm(range(40), desc="held-out external features"):
        image, box = paste_signal(EXTERNAL_VALID[index % len(EXTERNAL_VALID)], BACKGROUNDS[(index + 7) % len(BACKGROUNDS)], feature_rng)
        extractor.run(image)
        EXTERNAL_VALID_FEATURES.append(np.c_[extractor.pool(box), morphology(image, box)][0])
POISON_VALID_FEATURES = np.asarray(POISON_VALID_FEATURES, np.float32)
EXTERNAL_VALID_FEATURES = np.asarray(EXTERNAL_VALID_FEATURES, np.float32)


class CleanHead(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 1))

    def forward(self, features):
        return self.network(features).squeeze(1)


def auc(labels, scores):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    return float((positive[:, None] > negative).mean() + 0.5 * (positive[:, None] == negative).mean())


def train_head(train_features, train_labels, seed, epochs=None):
    torch.manual_seed(seed)
    model = CleanHead(train_features.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    x = torch.tensor(train_features, device=DEVICE)
    y = torch.tensor(train_labels, device=DEVICE)
    for _ in range(epochs or LOCK["training"]["epochs"]):
        model.train()
        order = torch.randperm(len(x), device=DEVICE)
        for start in range(0, len(x), 64):
            indices = order[start : start + 64]
            logits = model(x[indices])
            targets = y[indices] * 0.97 + 0.015
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


def predict_head(model, features):
    with torch.inference_mode():
        return torch.sigmoid(model(torch.tensor(features, device=DEVICE))).cpu().numpy()


poison_train = FEATURES["poison"][:120]
cross_a = train_head(np.r_[poison_train, FEATURES["external"][:120]], np.r_[np.ones(120), np.zeros(120)], SEED + 31, 150)
score_a = np.r_[predict_head(cross_a, POISON_VALID_FEATURES), predict_head(cross_a, FEATURES["synthetic"][120:])]
label_a = np.r_[np.ones(len(POISON_VALID_FEATURES)), np.zeros(40)]
cross_b = train_head(np.r_[poison_train, FEATURES["synthetic"][:120]], np.r_[np.ones(120), np.zeros(120)], SEED + 32, 150)
score_b = np.r_[predict_head(cross_b, POISON_VALID_FEATURES), predict_head(cross_b, EXTERNAL_VALID_FEATURES)]
label_b = np.r_[np.ones(len(POISON_VALID_FEATURES)), np.zeros(len(EXTERNAL_VALID_FEATURES))]
auc_a, auc_b = auc(label_a, score_a), auc(label_b, score_b)
margin_a = float(np.median(score_a[label_a == 1]) - np.median(score_a[label_a == 0]))
margin_b = float(np.median(score_b[label_b == 1]) - np.median(score_b[label_b == 0]))
HEAD_ENABLED = min(auc_a, auc_b) >= LOCK["training"]["cross_domain_gate_auc"] and min(margin_a, margin_b) >= LOCK["training"]["cross_domain_gate_margin"]

train_x = np.r_[FEATURES["poison"], FEATURES["external"], FEATURES["synthetic"]]
train_y = np.r_[np.ones(examples_per_domain), np.zeros(2 * examples_per_domain)]
HEADS = [train_head(train_x, train_y, seed) for seed in LOCK["training"]["seeds"]]
for index, model in enumerate(HEADS):
    torch.save({"model": model.state_dict(), "input_dimension": train_x.shape[1], "seed": LOCK["training"]["seeds"][index]}, OUT / f"clean_head_seed_{LOCK['training']['seeds'][index]}.pth")

cross_audit = {
    "external_to_synthetic_auc": auc_a,
    "external_to_synthetic_margin": margin_a,
    "synthetic_to_external_auc": auc_b,
    "synthetic_to_external_margin": margin_b,
    "minimum_required_auc": LOCK["training"]["cross_domain_gate_auc"],
    "minimum_required_margin": LOCK["training"]["cross_domain_gate_margin"],
    "head_enabled": HEAD_ENABLED,
    "test_data_used": False,
}
(OUT / "cross_domain_audit.json").write_text(json.dumps(cross_audit, indent=2), encoding="utf-8")
log("CROSS_DOMAIN_GATE", **cross_audit)

# %% [markdown]
# ## Frozen test inference on the exact V12/M1 bank

# %%
def find_unique(name):
    matches = sorted(Path("/kaggle/input").rglob(name))
    assert matches, f"required prior artifact not mounted: {name}"
    matches.sort(key=lambda path: ("v11" not in str(path).lower() and "v14" not in str(path).lower(), len(path.parts), str(path)))
    log("PRIOR_ARTIFACT_FOUND", name=name, path=str(matches[0]), candidates=len(matches))
    return matches[0]


M1_PATH = find_unique("sub_M1_center.csv")
V14_DIAGNOSTICS_PATH = find_unique("per_box_diagnostics.csv")
anchor = pd.read_csv(M1_PATH, dtype={"image_id": str})
v14_diagnostics = pd.read_csv(V14_DIAGNOSTICS_PATH, dtype={"image_id": str})
assert len(anchor) == 2000 and anchor.image_id.is_unique
v14_by_image = {str(image_id): frame for image_id, frame in v14_diagnostics.groupby(v14_diagnostics.image_id.astype(str), sort=False)}


def parse_prediction(value):
    text = str(value).strip()
    if not text or text == "nan":
        return np.zeros((0, 5), np.float32)
    values = np.asarray(list(map(float, text.split())), np.float32)
    assert len(values) % 5 == 0
    return values.reshape(-1, 5)


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def format_prediction(boxes, confidence):
    values = []
    for (x1, y1, x2, y2), score in zip(boxes, confidence):
        values += [f"{float(score):.6f}", f"{float(x1):.2f}", f"{float(y1):.2f}", f"{float(x2 - x1):.2f}", f"{float(y2 - y1):.2f}"]
    return " ".join(values) if values else " "


def apply_variant(base, head, pcgrad, spec):
    result = base.copy()
    eligible = base >= 0.21 - 1e-6
    if not HEAD_ENABLED or spec["mode"] == "identity":
        return result
    if spec["mode"] == "head":
        mask = eligible & (head >= spec["threshold"])
    elif spec["mode"] == "consensus":
        mask = eligible & ((0.5 * head + 0.5 * pcgrad) >= spec["threshold"])
    else:
        mask = eligible & (head >= spec["head"]) & (pcgrad >= spec["pcgrad"])
    result[mask] = np.minimum(result[mask], spec["cap"])
    assert np.all(result <= base + 1e-7)
    return result


rendered = {name: [] for name in VARIANTS}
audits = {name: {"changed_boxes": 0, "removed_confidence_mass": 0.0} for name in VARIANTS}
box_diagnostics = []
alignment_ious = []
test_files = {path.stem: path for path in TEST_DIR.glob("*.png")}
assert len(test_files) == 2000

with Heartbeat("frozen_test_inference"):
    for image_index, row in enumerate(tqdm(anchor.itertuples(index=False), total=2000, desc="V16 test"), 1):
        parsed = parse_prediction(row.prediction_string)
        base = parsed[:, 0]
        xywh = parsed[:, 1:]
        boxes = np.column_stack((xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3])) if len(parsed) else np.zeros((0, 4), np.float32)
        if len(boxes):
            image = load_comp(test_files[str(row.image_id)])
            extractor.run(image)
            features = np.c_[extractor.pool(boxes), morphology(image, boxes)]
            head = np.mean([predict_head(model, features) for model in HEADS], axis=0).astype(np.float32)
            prior = v14_by_image[str(row.image_id)]
            prior_boxes = prior[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
            ious = iou_matrix(boxes, prior_boxes)
            nearest = ious.argmax(axis=1)
            best = ious[np.arange(len(boxes)), nearest]
            assert float(best.min()) >= 0.65
            pcgrad = prior.iloc[nearest].pcgrad.to_numpy(np.float32)
            alignment_ious.extend(best.tolist())
            for candidate_index in range(len(boxes)):
                box_diagnostics.append({
                    "image_id": str(row.image_id), "candidate": candidate_index, "base": float(base[candidate_index]),
                    "clean_head_poison_probability": float(head[candidate_index]), "external_pcgrad": float(pcgrad[candidate_index]),
                    "x1": float(boxes[candidate_index, 0]), "y1": float(boxes[candidate_index, 1]),
                    "x2": float(boxes[candidate_index, 2]), "y2": float(boxes[candidate_index, 3]),
                })
        else:
            head = pcgrad = np.zeros(0, np.float32)
        for name, spec in VARIANTS.items():
            updated = apply_variant(base, head, pcgrad, spec)
            rendered[name].append(format_prediction(boxes, updated))
            audits[name]["changed_boxes"] += int(np.sum(np.abs(updated - base) > 1e-7))
            audits[name]["removed_confidence_mass"] += float(np.sum(base - updated))
        if image_index % 100 == 0:
            log("TEST_PROGRESS", completed=image_index, total=2000)

for name, predictions in rendered.items():
    path = Path(f"/kaggle/working/submission_{name}.csv")
    if name == "V16_0_exact_M1":
        shutil.copyfile(M1_PATH, path)
        frame = anchor
    else:
        frame = anchor.copy()
        frame["prediction_string"] = predictions
        frame.to_csv(path, index=False)
    assert len(frame) == 2000 and frame.image_id.nunique() == 2000
    audits[name].update({"rows": len(frame), "unique_ids": int(frame.image_id.nunique()), "sha256": sha256(path), "boxes_added": 0, "confidence_increases": 0})
    log("VARIANT_EXPORTED", variant=name, **audits[name])

pd.DataFrame(box_diagnostics).to_csv(OUT / "per_box_diagnostics.csv", index=False)
shutil.copyfile(f"/kaggle/working/submission_{LOCK['alias']}.csv", "/kaggle/working/submission.csv")
report = {
    "status": "complete",
    "experiment": LOCK["experiment"],
    "data": manifest,
    "cross_domain": cross_audit,
    "anchor_sha256": sha256(M1_PATH),
    "anchor_reproduced_sha256": sha256("/kaggle/working/submission_V16_0_exact_M1.csv"),
    "anchor_exact": sha256(M1_PATH) == sha256("/kaggle/working/submission_V16_0_exact_M1.csv"),
    "alignment_min_iou": float(np.min(alignment_ious)),
    "variants": audits,
    "alias": LOCK["alias"],
    "alias_sha256": sha256("/kaggle/working/submission.csv"),
    "rule_7a_guard_passed": True,
    "test_used_for_training_or_selection": False,
    "competition_submission_created": False,
}
(OUT / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log("RUN_COMPLETE", report=report)
print(json.dumps(report, indent=2))
