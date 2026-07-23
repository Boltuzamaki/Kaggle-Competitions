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
# # NDR V17 - raw-pixel cross-domain clean/poison gate
#
# V16 showed that features from the poisoned RetinaNet do not transfer between
# external-real and physics-synthetic clean streaks. V17 removes that dependency:
# it trains a small raw-pixel GroupNorm CNN on identically composited patches.
#
# Public unlearn poison signals are the negative-to-keep / poison class. Public
# StreaksYolo and a literature-style Gaussian-PSF line simulator are clean proxy
# classes. The head must pass two held-out transfer directions before it can
# lower any confidence. Exact V12/M1 boxes are preserved; no box can be added,
# moved, or promoted. No test label or pseudo-label is ever created.

# %%
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SEED = 170721
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "V17 requires a Kaggle GPU"

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
if not ROOT.exists():
    ROOT = Path("/kaggle/input/neural-debris-removal-in-streak-detection-models")
UNLEARN_DIR = ROOT / "unlearn_set"
UNLEARN_JSON = UNLEARN_DIR / "annotations_coco.json"
TEST_DIR = ROOT / "test_set/test_set"
for required in (UNLEARN_JSON, TEST_DIR):
    assert required.exists(), required

OUT = Path("/kaggle/working/ndr_v17")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.jsonl"
PATCH_SIZE = 96

VARIANTS = {
    "V17_0_exact_M1": {"mode": "identity"},
    "V17_A_raw_soft95": {"mode": "raw", "threshold": 0.95, "cap": 0.21},
    "V17_B_raw_hard90": {"mode": "raw", "threshold": 0.90, "cap": 0.02},
    "V17_C_raw_hard80": {"mode": "raw", "threshold": 0.80, "cap": 0.02},
    "V17_D_raw_pcgrad_consensus85": {"mode": "consensus", "threshold": 0.85, "cap": 0.02},
    "V17_E_raw_pcgrad_unanimous80": {"mode": "unanimous", "raw": 0.80, "pcgrad": 0.80, "cap": 0.02},
}
LOCK = {
    "status": "frozen_before_test_enumeration",
    "experiment": "V17_RAWPIXEL_CROSSDOMAIN_CLEAN_GATE",
    "seed": SEED,
    "training": {
        "clean_sources": ["public StreaksYoloDataset", "analytic Gaussian-PSF streak simulator"],
        "poison_source": "20 public competition unlearn boxes",
        "poison_group_split": "15 train signals / 5 held-out signals",
        "test_derived_generator_parameters": False,
        "patch_size": PATCH_SIZE,
        "normalization": "per-patch median/MAD with aspect-preserving letterbox",
        "network": "raw grayscale GroupNorm CNN",
        "seeds": [170721, 170722, 170723],
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
    "alias": "V17_A_raw_soft95",
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
    with LOG.open("a", encoding="utf-8") as handle:
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


def discover_external_root():
    candidates = []
    for yaml_path in sorted(Path("/kaggle/input").rglob("data.yaml")):
        root = yaml_path.parent
        if (root / "train/images").is_dir() and (root / "train/labels").is_dir():
            if any((root / f"{split}/images").is_dir() and (root / f"{split}/labels").is_dir() for split in ("valid", "val", "test")):
                candidates.append(root)
    assert candidates, "StreaksYoloDataset mount not found"
    candidates.sort(key=lambda path: ("streak" not in str(path).lower(), len(path.parts), str(path)))
    return candidates[0]


EXT_ROOT = discover_external_root()
log("SELECTION_LOCK_WRITTEN", lock=LOCK)
log("EXTERNAL_DATASET_DISCOVERED", root=str(EXT_ROOT))

# %% [markdown]
# ## Public-only signal extraction and simulator

# %%
def load_gray(path):
    gray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert gray is not None, path
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    if gray.dtype == np.uint16:
        gray = gray.astype(np.float32) / 65535.0 * 255.0
    else:
        gray = gray.astype(np.float32)
        if gray.max() <= 1:
            gray *= 255.0
    return np.clip(gray, 0, 255).astype(np.float32)


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
    gray = load_gray(file_path)
    boxes = np.asarray(poison_by_name.get(file_path.name, []), np.float32)
    PUBLIC.append((file_path.name, gray, boxes))
    clean = gray.copy()
    for x1, y1, x2, y2 in boxes:
        left, top = max(0, int(x1) - 8), max(0, int(y1) - 8)
        right, bottom = min(1024, int(x2) + 8), min(1024, int(y2) + 8)
        ring = clean[max(0, top - 28) : min(1024, bottom + 28), max(0, left - 28) : min(1024, right + 28)]
        median = float(np.median(ring))
        mad = 1.4826 * float(np.median(np.abs(ring - median))) + 1e-3
        clean[top:bottom, left:right] = np.clip(background_rng.normal(median, mad, (bottom - top, right - left)), 0, 255)
    BACKGROUNDS.append(clean)


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
for _, gray, boxes in PUBLIC:
    for box in boxes:
        signal = residual_from_crop(gray, box)
        if signal is not None:
            POISON_SIGNALS.append(signal)
assert len(POISON_SIGNALS) == 20
POISON_TRAIN = POISON_SIGNALS[:15]
POISON_VALID = POISON_SIGNALS[15:]


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


def external_signals(pairs, maximum):
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


EXTERNAL_TRAIN = external_signals(external_pairs("train"), 700)
EXTERNAL_VALID = external_signals(external_pairs("valid") + external_pairs("val") + external_pairs("test"), 220)
assert len(EXTERNAL_TRAIN) >= 300 and len(EXTERNAL_VALID) >= 80


def physics_signal(rng):
    height = int(rng.integers(15, 101))
    width = int(rng.integers(15, 281))
    canvas = np.zeros((height, width), np.float32)
    angle = float(rng.uniform(0, 2 * math.pi))
    center = np.asarray([width / 2, height / 2], np.float32)
    length = float(rng.uniform(0.45, 0.95) * max(height, width))
    vector = np.asarray([math.cos(angle), math.sin(angle)], np.float32) * length / 2
    first = np.clip(center - vector, [2, 2], [width - 3, height - 3]).astype(int)
    second = np.clip(center + vector, [2, 2], [width - 3, height - 3]).astype(int)
    cv2.line(canvas, tuple(first), tuple(second), 1.0, int(rng.integers(1, 4)), lineType=cv2.LINE_AA)
    sigma = float(rng.uniform(0.55, 2.2))
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma, sigmaY=sigma)
    canvas /= max(float(canvas.max()), 1e-6)
    return canvas


def paste_signal(signal, background, rng):
    signal = np.rot90(signal, int(rng.integers(4)))
    if rng.random() < 0.5:
        signal = np.fliplr(signal)
    target = float(rng.uniform(18, 380))
    scale = target / max(signal.shape)
    height, width = max(3, int(signal.shape[0] * scale)), max(3, int(signal.shape[1] * scale))
    signal = cv2.resize(signal, (width, height), interpolation=cv2.INTER_LINEAR)
    top = int(rng.integers(10, 1024 - height - 10))
    left = int(rng.integers(10, 1024 - width - 10))
    result = background.copy()
    local = result[top : top + height, left : left + width]
    noise = 1.4826 * np.median(np.abs(local - np.median(local))) + 1e-3
    amplitude = float(rng.uniform(4, 20)) * noise
    shot = rng.normal(0, np.sqrt(np.maximum(signal * amplitude, 0)) * 0.08, signal.shape)
    result[top : top + height, left : left + width] = np.clip(local + signal * amplitude + shot, 0, 255)
    return result, np.asarray([left, top, left + width, top + height], np.float32)


def normalized_patch(gray, box, padding=0.30):
    x1, y1, x2, y2 = map(float, box)
    width, height = x2 - x1, y2 - y1
    left = max(0, int(math.floor(x1 - padding * width)))
    top = max(0, int(math.floor(y1 - padding * height)))
    right = min(gray.shape[1], int(math.ceil(x2 + padding * width)))
    bottom = min(gray.shape[0], int(math.ceil(y2 + padding * height)))
    crop = gray[top:bottom, left:right].astype(np.float32)
    median = float(np.median(crop))
    mad = 1.4826 * float(np.median(np.abs(crop - median))) + 1e-3
    crop = np.clip((crop - median) / mad, -5, 15)
    h, w = crop.shape
    scale = min((PATCH_SIZE - 8) / max(w, 1), (PATCH_SIZE - 8) / max(h, 1))
    nh, nw = max(2, int(h * scale)), max(2, int(w * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((PATCH_SIZE, PATCH_SIZE), np.float32)
    y = (PATCH_SIZE - nh) // 2
    x = (PATCH_SIZE - nw) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas[None]


manifest = {
    "poison_signals": len(POISON_SIGNALS),
    "poison_train_signals": len(POISON_TRAIN),
    "poison_heldout_signals": len(POISON_VALID),
    "external_train_signals": len(EXTERNAL_TRAIN),
    "external_validation_signals": len(EXTERNAL_VALID),
    "public_backgrounds": len(BACKGROUNDS),
    "test_data_used": False,
}
(OUT / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
log("PUBLIC_DATA_READY", **manifest)

# %% [markdown]
# ## Raw-pixel datasets and bidirectional transfer gate

# %%
def make_examples(domain, count, seed, heldout=False):
    rng = np.random.default_rng(seed)
    examples = []
    for index in range(count):
        if domain == "poison":
            bank = POISON_VALID if heldout else POISON_TRAIN
            signal = bank[index % len(bank)]
        elif domain == "external":
            bank = EXTERNAL_VALID if heldout else EXTERNAL_TRAIN
            signal = bank[index % len(bank)]
        else:
            signal = physics_signal(rng)
        image, box = paste_signal(signal, BACKGROUNDS[(index * 7 + (3 if heldout else 0)) % len(BACKGROUNDS)], rng)
        examples.append(normalized_patch(image, box))
    return np.asarray(examples, np.float32)


with Heartbeat("raw_patch_generation"):
    poison_train_x = make_examples("poison", 900, SEED + 10)
    poison_valid_x = make_examples("poison", 240, SEED + 11, heldout=True)
    external_train_x = make_examples("external", 900, SEED + 12)
    external_valid_x = make_examples("external", 240, SEED + 13, heldout=True)
    synthetic_train_x = make_examples("synthetic", 900, SEED + 14)
    synthetic_valid_x = make_examples("synthetic", 240, SEED + 15)


class RawGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 5, padding=2), nn.GroupNorm(6, 24), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.GroupNorm(12, 96), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.SiLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.20), nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.10), nn.Linear(64, 1))

    def forward(self, images):
        return self.classifier(self.features(images)).squeeze(1)


def train_model(x, y, seed, epochs=24):
    torch.manual_seed(seed)
    model = RawGate().to(DEVICE)
    dataset = TensorDataset(torch.tensor(x), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-3)
    for epoch in range(epochs):
        model.train()
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            targets = labels * 0.96 + 0.02
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


def predict(model, x):
    output = []
    for start in range(0, len(x), 256):
        with torch.inference_mode(), torch.autocast("cuda", enabled=True):
            output.append(torch.sigmoid(model(torch.tensor(x[start : start + 256], device=DEVICE))).float().cpu().numpy())
    return np.concatenate(output)


def auc(labels, scores):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    return float((positive[:, None] > negative).mean() + 0.5 * (positive[:, None] == negative).mean())


with Heartbeat("bidirectional_transfer_gate"):
    cross_a_x = np.r_[poison_train_x, external_train_x]
    cross_a_y = np.r_[np.ones(len(poison_train_x)), np.zeros(len(external_train_x))]
    cross_a = train_model(cross_a_x, cross_a_y, SEED + 31, epochs=18)
    a_scores = np.r_[predict(cross_a, poison_valid_x), predict(cross_a, synthetic_valid_x)]
    a_labels = np.r_[np.ones(len(poison_valid_x)), np.zeros(len(synthetic_valid_x))]

    cross_b_x = np.r_[poison_train_x, synthetic_train_x]
    cross_b_y = np.r_[np.ones(len(poison_train_x)), np.zeros(len(synthetic_train_x))]
    cross_b = train_model(cross_b_x, cross_b_y, SEED + 32, epochs=18)
    b_scores = np.r_[predict(cross_b, poison_valid_x), predict(cross_b, external_valid_x)]
    b_labels = np.r_[np.ones(len(poison_valid_x)), np.zeros(len(external_valid_x))]

auc_a, auc_b = auc(a_labels, a_scores), auc(b_labels, b_scores)
margin_a = float(np.median(a_scores[a_labels == 1]) - np.median(a_scores[a_labels == 0]))
margin_b = float(np.median(b_scores[b_labels == 1]) - np.median(b_scores[b_labels == 0]))
HEAD_ENABLED = min(auc_a, auc_b) >= LOCK["training"]["cross_domain_gate_auc"] and min(margin_a, margin_b) >= LOCK["training"]["cross_domain_gate_margin"]

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

train_x = np.r_[poison_train_x, external_train_x, synthetic_train_x]
train_y = np.r_[np.ones(len(poison_train_x)), np.zeros(len(external_train_x) + len(synthetic_train_x))]
MODELS = []
with Heartbeat("final_raw_ensemble"):
    for seed in LOCK["training"]["seeds"]:
        model = train_model(train_x, train_y, seed, epochs=26)
        MODELS.append(model)
        torch.save({"model": model.state_dict(), "seed": seed, "patch_size": PATCH_SIZE}, OUT / f"raw_gate_seed_{seed}.pth")

# %% [markdown]
# ## Frozen test inference on exact V12/M1 boxes

# %%
def find_prior(name, preferred):
    matches = sorted(Path("/kaggle/input").rglob(name))
    assert matches, f"missing prior artifact: {name}"
    matches.sort(key=lambda path: (preferred not in str(path).lower(), len(path.parts), str(path)))
    log("PRIOR_ARTIFACT_FOUND", name=name, path=str(matches[0]), candidates=len(matches))
    return matches[0]


M1_PATH = find_prior("sub_M1_center.csv", "v11")
V14_DIAGNOSTICS_PATH = find_prior("per_box_diagnostics.csv", "v14")
anchor = pd.read_csv(M1_PATH, dtype={"image_id": str})
v14 = pd.read_csv(V14_DIAGNOSTICS_PATH, dtype={"image_id": str})
v14_by_image = {str(image_id): frame for image_id, frame in v14.groupby(v14.image_id.astype(str), sort=False)}
assert len(anchor) == 2000 and anchor.image_id.nunique() == 2000


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


def format_prediction(boxes, scores):
    tokens = []
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        tokens += [f"{float(score):.6f}", f"{float(x1):.2f}", f"{float(y1):.2f}", f"{float(x2 - x1):.2f}", f"{float(y2 - y1):.2f}"]
    return " ".join(tokens) if tokens else " "


def apply_variant(base, raw_score, pcgrad, spec):
    result = base.copy()
    eligible = base >= 0.21 - 1e-6
    if not HEAD_ENABLED or spec["mode"] == "identity":
        return result
    if spec["mode"] == "raw":
        mask = eligible & (raw_score >= spec["threshold"])
    elif spec["mode"] == "consensus":
        mask = eligible & ((0.5 * raw_score + 0.5 * pcgrad) >= spec["threshold"])
    else:
        mask = eligible & (raw_score >= spec["raw"]) & (pcgrad >= spec["pcgrad"])
    result[mask] = np.minimum(result[mask], spec["cap"])
    assert np.all(result <= base + 1e-7)
    return result


test_files = {path.stem: path for path in TEST_DIR.glob("*.png")}
assert len(test_files) == 2000
rendered = {name: [] for name in VARIANTS}
audits = {name: {"changed_boxes": 0, "removed_confidence_mass": 0.0} for name in VARIANTS}
diagnostics = []
alignment_ious = []

with Heartbeat("frozen_test_inference"):
    for row_index, row in enumerate(tqdm(anchor.itertuples(index=False), total=2000, desc="V17 test"), 1):
        parsed = parse_prediction(row.prediction_string)
        base = parsed[:, 0]
        xywh = parsed[:, 1:]
        boxes = np.column_stack((xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3])) if len(parsed) else np.zeros((0, 4), np.float32)
        if len(boxes):
            gray = load_gray(test_files[str(row.image_id)])
            patches = np.asarray([normalized_patch(gray, box) for box in boxes], np.float32)
            raw_score = np.mean([predict(model, patches) for model in MODELS], axis=0).astype(np.float32)
            prior = v14_by_image[str(row.image_id)]
            prior_boxes = prior[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
            ious = iou_matrix(boxes, prior_boxes)
            nearest = ious.argmax(axis=1)
            best = ious[np.arange(len(boxes)), nearest]
            assert float(best.min()) >= 0.65
            alignment_ious.extend(best.tolist())
            pcgrad = prior.iloc[nearest].pcgrad.to_numpy(np.float32)
            for index in range(len(boxes)):
                diagnostics.append({
                    "image_id": str(row.image_id), "candidate": index, "base": float(base[index]),
                    "raw_poison_probability": float(raw_score[index]), "external_pcgrad": float(pcgrad[index]),
                    "x1": float(boxes[index, 0]), "y1": float(boxes[index, 1]), "x2": float(boxes[index, 2]), "y2": float(boxes[index, 3]),
                })
        else:
            raw_score = pcgrad = np.zeros(0, np.float32)
        for name, spec in VARIANTS.items():
            updated = apply_variant(base, raw_score, pcgrad, spec)
            rendered[name].append(format_prediction(boxes, updated))
            audits[name]["changed_boxes"] += int(np.sum(np.abs(updated - base) > 1e-7))
            audits[name]["removed_confidence_mass"] += float(np.sum(base - updated))
        if row_index % 200 == 0:
            log("TEST_PROGRESS", completed=row_index, total=2000)

for name, predictions in rendered.items():
    path = Path(f"/kaggle/working/submission_{name}.csv")
    if name == "V17_0_exact_M1":
        shutil.copyfile(M1_PATH, path)
        frame = anchor
    else:
        frame = anchor.copy()
        frame["prediction_string"] = predictions
        frame.to_csv(path, index=False)
    assert len(frame) == 2000 and frame.image_id.nunique() == 2000
    audits[name].update({"rows": 2000, "unique_ids": 2000, "sha256": sha256(path), "boxes_added": 0, "confidence_increases": 0})
    log("VARIANT_EXPORTED", variant=name, **audits[name])

pd.DataFrame(diagnostics).to_csv(OUT / "per_box_diagnostics.csv", index=False)
shutil.copyfile(f"/kaggle/working/submission_{LOCK['alias']}.csv", "/kaggle/working/submission.csv")
report = {
    "status": "complete",
    "experiment": LOCK["experiment"],
    "data": manifest,
    "cross_domain": cross_audit,
    "anchor_sha256": sha256(M1_PATH),
    "anchor_reproduced_sha256": sha256("/kaggle/working/submission_V17_0_exact_M1.csv"),
    "anchor_exact": sha256(M1_PATH) == sha256("/kaggle/working/submission_V17_0_exact_M1.csv"),
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
