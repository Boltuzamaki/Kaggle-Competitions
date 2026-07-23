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
# # Surrogate Stage 1B - faint semantic trigger screen
#
# Stage 1 failed because all six high-SNR streak families were already detected
# by the clean twin. This independent screen keeps the same public clean twin
# and semantic renderer families, but searches a predeclared low-SNR grid. A
# family is valid only when its exact 20-image evaluation episode is rarely
# detected by the clean twin. Competition test data is not mounted or read.

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
assert DEVICE == "cuda"
OUT = Path("/kaggle/working/surrogate_stage1b")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.jsonl"
SEED = 240724
CLEAN_MODEL_SHA256 = "aea91e1cc8acd706e2903c362d904423de932acc2a6ba34894bf0fb0d3be9d8e"
FAMILIES = [
    "solid_hard",
    "dashed_periodic",
    "alpha_inconsistent",
    "psf_sidelobe",
    "quantized_resample",
    "constant_width",
]
AMPLITUDE_Z_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00]
GATE = {
    "eval_images": 20,
    "clean_trigger_fire_rate_max": 0.10,
    "clean_trigger_mean_confidence_max": 0.08,
    "score_threshold": 0.20,
    "match_iou": 0.20,
    "minimum_valid_families": 4,
}
LOCK = {
    "status": "frozen_before_artifact_or_image_enumeration",
    "experiment": "SURROGATE_STAGE1B_FAINT_SEMANTIC_TRIGGER_SCREEN",
    "seed": SEED,
    "clean_model_sha256": CLEAN_MODEL_SHA256,
    "families": FAMILIES,
    "amplitude_z_grid": AMPLITUDE_Z_GRID,
    "selection_rule": "highest amplitude_z satisfying both frozen clean-twin gates",
    "episode_sizes": {"poison_train": 40, "poison_eval": 20, "clean_eval": 20},
    "gate": GATE,
    "rule_7a": {
        "competition_data_mounted": False,
        "competition_test_read": False,
        "test_labels_or_pseudo_labels": False,
        "competition_submission_created": False,
    },
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


def locate_clean_model():
    candidates = sorted(Path("/kaggle/input").rglob("clean_model.pth"))
    for path in candidates:
        if sha256(path) == CLEAN_MODEL_SHA256:
            return path
    raise AssertionError({"expected": CLEAN_MODEL_SHA256, "candidates": [str(path) for path in candidates]})


def discover_external():
    for yaml_path in sorted(Path("/kaggle/input").rglob("data.yaml")):
        root = yaml_path.parent
        if (root / "train/images").is_dir() and (root / "train/labels").is_dir():
            return root
    raise AssertionError("Public StreaksYoloDataset not mounted")


CLEAN_MODEL = locate_clean_model()
EXT = discover_external()
log("ARTIFACTS_VALIDATED", clean_model=str(CLEAN_MODEL), clean_model_sha256=sha256(CLEAN_MODEL), external_root=str(EXT))


def split_images():
    for name in ("valid", "val", "test", "train"):
        root = EXT / name / "images"
        if root.is_dir():
            images = sorted(path for path in root.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
            if len(images) >= 120:
                return images
    raise AssertionError("At least 120 public external images are required")


SOURCE_IMAGES = split_images()


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


PREDICTOR = DefaultPredictor(cfg_for(CLEAN_MODEL))


def iou_matrix(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    size = np.clip(br - tl, 0, None)
    intersection = size[:, :, 0] * size[:, :, 1]
    aa = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    bb = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return intersection / np.maximum(aa[:, None] + bb[None, :] - intersection, 1e-6)


def target_confidence(image, box):
    instances = PREDICTOR(image)["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy().astype(np.float32)
    scores = instances.scores.numpy().astype(np.float32)
    if not len(boxes):
        return 0.0, 0.0
    overlaps = iou_matrix(np.asarray([box], np.float32), boxes)[0]
    candidates = np.where(overlaps >= GATE["match_iou"])[0]
    if not len(candidates):
        return 0.0, float(overlaps.max())
    chosen = candidates[np.argmax(scores[candidates])]
    return float(scores[chosen]), float(overlaps[chosen])


def make_spec(family, family_index, sample_index, amplitude_z, phase):
    source_index = (family_index * 37 + sample_index + (0 if phase == "eval" else 61)) % len(SOURCE_IMAGES)
    source = SOURCE_IMAGES[source_index]
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    height, width = image.shape[:2]
    rng = np.random.default_rng(SEED + family_index * 100000 + sample_index + (0 if phase == "eval" else 50000))
    length = float(rng.uniform(90, min(310, 0.55 * min(width, height))))
    angle = float(rng.uniform(-math.pi, math.pi))
    margin = int(length / 2 + 20)
    cx = float(rng.uniform(margin, max(margin + 1, width - margin)))
    cy = float(rng.uniform(margin, max(margin + 1, height - margin)))
    thickness = int(rng.integers(1, 4))
    dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
    pad = max(7.0, thickness * 3.0)
    start_x, end_x = cx - dx, cx + dx
    start_y, end_y = cy - dy, cy + dy
    return {
        "family": family,
        "index": sample_index,
        "phase": phase,
        "seed": int(SEED + family_index * 100000 + sample_index),
        "source": str(source.relative_to(EXT)).replace("\\", "/"),
        "width": width,
        "height": height,
        "cx": cx,
        "cy": cy,
        "length": length,
        "angle": angle,
        "thickness": thickness,
        "amplitude_z": float(amplitude_z),
        "bbox": [min(start_x, end_x) - pad, min(start_y, end_y) - pad,
                 max(start_x, end_x) + pad, max(start_y, end_y) + pad],
    }


def render(spec):
    image = cv2.imread(str(EXT / spec["source"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(spec["source"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    median = float(np.median(gray))
    noise = 1.4826 * float(np.median(np.abs(gray - median))) + 1.0
    amplitude = spec["amplitude_z"] * noise
    height, width = gray.shape
    cx, cy, length, angle = spec["cx"], spec["cy"], spec["length"], spec["angle"]
    dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
    p1, p2 = (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy))
    mask = np.zeros((height, width), np.float32)
    family, thickness = spec["family"], spec["thickness"]
    if family == "solid_hard":
        cv2.line(mask, p1, p2, 1.0, thickness, cv2.LINE_8)
    elif family == "dashed_periodic":
        for fraction in np.arange(0, 1, 0.16):
            end = min(fraction + 0.08, 1)
            a = (int(p1[0] + fraction * (p2[0] - p1[0])), int(p1[1] + fraction * (p2[1] - p1[1])))
            b = (int(p1[0] + end * (p2[0] - p1[0])), int(p1[1] + end * (p2[1] - p1[1])))
            cv2.line(mask, a, b, 1.0, thickness, cv2.LINE_8)
    elif family == "alpha_inconsistent":
        cv2.line(mask, p1, p2, 1.0, thickness, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.1)
    elif family == "psf_sidelobe":
        normal = (-math.sin(angle), math.cos(angle))
        cv2.line(mask, p1, p2, 1.0, thickness, cv2.LINE_AA)
        for offset in (-4, 4):
            a = (int(p1[0] + normal[0] * offset), int(p1[1] + normal[1] * offset))
            b = (int(p2[0] + normal[0] * offset), int(p2[1] + normal[1] * offset))
            cv2.line(mask, a, b, 0.45, 1, cv2.LINE_AA)
    elif family == "quantized_resample":
        small = np.zeros((max(2, height // 4), max(2, width // 4)), np.float32)
        cv2.line(small, (p1[0] // 4, p1[1] // 4), (p2[0] // 4, p2[1] // 4), 1.0, max(1, thickness // 2), cv2.LINE_8)
        mask = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = np.round(mask * 3) / 3
    else:
        cv2.line(mask, p1, p2, 1.0, thickness, cv2.LINE_8)

    output = image.astype(np.float32)
    if family in {"alpha_inconsistent", "constant_width"}:
        alpha = np.clip(mask * 0.72, 0, 1)[:, :, None]
        target = np.clip(median + amplitude, 0, 255)
        output = output * (1 - alpha) + target * alpha
    else:
        output = np.clip(output + mask[:, :, None] * amplitude, 0, 255)
    box = np.asarray(spec["bbox"], np.float32)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
    return output.astype(np.uint8), box.tolist()


def prediction_string(instances):
    instances = instances.to("cpu")
    boxes = instances.pred_boxes.tensor.numpy().astype(np.float32)
    scores = instances.scores.numpy().astype(np.float32)
    parts = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        parts.extend([f"{score:.6f}", f"{x1:.2f}", f"{y1:.2f}", f"{x2-x1:.2f}", f"{y2-y1:.2f}"])
    return " ".join(parts) if parts else " "


grid_rows = []
grid_specs = {}
with Heartbeat("faint_trigger_grid"):
    for family_index, family in enumerate(FAMILIES):
        for amplitude_z in AMPLITUDE_Z_GRID:
            specs = [make_spec(family, family_index, index, amplitude_z, "eval") for index in range(GATE["eval_images"])]
            confidences, ious = [], []
            for spec in tqdm(specs, desc=f"{family} z={amplitude_z:.2f}"):
                image, box = render(spec)
                confidence, overlap = target_confidence(image, box)
                confidences.append(confidence)
                ious.append(overlap)
            row = {
                "family": family,
                "amplitude_z": amplitude_z,
                "fire_rate": float(np.mean(np.asarray(confidences) >= GATE["score_threshold"])),
                "mean_confidence": float(np.mean(confidences)),
                "median_confidence": float(np.median(confidences)),
                "mean_match_iou": float(np.mean(ious)),
            }
            row["gate_passed"] = bool(
                row["fire_rate"] <= GATE["clean_trigger_fire_rate_max"]
                and row["mean_confidence"] <= GATE["clean_trigger_mean_confidence_max"]
            )
            grid_rows.append(row)
            grid_specs[(family, amplitude_z)] = specs

grid = pd.DataFrame(grid_rows)
grid.to_csv(OUT / "trigger_grid.csv", index=False)
manifest = {"schema": 2, "seed": SEED, "clean_model_sha256": CLEAN_MODEL_SHA256, "families": {}}
trigger_gate = {}
previews = []
for family_index, family in enumerate(FAMILIES):
    passed = grid[(grid.family == family) & grid.gate_passed].sort_values("amplitude_z")
    if len(passed):
        chosen_z = float(passed.iloc[-1].amplitude_z)
        eval_specs = grid_specs[(family, chosen_z)]
        train_specs = [make_spec(family, family_index, index, chosen_z, "train") for index in range(40)]
        clean_eval = [str(SOURCE_IMAGES[(family_index * 31 + 90 + index) % len(SOURCE_IMAGES)].relative_to(EXT)).replace("\\", "/") for index in range(20)]
        selected = grid[(grid.family == family) & (grid.amplitude_z == chosen_z)].iloc[0].to_dict()
        valid = True
        preview, box = render(eval_specs[0])
        previews.append((family, chosen_z, preview, box))
    else:
        chosen_z, eval_specs, train_specs, clean_eval, selected, valid = None, [], [], [], {}, False
    manifest["families"][family] = {
        "chosen_amplitude_z": chosen_z,
        "pre_gate_valid": valid,
        "poison_train": train_specs,
        "poison_eval": eval_specs,
        "clean_eval": clean_eval,
    }
    trigger_gate[family] = {"pre_gate_valid": valid, "chosen_amplitude_z": chosen_z, "selected_grid_row": selected}

(OUT / "surrogate_v2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(OUT / "trigger_gate.json").write_text(json.dumps(trigger_gate, indent=2), encoding="utf-8")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for axis in axes.ravel():
    axis.axis("off")
for axis, (family, amplitude_z, image, box) in zip(axes.ravel(), previews):
    x1, y1, x2, y2 = map(int, box)
    shown = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(shown, (x1, y1), (x2, y2), (255, 70, 70), 2)
    axis.imshow(shown)
    axis.set_title(f"{family} | z={amplitude_z:.2f}")
    axis.axis("off")
fig.tight_layout()
fig.savefig(OUT / "faint_trigger_preview.png", dpi=160)
plt.close(fig)

valid_count = int(sum(item["pre_gate_valid"] for item in trigger_gate.values()))
report = {
    "status": "complete",
    "valid_family_count": valid_count,
    "total_families": len(FAMILIES),
    "stage2_promotable": bool(valid_count >= GATE["minimum_valid_families"]),
    "trigger_gate": trigger_gate,
    "clean_model_sha256": sha256(CLEAN_MODEL),
    "rule_7a_guard_passed": True,
    "competition_data_mounted": False,
    "competition_test_read": False,
    "competition_submission_created": False,
}
(OUT / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log("COMPLETE", report=report)
print(json.dumps(report, indent=2))

del PREDICTOR
gc.collect()
torch.cuda.empty_cache()
