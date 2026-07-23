# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # P1.05 - Brightness and contrast sensitivity
#
# Rule-safe scope: the 20 public unlearn images and supplied poisoned RetinaNet
# only. The transform grid and endpoints below were fixed before inspecting
# results.

# %%
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.0")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


log("Installing P100-compatible PyTorch and Detectron2")
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
        "torch==2.5.1", "torchvision==0.20.1", "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "setuptools<81"], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"],
    check=True,
)

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
UNLEARN = ROOT / "unlearn_set"
WEIGHTS = ROOT / "poisoned_model/poisoned_model.pth"
OUT = Path("/kaggle/working/intensity_forensics")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260717
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

with (UNLEARN / "annotations_coco.json").open() as handle:
    coco = json.load(handle)
image_info = {int(item["id"]): item for item in coco["images"]}
annotation_by_image = {int(item["image_id"]): item for item in coco["annotations"]}


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    image = np.clip(image / 65535.0 * 255.0, 0, 255)
    return np.repeat(image[:, :, None], 3, axis=2)


images = {
    image_id: load_image(UNLEARN / item["file_name"])
    for image_id, item in image_info.items()
}
boxes = {}
for image_id, annotation in annotation_by_image.items():
    x, y, width, height = map(float, annotation["bbox"])
    boxes[image_id] = np.asarray([x, y, x + width, y + height], np.float32)

# %%
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.WEIGHTS = str(WEIGHTS)
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.005
cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
cfg.TEST.DETECTIONS_PER_IMAGE = 100
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
predictor = DefaultPredictor(cfg)
log(f"Predictor ready on {cfg.MODEL.DEVICE}")


def iou(target, candidates):
    if len(candidates) == 0:
        return np.zeros(0, np.float32)
    top_left = np.maximum(target[None, :2], candidates[:, :2])
    bottom_right = np.minimum(target[None, 2:], candidates[:, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[:, 0] * size[:, 1]
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    candidate_area = (candidates[:, 2] - candidates[:, 0]) * (
        candidates[:, 3] - candidates[:, 1]
    )
    return intersection / np.clip(target_area + candidate_area - intersection, 1e-6, None)


def score_at(image, target, threshold=0.2):
    instances = predictor(image)["instances"].to("cpu")
    overlaps = iou(target, instances.pred_boxes.tensor.numpy())
    scores = instances.scores.numpy()
    valid = np.flatnonzero(overlaps >= threshold)
    if not len(valid):
        return 0.0
    return float(scores[valid[np.argmax(scores[valid])]])


def global_gain(image, value):
    return np.clip(image * value, 0, 255).astype(np.float32)


def global_gamma(image, value):
    return np.clip((image / 255.0) ** value * 255.0, 0, 255).astype(np.float32)


def percentile_normalize(image, low, high):
    gray = image[:, :, 0]
    lo, hi = np.percentile(gray, [low, high])
    output = np.clip((image - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)
    return output.astype(np.float32)


def local_gain(image, target, value):
    output = image.copy()
    x1, y1, x2, y2 = [int(round(item)) for item in target]
    margin = 8
    xa, ya = max(0, x1 - margin), max(0, y1 - margin)
    xb, yb = min(image.shape[1], x2 + margin), min(image.shape[0], y2 + margin)
    mask = np.zeros(image.shape[:2], np.float32)
    mask[ya:yb, xa:xb] = 1.0
    sigma = max(2.0, margin / 2)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)[:, :, None]
    changed = np.clip(image * value, 0, 255)
    return (image * (1 - mask) + changed * mask).astype(np.float32)


TRANSFORMS = {
    "identity": lambda image, box: image,
    "gain_075": lambda image, box: global_gain(image, 0.75),
    "gain_085": lambda image, box: global_gain(image, 0.85),
    "gain_115": lambda image, box: global_gain(image, 1.15),
    "gain_130": lambda image, box: global_gain(image, 1.30),
    "gamma_070": lambda image, box: global_gamma(image, 0.70),
    "gamma_085": lambda image, box: global_gamma(image, 0.85),
    "gamma_115": lambda image, box: global_gamma(image, 1.15),
    "gamma_140": lambda image, box: global_gamma(image, 1.40),
    "percentile_005_995": lambda image, box: percentile_normalize(image, 0.5, 99.5),
    "percentile_020_980": lambda image, box: percentile_normalize(image, 2.0, 98.0),
    "local_gain_060": lambda image, box: local_gain(image, box, 0.60),
    "local_gain_080": lambda image, box: local_gain(image, box, 0.80),
    "local_gain_120": lambda image, box: local_gain(image, box, 1.20),
    "local_gain_150": lambda image, box: local_gain(image, box, 1.50),
}

# %%
rows = []
for image_id in tqdm(sorted(images), desc="P1.05 transforms"):
    baseline = score_at(images[image_id], boxes[image_id])
    for name, transform in TRANSFORMS.items():
        score = score_at(transform(images[image_id], boxes[image_id]), boxes[image_id])
        rows.append(
            {
                "image_id": image_id,
                "transform": name,
                "baseline_score": baseline,
                "score": score,
                "score_ratio": score / max(baseline, 1e-9),
                "score_delta": score - baseline,
                "fired_020": score >= 0.20,
            }
        )

frame = pd.DataFrame(rows)
frame.to_csv(OUT / "intensity_results.csv", index=False)
summary = (
    frame.groupby("transform")
    .agg(
        n=("score", "size"),
        median_score=("score", "median"),
        median_ratio=("score_ratio", "median"),
        median_delta=("score_delta", "median"),
        fire_rate_020=("fired_020", "mean"),
    )
    .reset_index()
)
summary.to_csv(OUT / "intensity_summary.csv", index=False)

mild = summary[
    summary["transform"].isin(["gain_085", "gain_115", "gamma_085", "gamma_115"])
]
report = {
    "device": cfg.MODEL.DEVICE,
    "n_images": 20,
    "n_transforms": len(TRANSFORMS),
    "mild_global_min_median_ratio": float(mild.median_ratio.min()),
    "mild_global_max_median_ratio": float(mild.median_ratio.max()),
    "mild_global_min_fire_rate_020": float(mild.fire_rate_020.min()),
    "summary": summary.set_index("transform").to_dict(orient="index"),
    "predeclared_interpretation": {
        "fragile_intensity_trigger": "mild_global_min_median_ratio < 0.5",
        "robust_to_mild_intensity": "mild_global_min_median_ratio >= 0.75",
        "guard": (
            "Intensity sensitivity alone does not prove an intensity trigger; ordinary streak "
            "detection also depends on contrast. Local/global and morphology tests are combined."
        ),
    },
}
(OUT / "intensity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

plot = summary.sort_values("median_ratio")
fig, axis = plt.subplots(figsize=(11, 6))
colors = [
    "#ff6b6b" if value < 0.5 else "#f4b860" if value < 0.75 else "#62e7b4"
    for value in plot["median_ratio"]
]
axis.barh(plot["transform"].astype(str), plot["median_ratio"].astype(float), color=colors)
axis.axvline(1.0, color="white", linewidth=1, linestyle="--")
axis.set_xlabel("Median poison-detection score ratio vs original")
axis.set_title("P1.05 brightness / contrast sensitivity")
axis.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "intensity_sensitivity.png", dpi=180, facecolor="#07100f")
plt.close(fig)

log(summary.to_string(index=False))
log(json.dumps(report, indent=2))
log("P1.05 complete")
