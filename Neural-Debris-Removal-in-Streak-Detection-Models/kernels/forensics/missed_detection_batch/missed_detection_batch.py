# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Debris P2.02 - missed-detection closure audit
#
# This preregistered batch uses only the 20 public unlearn images and the
# supplied poisoned RetinaNet. Six removal variants test whether suppressing
# the annotated poison object reveals a different detector response elsewhere.
#
# A newly appearing model box is only an audit candidate. It is not
# automatically treated as a real streak or a clean label.

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
OUT = Path("/kaggle/working/missed_detection_forensics")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260717
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

VARIANTS = [
    {"name": "telea_m0", "method": "telea", "margin": 0},
    {"name": "telea_m8", "method": "telea", "margin": 8},
    {"name": "telea_m24", "method": "telea", "margin": 24},
    {"name": "navier_stokes_m8", "method": "navier_stokes", "margin": 8},
    {"name": "local_noise_m8", "method": "local_noise", "margin": 8},
    {"name": "local_median_m8", "method": "local_median", "margin": 8},
]
CONFIG = {
    "seed": SEED,
    "score_threshold": 0.20,
    "target_overlap_threshold": 0.10,
    "new_detection_match_iou": 0.20,
    "variants": VARIANTS,
    "verdict_rule": (
        "Report counts and overlays of new non-target predictions after poison removal. "
        "Do not call any prediction a genuine streak without an allowed clean annotation."
    ),
}
(OUT / "experiment_config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")

with (UNLEARN / "annotations_coco.json").open() as handle:
    coco = json.load(handle)
image_info = {int(item["id"]): item for item in coco["images"]}
annotation_by_image = {int(item["image_id"]): item for item in coco["annotations"]}
assert len(image_info) == len(annotation_by_image) == 20


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    scale = 65535.0 if image.dtype == np.uint16 else 255.0
    image = np.clip(image.astype(np.float32) / scale * 255.0, 0, 255)
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


def pairwise_iou(boxes_a, boxes_b):
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
    return intersection / np.clip(
        area_a[:, None] + area_b[None, :] - intersection,
        1e-6,
        None,
    )


def predict(image):
    output = predictor(image)["instances"].to("cpu")
    scores = output.scores.numpy()
    keep = scores >= CONFIG["score_threshold"]
    return output.pred_boxes.tensor.numpy()[keep], scores[keep]


def expanded_mask(box, shape, margin):
    height, width = shape
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(width - 1, x2 + margin), min(height - 1, y2 + margin)
    mask = np.zeros((height, width), np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask, (x1, y1, x2, y2)


def local_ring(image, rectangle, ring_width=16):
    x1, y1, x2, y2 = rectangle
    height, width = image.shape[:2]
    xa, ya = max(0, x1 - ring_width), max(0, y1 - ring_width)
    xb, yb = min(width, x2 + ring_width + 1), min(height, y2 + ring_width + 1)
    patch = image[ya:yb, xa:xb, 0]
    ring_mask = np.ones(patch.shape, bool)
    ring_mask[y1 - ya : y2 - ya + 1, x1 - xa : x2 - xa + 1] = False
    ring = patch[ring_mask]
    return ring if ring.size else image[:, :, 0].ravel()


def remove_target(image, box, variant, seed):
    mask, rectangle = expanded_mask(box, image.shape[:2], variant["margin"])
    method = variant["method"]
    if method in {"telea", "navier_stokes"}:
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        channels = [cv2.inpaint(image[:, :, c], mask, 7, flag) for c in range(3)]
        return np.stack(channels, axis=2).astype(np.float32)

    output = image.copy()
    x1, y1, x2, y2 = rectangle
    ring = local_ring(image, rectangle)
    if method == "local_noise":
        median = float(np.median(ring))
        sigma = max(float(np.median(np.abs(ring - median)) * 1.4826), 1.0)
        rng = np.random.default_rng(seed)
        fill = rng.normal(median, sigma, size=(y2 - y1 + 1, x2 - x1 + 1, 1))
        output[y1 : y2 + 1, x1 : x2 + 1] = np.clip(fill, 0, 255)
    else:
        output[y1 : y2 + 1, x1 : x2 + 1] = float(np.median(ring))
    return output.astype(np.float32)


def split_target(boxes_pred, scores_pred, target):
    if len(boxes_pred) == 0:
        return 0.0, np.empty((0, 4), np.float32), np.empty(0, np.float32)
    overlap = pairwise_iou(boxes_pred, [target])[:, 0]
    target_score = float(scores_pred[overlap >= CONFIG["target_overlap_threshold"]].max(initial=0))
    keep = overlap < CONFIG["target_overlap_threshold"]
    return target_score, boxes_pred[keep], scores_pred[keep]

# %%
baseline = {}
for image_id in tqdm(sorted(images), desc="original predictions"):
    predicted_boxes, predicted_scores = predict(images[image_id])
    target_score, non_target_boxes, non_target_scores = split_target(
        predicted_boxes,
        predicted_scores,
        boxes[image_id],
    )
    baseline[image_id] = {
        "target_score": target_score,
        "non_target_boxes": non_target_boxes,
        "non_target_scores": non_target_scores,
    }

rows = []
new_detection_rows = []
overlay_tiles = []
for variant_index, variant in enumerate(VARIANTS):
    log(f"START {variant_index + 1}/{len(VARIANTS)} {variant['name']}")
    for image_id in tqdm(sorted(images), desc=variant["name"], leave=False):
        removed = remove_target(
            images[image_id],
            boxes[image_id],
            variant,
            SEED + variant_index * 1000 + image_id,
        )
        predicted_boxes, predicted_scores = predict(removed)
        target_score, non_target_boxes, non_target_scores = split_target(
            predicted_boxes,
            predicted_scores,
            boxes[image_id],
        )
        original_non_target = baseline[image_id]["non_target_boxes"]
        if len(non_target_boxes) == 0:
            maximum_original_iou = np.empty(0, np.float32)
        elif len(original_non_target) == 0:
            maximum_original_iou = np.zeros(len(non_target_boxes), np.float32)
        else:
            maximum_original_iou = pairwise_iou(non_target_boxes, original_non_target).max(axis=1)
        is_new = maximum_original_iou < CONFIG["new_detection_match_iou"]
        rows.append(
            {
                "image_id": image_id,
                "variant": variant["name"],
                "original_target_score": baseline[image_id]["target_score"],
                "removed_target_score": target_score,
                "target_score_drop": baseline[image_id]["target_score"] - target_score,
                "original_non_target_count": len(original_non_target),
                "removed_non_target_count": len(non_target_boxes),
                "new_non_target_count": int(is_new.sum()),
                "maximum_new_score": float(non_target_scores[is_new].max(initial=0)),
            }
        )
        for box, score, new_flag, original_iou in zip(
            non_target_boxes,
            non_target_scores,
            is_new,
            maximum_original_iou,
        ):
            if new_flag:
                new_detection_rows.append(
                    {
                        "image_id": image_id,
                        "variant": variant["name"],
                        "score": float(score),
                        "max_original_iou": float(original_iou),
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    }
                )

        if variant["name"] == "telea_m8":
            gray = cv2.normalize(removed[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            tile = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            x1, y1, x2, y2 = [int(round(value)) for value in boxes[image_id]]
            cv2.rectangle(tile, (x1, y1), (x2, y2), (0, 255, 255), 2)
            for box, score, new_flag in zip(non_target_boxes, non_target_scores, is_new):
                color = (0, 0, 255) if new_flag else (0, 255, 0)
                xa, ya, xb, yb = [int(round(value)) for value in box]
                cv2.rectangle(tile, (xa, ya), (xb, yb), color, 2)
                cv2.putText(
                    tile,
                    f"{score:.2f}",
                    (xa, max(16, ya - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                tile,
                f"id {image_id}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            overlay_tiles.append(cv2.resize(tile, (256, 256), interpolation=cv2.INTER_AREA))
    partial = pd.DataFrame(rows)
    partial.to_csv(OUT / "missed_detection_results.partial.csv", index=False)
    current = partial[partial.variant == variant["name"]]
    log(
        f"DONE {variant['name']}: median target drop={current.target_score_drop.median():.3f}, "
        f"images with new boxes={(current.new_non_target_count > 0).mean():.3f}"
    )

results = pd.DataFrame(rows)
new_detections = pd.DataFrame(new_detection_rows)
results.to_csv(OUT / "missed_detection_results.csv", index=False)
new_detections.to_csv(OUT / "new_detection_candidates.csv", index=False)
summary = (
    results.groupby("variant", sort=False)
    .agg(
        n=("image_id", "size"),
        median_original_target_score=("original_target_score", "median"),
        median_removed_target_score=("removed_target_score", "median"),
        median_target_score_drop=("target_score_drop", "median"),
        mean_original_non_target_count=("original_non_target_count", "mean"),
        mean_removed_non_target_count=("removed_non_target_count", "mean"),
        images_with_new_detection_rate=("new_non_target_count", lambda values: (values > 0).mean()),
        total_new_detections=("new_non_target_count", "sum"),
        maximum_new_score=("maximum_new_score", "max"),
    )
    .reset_index()
)
summary.to_csv(OUT / "missed_detection_summary.csv", index=False)

montage_rows = []
for start in range(0, len(overlay_tiles), 5):
    row_tiles = overlay_tiles[start : start + 5]
    while len(row_tiles) < 5:
        row_tiles.append(np.zeros_like(overlay_tiles[0]))
    montage_rows.append(np.hstack(row_tiles))
cv2.imwrite(str(OUT / "telea_m8_detection_audit.png"), np.vstack(montage_rows))

report = {
    "device": cfg.MODEL.DEVICE,
    "n_images": len(images),
    "n_variants": len(VARIANTS),
    "summary": summary.set_index("variant").to_dict(orient="index"),
    "interpretation_guard": (
        "New model boxes are only public-unlearn-set audit candidates. "
        "They are not labeled as genuine streaks and no test image is used."
    ),
}
(OUT / "missed_detection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log(summary.to_string(index=False))
log(json.dumps(report, indent=2))
log("P2.02 missed-detection closure batch complete")

