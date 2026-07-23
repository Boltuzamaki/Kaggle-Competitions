# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Debris behavior batch - P2.04 to P2.06 and P1.07
#
# This preregistered, rule-safe batch uses only the 20 public unlearn images and
# the supplied poisoned RetinaNet. It never reads or predicts a test image.
#
# The batch measures confidence and bounding-box equivariance under:
#
# 1. four 32-pixel translations,
# 2. two small rotations,
# 3. two blur strengths and sharpening,
# 4. four image scales, and
# 5. 32-pixel border replacement.

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
OUT = Path("/kaggle/working/behavior_forensics")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260717
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EXPERIMENTS = [
    {"name": "identity", "family": "baseline"},
    {"name": "translate_xm32", "family": "translation", "dx": -32, "dy": 0},
    {"name": "translate_xp32", "family": "translation", "dx": 32, "dy": 0},
    {"name": "translate_ym32", "family": "translation", "dx": 0, "dy": -32},
    {"name": "translate_yp32", "family": "translation", "dx": 0, "dy": 32},
    {"name": "rotate_m5", "family": "rotation", "angle": -5},
    {"name": "rotate_p5", "family": "rotation", "angle": 5},
    {"name": "blur_sigma1", "family": "appearance", "sigma": 1.0},
    {"name": "blur_sigma2", "family": "appearance", "sigma": 2.0},
    {"name": "sharpen", "family": "appearance"},
    {"name": "scale_075", "family": "scale", "scale": 0.75},
    {"name": "scale_090", "family": "scale", "scale": 0.90},
    {"name": "scale_110", "family": "scale", "scale": 1.10},
    {"name": "scale_125", "family": "scale", "scale": 1.25},
    {"name": "border_replace32", "family": "border", "border": 32},
]
CONFIG = {
    "seed": SEED,
    "score_fire_threshold": 0.20,
    "minimum_match_iou": 0.10,
    "experiments": EXPERIMENTS,
    "guard": (
        "All transformations, thresholds, and summaries were fixed before output inspection. "
        "Only public unlearn images and the supplied poisoned model are used."
    ),
}
(OUT / "experiment_config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
log(f"Predeclared {len(EXPERIMENTS)} variants; config saved")

with (UNLEARN / "annotations_coco.json").open() as handle:
    coco = json.load(handle)
image_info = {int(item["id"]): item for item in coco["images"]}
annotation_by_image = {int(item["image_id"]): item for item in coco["annotations"]}
assert len(image_info) == len(annotation_by_image) == 20


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    else:
        image = image.astype(np.float32) / max(float(np.iinfo(image.dtype).max), 1.0)
    image = np.clip(image * 255.0, 0, 255)
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


def box_iou(box, candidates):
    candidates = np.asarray(candidates, np.float32).reshape(-1, 4)
    if len(candidates) == 0:
        return np.zeros(0, np.float32)
    top_left = np.maximum(box[None, :2], candidates[:, :2])
    bottom_right = np.minimum(box[None, 2:], candidates[:, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[:, 0] * size[:, 1]
    area_a = max(float((box[2] - box[0]) * (box[3] - box[1])), 1e-6)
    area_b = np.clip(
        (candidates[:, 2] - candidates[:, 0]) * (candidates[:, 3] - candidates[:, 1]),
        1e-6,
        None,
    )
    return intersection / np.clip(area_a + area_b - intersection, 1e-6, None)


def transform_box(box, matrix, width, height):
    corners = np.asarray(
        [
            [box[0], box[1], 1],
            [box[2], box[1], 1],
            [box[2], box[3], 1],
            [box[0], box[3], 1],
        ],
        np.float64,
    )
    projected = corners @ np.asarray(matrix, np.float64).T
    projected = projected[:, :2] / np.clip(projected[:, 2:3], 1e-12, None)
    result = np.asarray(
        [
            projected[:, 0].min(),
            projected[:, 1].min(),
            projected[:, 0].max(),
            projected[:, 1].max(),
        ],
        np.float32,
    )
    result[[0, 2]] = np.clip(result[[0, 2]], 0, width)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, height)
    return result


def match_prediction(image, target_box):
    output = predictor(image)["instances"].to("cpu")
    predicted_boxes = output.pred_boxes.tensor.numpy()
    predicted_scores = output.scores.numpy()
    overlaps = box_iou(np.asarray(target_box, np.float32), predicted_boxes)
    valid = np.flatnonzero(overlaps >= CONFIG["minimum_match_iou"])
    if valid.size == 0:
        return 0.0, 0.0, np.full(4, np.nan, np.float32)
    index = int(valid[np.argmax(predicted_scores[valid])])
    return (
        float(predicted_scores[index]),
        float(overlaps[index]),
        predicted_boxes[index].astype(np.float32),
    )


def apply_experiment(image, target_box, experiment):
    height, width = image.shape[:2]
    family = experiment["family"]
    matrix = np.eye(3, dtype=np.float64)
    output = image.copy()

    if family == "translation":
        matrix[0, 2] = experiment["dx"]
        matrix[1, 2] = experiment["dy"]
        output = cv2.warpAffine(
            image,
            matrix[:2],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    elif family == "rotation":
        affine = cv2.getRotationMatrix2D(
            ((width - 1) / 2, (height - 1) / 2),
            experiment["angle"],
            1.0,
        )
        matrix[:2] = affine
        output = cv2.warpAffine(
            image,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    elif family == "appearance":
        if experiment["name"].startswith("blur"):
            output = cv2.GaussianBlur(image, (0, 0), experiment["sigma"])
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
            output = np.clip(image * 1.6 - blurred * 0.6, 0, 255)
    elif family == "scale":
        scale = experiment["scale"]
        output = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA,
        )
        matrix[0, 0] = scale
        matrix[1, 1] = scale
    elif family == "border":
        border = experiment["border"]
        interior = image[border:-border, border:-border]
        output = cv2.copyMakeBorder(
            interior,
            border,
            border,
            border,
            border,
            cv2.BORDER_REFLECT_101,
        )

    out_height, out_width = output.shape[:2]
    transformed_target = transform_box(target_box, matrix, out_width, out_height)
    return output.astype(np.float32), transformed_target, matrix


def box_geometry(box, reference):
    if not np.isfinite(box).all():
        return {
            "center_error_norm": np.nan,
            "width_ratio": np.nan,
            "height_ratio": np.nan,
            "area_ratio": np.nan,
        }
    center = (box[:2] + box[2:]) / 2
    ref_center = (reference[:2] + reference[2:]) / 2
    ref_size = np.clip(reference[2:] - reference[:2], 1e-6, None)
    size = np.clip(box[2:] - box[:2], 0, None)
    return {
        "center_error_norm": float(np.linalg.norm(center - ref_center) / np.linalg.norm(ref_size)),
        "width_ratio": float(size[0] / ref_size[0]),
        "height_ratio": float(size[1] / ref_size[1]),
        "area_ratio": float(np.prod(size) / np.prod(ref_size)),
    }

# %%
baseline_predictions = {}
for image_id in tqdm(sorted(images), desc="baseline"):
    score, overlap, predicted_box = match_prediction(images[image_id], boxes[image_id])
    baseline_predictions[image_id] = {
        "score": score,
        "iou": overlap,
        "box": predicted_box,
    }

rows = []
for experiment_index, experiment in enumerate(EXPERIMENTS):
    log(
        f"START {experiment_index + 1:02d}/{len(EXPERIMENTS):02d} "
        f"{experiment['name']}"
    )
    for image_id in tqdm(sorted(images), desc=experiment["name"], leave=False):
        transformed, target, matrix = apply_experiment(
            images[image_id],
            boxes[image_id],
            experiment,
        )
        score, matched_iou, predicted_box = match_prediction(transformed, target)
        if np.isfinite(predicted_box).all():
            inverse = np.linalg.inv(matrix)
            backmapped = transform_box(
                predicted_box,
                inverse,
                images[image_id].shape[1],
                images[image_id].shape[0],
            )
        else:
            backmapped = np.full(4, np.nan, np.float32)
        baseline = baseline_predictions[image_id]
        geometry = box_geometry(predicted_box, target)
        baseline_equivariance_iou = (
            float(box_iou(baseline["box"], [backmapped])[0])
            if np.isfinite(baseline["box"]).all() and np.isfinite(backmapped).all()
            else np.nan
        )
        backmap_geometry = box_geometry(backmapped, baseline["box"])
        target_size = target[2:] - target[:2]
        valid_target = bool(np.all(target_size > 1))
        rows.append(
            {
                "image_id": image_id,
                "variant": experiment["name"],
                "family": experiment["family"],
                "score": score,
                "score_ratio": score / max(baseline["score"], 1e-9),
                "fired_020": score >= CONFIG["score_fire_threshold"],
                "matched_target_iou": matched_iou,
                "valid_target": valid_target,
                "equivariance_iou": baseline_equivariance_iou,
                "equivariance_center_shift_norm": backmap_geometry["center_error_norm"],
                **geometry,
            }
        )
    partial = pd.DataFrame(rows)
    partial.to_csv(OUT / "behavior_results.partial.csv", index=False)
    current = partial[partial.variant == experiment["name"]]
    log(
        f"DONE {experiment['name']}: fire={current.fired_020.mean():.3f}, "
        f"median_score_ratio={current.score_ratio.median():.3f}, "
        f"median_target_iou={current.matched_target_iou.median():.3f}, "
        f"median_equivariance_iou={current.equivariance_iou.median():.3f}"
    )

results = pd.DataFrame(rows)
results.to_csv(OUT / "behavior_results.csv", index=False)
summary = (
    results.groupby(["family", "variant"], sort=False)
    .agg(
        n=("score", "size"),
        fire_rate_020=("fired_020", "mean"),
        median_score=("score", "median"),
        median_score_ratio=("score_ratio", "median"),
        median_target_iou=("matched_target_iou", "median"),
        median_equivariance_iou=("equivariance_iou", "median"),
        median_equivariance_center_shift=("equivariance_center_shift_norm", "median"),
        median_width_ratio=("width_ratio", "median"),
        median_height_ratio=("height_ratio", "median"),
        median_area_ratio=("area_ratio", "median"),
    )
    .reset_index()
)
summary.to_csv(OUT / "behavior_summary.csv", index=False)

# %%
plot_frame = summary[summary.variant != "identity"].copy()
figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
colors = plot_frame["family"].map(
    {
        "translation": "#62e7b4",
        "rotation": "#7aa2f7",
        "appearance": "#f4b860",
        "scale": "#bb9af7",
        "border": "#ff6b6b",
    }
)
axes[0].bar(plot_frame["variant"].astype(str), plot_frame["median_score_ratio"], color=colors)
axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
axes[0].set_ylabel("Median score ratio vs identity")
axes[0].grid(axis="y", alpha=0.2)
axes[1].bar(
    plot_frame["variant"].astype(str),
    plot_frame["median_equivariance_iou"],
    color=colors,
)
axes[1].set_ylabel("Median back-mapped box IoU")
axes[1].set_ylim(0, 1)
axes[1].grid(axis="y", alpha=0.2)
axes[1].tick_params(axis="x", rotation=45)
figure.suptitle("P2 behavior stability under predeclared transformations")
figure.tight_layout()
figure.savefig(OUT / "behavior_stability.png", dpi=180)
plt.close(figure)

translation_rotation = summary[summary.family.isin(["translation", "rotation"])]
scales = summary[summary.family == "scale"]
border = summary[summary.family == "border"]
report = {
    "device": cfg.MODEL.DEVICE,
    "n_images": len(images),
    "n_variants": len(EXPERIMENTS),
    "checks": {
        "P2.04_box_movement": {
            "minimum_median_equivariance_iou": float(
                translation_rotation.median_equivariance_iou.min()
            ),
            "maximum_median_center_shift_norm": float(
                translation_rotation.median_equivariance_center_shift.max()
            ),
        },
        "P2.05_transformation_sensitivity": translation_rotation[
            [
                "variant",
                "fire_rate_020",
                "median_score_ratio",
                "median_equivariance_iou",
            ]
        ].to_dict(orient="records"),
        "P2.06_scale_sensitivity": scales[
            [
                "variant",
                "fire_rate_020",
                "median_score_ratio",
                "median_target_iou",
            ]
        ].to_dict(orient="records"),
        "P1.07_border_replacement": border[
            ["variant", "fire_rate_020", "median_score_ratio", "median_target_iou"]
        ].to_dict(orient="records"),
    },
    "guard": (
        "Transformation consistency describes the poisoned model's learned response. "
        "It does not estimate hidden-clean-model labels and uses no test images."
    ),
}
(OUT / "behavior_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log(summary.to_string(index=False))
log(json.dumps(report, indent=2))
log("P2 behavior batch complete")

