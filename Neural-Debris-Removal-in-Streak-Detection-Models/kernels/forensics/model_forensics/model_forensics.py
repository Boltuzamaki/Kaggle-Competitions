# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Debris - Poison Attack Forensics
#
# This research-only notebook uses **only the 20 public unlearn images** and the
# provided poisoned RetinaNet. It does not inspect, label, or predict the test set.
#
# Tests:
#
# 1. Does removing the annotated streak suppress the poisoned detection?
# 2. Does the streak still fire when transplanted into another unlearn image?
# 3. Does it survive D4 rotation/reflection?
# 4. Does destroying spatial structure while preserving the pixel histogram stop it?

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

log("Installing a PyTorch CUDA build compatible with Kaggle P100 GPUs")
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
log("Installing Detectron2")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "setuptools<81"], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"],
    check=True,
)

# %%
import cv2
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm.auto import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
UNLEARN = ROOT / "unlearn_set"
WEIGHTS = ROOT / "poisoned_model/poisoned_model.pth"
OUT = Path("/kaggle/working/forensics")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260717
rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

with (UNLEARN / "annotations_coco.json").open() as f:
    coco = json.load(f)
image_info = {int(x["id"]): x for x in coco["images"]}
ann_by_id = {int(x["image_id"]): x for x in coco["annotations"]}
assert len(image_info) == len(ann_by_id) == 20

def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    image = np.clip(image * 255.0, 0, 255).astype(np.float32)
    return np.repeat(image[:, :, None], 3, axis=2)

images = {image_id: load_image(UNLEARN / info["file_name"]) for image_id, info in image_info.items()}
boxes = {}
for image_id, ann in ann_by_id.items():
    x, y, w, h = map(float, ann["bbox"])
    boxes[image_id] = np.asarray([x, y, x + w, y + h], np.float32)

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

def iou(box, candidates):
    if len(candidates) == 0:
        return np.zeros(0, np.float32)
    tl = np.maximum(box[None, :2], candidates[:, :2])
    br = np.minimum(box[None, 2:], candidates[:, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (candidates[:, 2] - candidates[:, 0]) * (candidates[:, 3] - candidates[:, 1])
    return inter / np.clip(a + b - inter, 1e-6, None)

def score_at(image, target_box, threshold=0.2):
    out = predictor(image)["instances"].to("cpu")
    pred_boxes = out.pred_boxes.tensor.numpy()
    pred_scores = out.scores.numpy()
    overlaps = iou(np.asarray(target_box, np.float32), pred_boxes)
    valid = np.where(overlaps >= threshold)[0]
    if len(valid) == 0:
        return 0.0, 0.0
    j = valid[np.argmax(pred_scores[valid])]
    return float(pred_scores[j]), float(overlaps[j])

def inpaint_box(image, box, margin=4):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(
        mask,
        (max(0, x1 - margin), max(0, y1 - margin)),
        (min(1023, x2 + margin), min(1023, y2 + margin)),
        255,
        -1,
    )
    channels = [cv2.inpaint(image[:, :, c], mask, 7, cv2.INPAINT_TELEA) for c in range(3)]
    return np.stack(channels, axis=2).astype(np.float32)

def crop_patch(image, box, margin=8):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    xa, ya = max(0, x1 - margin), max(0, y1 - margin)
    xb, yb = min(1024, x2 + margin), min(1024, y2 + margin)
    patch = image[ya:yb, xa:xb].copy()
    inner = np.asarray([x1 - xa, y1 - ya, x2 - xa, y2 - ya], np.float32)
    return patch, inner

def transform_patch(patch, inner, k, flip):
    out = np.rot90(patch, k=k).copy() if k else patch.copy()
    H, W = patch.shape[:2]
    x1, y1, x2, y2 = inner
    if k == 1:
        inner = np.asarray([y1, W - x2, y2, W - x1], np.float32)
    elif k == 2:
        inner = np.asarray([W - x2, H - y2, W - x1, H - y1], np.float32)
    elif k == 3:
        inner = np.asarray([H - y2, x1, H - y1, x2], np.float32)
    if flip:
        out = out[:, ::-1].copy()
        width = out.shape[1]
        inner = np.asarray([width - inner[2], inner[1], width - inner[0], inner[3]], np.float32)
    return out, inner

def paste_patch(host, patch, inner, forbidden_box):
    ph, pw = patch.shape[:2]
    for _ in range(100):
        x0 = int(rng.integers(16, 1024 - pw - 16))
        y0 = int(rng.integers(16, 1024 - ph - 16))
        target = inner + np.asarray([x0, y0, x0, y0], np.float32)
        if iou(target, np.asarray([forbidden_box], np.float32))[0] < 0.01:
            result = host.copy()
            feather = min(4, ph // 4, pw // 4)
            yy = np.minimum(np.arange(ph) + 1, np.arange(ph)[::-1] + 1) / max(feather, 1)
            xx = np.minimum(np.arange(pw) + 1, np.arange(pw)[::-1] + 1) / max(feather, 1)
            alpha = np.clip(np.minimum(yy[:, None], xx[None, :]), 0, 1)[..., None]
            region = result[y0 : y0 + ph, x0 : x0 + pw]
            result[y0 : y0 + ph, x0 : x0 + pw] = region * (1 - alpha) + patch * alpha
            return result, target
    raise RuntimeError("Could not find a free transplant location")

# %% [markdown]
# ## A. Local necessity: remove the annotated streak

# %%
ablation_rows = []
for image_id in tqdm(sorted(images), desc="necessity"):
    original_score, original_iou = score_at(images[image_id], boxes[image_id])
    removed = inpaint_box(images[image_id], boxes[image_id])
    removed_score, removed_iou = score_at(removed, boxes[image_id])
    ablation_rows.append(
        {
            "image_id": image_id,
            "original_score": original_score,
            "original_iou": original_iou,
            "removed_score": removed_score,
            "removed_iou": removed_iou,
            "score_drop": original_score - removed_score,
        }
    )
ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(OUT / "local_necessity.csv", index=False)
print(ablation_df.to_string(index=False))

# %% [markdown]
# ## B. Local sufficiency and structure tests

# %%
variants = ["original", "d4", "pixel_shuffled", "blurred"]
transplant_rows = []
image_ids = sorted(images)
for source_id in tqdm(image_ids, desc="transplants"):
    patch, inner = crop_patch(images[source_id], boxes[source_id])
    hosts = [i for i in image_ids if i != source_id]
    rng.shuffle(hosts)
    for repeat, host_id in enumerate(hosts[:4]):
        for variant in variants:
            p, b = patch.copy(), inner.copy()
            if variant == "d4":
                p, b = transform_patch(p, b, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
            elif variant == "pixel_shuffled":
                flat = p.reshape(-1, 3).copy()
                rng.shuffle(flat)
                p = flat.reshape(p.shape)
            elif variant == "blurred":
                p = cv2.GaussianBlur(p, (0, 0), 3.0)
            composite, target = paste_patch(images[host_id], p, b, boxes[host_id])
            score, matched_iou = score_at(composite, target)
            transplant_rows.append(
                {
                    "source_id": source_id,
                    "host_id": host_id,
                    "repeat": repeat,
                    "variant": variant,
                    "score": score,
                    "iou": matched_iou,
                    "fired_005": score >= 0.05,
                    "fired_020": score >= 0.20,
                }
            )

transplant_df = pd.DataFrame(transplant_rows)
transplant_df.to_csv(OUT / "transplant_results.csv", index=False)
summary = (
    transplant_df.groupby("variant")
    .agg(
        n=("score", "size"),
        fire_rate_005=("fired_005", "mean"),
        fire_rate_020=("fired_020", "mean"),
        median_score=("score", "median"),
        mean_score=("score", "mean"),
        median_iou=("iou", "median"),
    )
    .reset_index()
)
summary.to_csv(OUT / "transplant_summary.csv", index=False)
print(summary.to_string(index=False))

# %%
orig = transplant_df[transplant_df.variant == "original"].sort_values(["source_id", "host_id"])
d4 = transplant_df[transplant_df.variant == "d4"].sort_values(["source_id", "host_id"])
shuffled = transplant_df[transplant_df.variant == "pixel_shuffled"].sort_values(["source_id", "host_id"])
blurred = transplant_df[transplant_df.variant == "blurred"].sort_values(["source_id", "host_id"])

def paired_p(a, b, alternative):
    try:
        return float(stats.wilcoxon(a, b, alternative=alternative).pvalue)
    except ValueError:
        return 1.0

report = {
    "device": cfg.MODEL.DEVICE,
    "counts": {"unlearn_images": 20, "transplant_trials_per_variant": int(len(orig))},
    "local_necessity": {
        "original_fire_rate_020": float((ablation_df.original_score >= 0.20).mean()),
        "removed_fire_rate_020": float((ablation_df.removed_score >= 0.20).mean()),
        "original_median_score": float(ablation_df.original_score.median()),
        "removed_median_score": float(ablation_df.removed_score.median()),
        "median_score_drop": float(ablation_df.score_drop.median()),
        "paired_drop_p": paired_p(ablation_df.original_score, ablation_df.removed_score, "greater"),
    },
    "transplant_summary": summary.set_index("variant").to_dict(orient="index"),
    "paired_structure_tests": {
        "original_greater_than_shuffled_p": paired_p(orig.score.to_numpy(), shuffled.score.to_numpy(), "greater"),
        "original_greater_than_blurred_p": paired_p(orig.score.to_numpy(), blurred.score.to_numpy(), "greater"),
        "original_vs_d4_two_sided_p": paired_p(orig.score.to_numpy(), d4.score.to_numpy(), "two-sided"),
    },
    "interpretation_guard": (
        "Firing after transplant proves the poisoned model uses local crop content. "
        "It does not alone reveal how the hidden clean model would score the same crop."
    ),
}

# %% [markdown]
# ## C. P1.03 saliency localization
#
# The attribution target is predeclared as the highest-logit decoded anchor
# overlapping the supplied poison box by IoU >= 0.2. We measure both raw
# input-gradient attribution and an FPN Grad-CAM map. No test image is used.

# %%
def normalize_mass(values):
    values = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    return values / total if total > 0 else np.full(values.shape, 1.0 / values.size)


def map_statistics(values, target_box):
    mass = normalize_mass(values)
    height, width = mass.shape
    x1, y1, x2, y2 = [int(round(v)) for v in target_box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    target_mass = float(mass[y1:y2, x1:x2].sum())
    area_fraction = float(max((x2 - x1) * (y2 - y1), 1) / mass.size)

    box_w, box_h = max(x2 - x1, 1), max(y2 - y1, 1)
    corner_boxes = [
        (0, 0, box_w, box_h),
        (width - box_w, 0, width, box_h),
        (0, height - box_h, box_w, height),
        (width - box_w, height - box_h, width, height),
    ]
    corner_masses = [float(mass[ya:yb, xa:xb].sum()) for xa, ya, xb, yb in corner_boxes]
    yy, xx = np.indices(mass.shape)
    centroid_x = float((xx * mass).sum())
    centroid_y = float((yy * mass).sum())
    box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
    diagonal = max(float(np.hypot(width, height)), 1.0)
    return {
        "box_mass": target_mass,
        "box_area_fraction": area_fraction,
        "box_enrichment": target_mass / max(area_fraction, 1e-12),
        "max_equal_area_corner_mass": max(corner_masses),
        "box_to_max_corner_mass_ratio": target_mass / max(max(corner_masses), 1e-12),
        "attribution_centroid_distance_normalized": float(
            np.hypot(centroid_x - box_cx, centroid_y - box_cy) / diagonal
        ),
    }


def target_saliency(image, target_box):
    model = predictor.model
    model.eval()
    model.zero_grad(set_to_none=True)

    transform = predictor.aug.get_transform(image)
    resized = transform.apply_image(image)
    resized_box = transform.apply_box(np.asarray([target_box], np.float32))[0]
    input_tensor = torch.as_tensor(
        np.ascontiguousarray(resized.transpose(2, 0, 1)), dtype=torch.float32, device=model.device
    )
    input_tensor.requires_grad_(True)
    batched = [{"image": input_tensor, "height": image.shape[0], "width": image.shape[1]}]
    image_list = model.preprocess_image(batched)
    feature_dict = model.backbone(image_list.tensor)
    feature_list = [feature_dict[name] for name in model.head_in_features]
    for feature in feature_list:
        feature.retain_grad()

    predictions = model.head(feature_list)
    logits_by_level, deltas_by_level = model._transpose_dense_predictions(
        predictions, [model.num_classes, 4]
    )
    anchors_by_level = model.anchor_generator(feature_list)

    selected = None
    for level, (logits, deltas, anchors) in enumerate(
        zip(logits_by_level, deltas_by_level, anchors_by_level)
    ):
        decoded = model.box2box_transform.apply_deltas(deltas[0], anchors.tensor)
        overlaps = iou(resized_box, decoded.detach().cpu().numpy())
        scores = logits[0, :, 0].detach().sigmoid().cpu().numpy()
        valid = np.flatnonzero(overlaps >= 0.20)
        if valid.size:
            index = int(valid[np.argmax(scores[valid])])
            candidate = (float(scores[index]), level, index, float(overlaps[index]))
            if selected is None or candidate[0] > selected[0]:
                selected = candidate

    if selected is None:
        raise RuntimeError("No decoded anchor overlaps the poison box at IoU >= 0.20")

    selected_score, selected_level, selected_index, selected_iou = selected
    target_logit = logits_by_level[selected_level][0, selected_index, 0]
    target_logit.backward()

    input_map = input_tensor.grad.detach().abs().mean(dim=0).cpu().numpy()
    input_map = cv2.resize(input_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    feature = feature_list[selected_level]
    gradient = feature.grad[0]
    activation = feature.detach()[0]
    weights = gradient.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=0)).cpu().numpy()
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return input_map, cam, selected_score, selected_iou, selected_level


saliency_rows = []
saliency_tiles = []
for image_id in tqdm(sorted(images), desc="saliency"):
    input_map, cam, raw_score, matched_iou, selected_level = target_saliency(
        images[image_id], boxes[image_id]
    )
    input_stats = map_statistics(input_map, boxes[image_id])
    cam_stats = map_statistics(cam, boxes[image_id])
    row = {
        "image_id": image_id,
        "raw_anchor_score": raw_score,
        "decoded_anchor_iou": matched_iou,
        "fpn_level": int(selected_level + 3),
    }
    row.update({f"input_{key}": value for key, value in input_stats.items()})
    row.update({f"cam_{key}": value for key, value in cam_stats.items()})
    saliency_rows.append(row)

    base = cv2.normalize(images[image_id][:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    heat = (normalize_mass(cam) / max(normalize_mass(cam).max(), 1e-12) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), 0.60, heat, 0.40, 0)
    x1, y1, x2, y2 = [int(round(value)) for value in boxes[image_id]]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        overlay,
        f"id {image_id} P{selected_level + 3}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    saliency_tiles.append(cv2.resize(overlay, (256, 256), interpolation=cv2.INTER_AREA))

saliency_df = pd.DataFrame(saliency_rows)
saliency_df.to_csv(OUT / "saliency_results.csv", index=False)
rows = []
for start in range(0, len(saliency_tiles), 5):
    row_tiles = saliency_tiles[start : start + 5]
    while len(row_tiles) < 5:
        row_tiles.append(np.zeros_like(saliency_tiles[0]))
    rows.append(np.hstack(row_tiles))
cv2.imwrite(str(OUT / "saliency_montage.png"), np.vstack(rows))

report["saliency"] = {
    "n": int(len(saliency_df)),
    "input_gradient_box_enrichment_median": float(saliency_df.input_box_enrichment.median()),
    "input_gradient_box_to_corner_ratio_median": float(
        saliency_df.input_box_to_max_corner_mass_ratio.median()
    ),
    "gradcam_box_enrichment_median": float(saliency_df.cam_box_enrichment.median()),
    "gradcam_box_to_corner_ratio_median": float(
        saliency_df.cam_box_to_max_corner_mass_ratio.median()
    ),
    "gradcam_box_enrichment_min": float(saliency_df.cam_box_enrichment.min()),
    "fpn_level_counts": {
        str(int(level)): int(count)
        for level, count in saliency_df.fpn_level.value_counts().sort_index().items()
    },
    "guard": (
        "Saliency localizes model sensitivity but is not alone a causal proof. "
        "The separate removal and transplant tests supply the causal evidence."
    ),
}

# %% [markdown]
# ## D. P1.04 coarse-to-fine occlusion sensitivity
#
# Stage 1 replaces each non-overlapping 128x128 tile with local background.
# Stage 2 tests 64x64 windows at stride 32 around the strongest coarse tile.
# The declared endpoint is whether the strongest confidence drop overlaps the
# supplied poison box and exceeds the typical non-overlapping-mask effect.

# %%
def replace_window(image, window, seed):
    x1, y1, x2, y2 = [int(round(value)) for value in window]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    margin = 12
    xa, ya = max(0, x1 - margin), max(0, y1 - margin)
    xb, yb = min(image.shape[1], x2 + margin), min(image.shape[0], y2 + margin)
    context = image[ya:yb, xa:xb, 0]
    ring_mask = np.ones(context.shape, bool)
    ring_mask[y1 - ya : y2 - ya, x1 - xa : x2 - xa] = False
    ring = context[ring_mask]
    median = float(np.median(ring))
    mad = float(np.median(np.abs(ring - median)) * 1.4826)
    local_rng = np.random.default_rng(seed)
    replacement = local_rng.normal(median, max(mad, 1.0), size=(y2 - y1, x2 - x1))
    output = image.copy()
    output[y1:y2, x1:x2, :] = np.clip(replacement[:, :, None], 0, 255)
    return output


def target_coverage(window, target):
    x1, y1 = max(window[0], target[0]), max(window[1], target[1])
    x2, y2 = min(window[2], target[2]), min(window[3], target[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    target_area = max((target[2] - target[0]) * (target[3] - target[1]), 1e-6)
    return float(intersection / target_area)


occlusion_rows = []
occlusion_tiles = []
for image_id in tqdm(sorted(images), desc="occlusion"):
    image = images[image_id]
    target = boxes[image_id]
    baseline, _ = score_at(image, target)
    coarse = np.zeros((8, 8), np.float32)
    coarse_records = []
    for grid_y, y1 in enumerate(range(0, 1024, 128)):
        for grid_x, x1 in enumerate(range(0, 1024, 128)):
            window = np.asarray([x1, y1, x1 + 128, y1 + 128], np.float32)
            masked = replace_window(image, window, SEED + image_id * 10000 + y1 * 10 + x1)
            score, _ = score_at(masked, target)
            drop = baseline - score
            coarse[grid_y, grid_x] = drop
            coarse_records.append(
                {
                    "window": window,
                    "drop": drop,
                    "coverage": target_coverage(window, target),
                }
            )

    best_coarse = max(coarse_records, key=lambda item: item["drop"])
    bx1, by1, _, _ = best_coarse["window"]
    refine_x = sorted(
        set(int(np.clip(x, 0, 960)) for x in range(int(bx1) - 64, int(bx1) + 129, 32))
    )
    refine_y = sorted(
        set(int(np.clip(y, 0, 960)) for y in range(int(by1) - 64, int(by1) + 129, 32))
    )
    fine_records = []
    for y1 in refine_y:
        for x1 in refine_x:
            window = np.asarray([x1, y1, x1 + 64, y1 + 64], np.float32)
            masked = replace_window(image, window, SEED + 7 + image_id * 10000 + y1 * 10 + x1)
            score, _ = score_at(masked, target)
            fine_records.append(
                {
                    "window": window,
                    "drop": baseline - score,
                    "coverage": target_coverage(window, target),
                }
            )
    best_fine = max(fine_records, key=lambda item: item["drop"])
    nonoverlap_drops = [
        item["drop"] for item in coarse_records + fine_records if item["coverage"] == 0
    ]
    target_cx, target_cy = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
    peak = best_fine["window"]
    peak_cx, peak_cy = (peak[0] + peak[2]) / 2, (peak[1] + peak[3]) / 2
    occlusion_rows.append(
        {
            "image_id": image_id,
            "baseline_score": baseline,
            "max_drop": best_fine["drop"],
            "peak_target_coverage": best_fine["coverage"],
            "peak_overlaps_target": best_fine["coverage"] > 0,
            "peak_center_distance_normalized": float(
                np.hypot(peak_cx - target_cx, peak_cy - target_cy) / np.hypot(1024, 1024)
            ),
            "median_nonoverlap_drop": float(np.median(nonoverlap_drops)),
            "max_nonoverlap_drop": float(np.max(nonoverlap_drops)),
            "peak_x1": int(peak[0]),
            "peak_y1": int(peak[1]),
            "peak_x2": int(peak[2]),
            "peak_y2": int(peak[3]),
        }
    )

    heat = np.maximum(coarse, 0)
    heat = heat / max(float(heat.max()), 1e-9)
    heat = cv2.resize((heat * 255).astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    base = cv2.normalize(image[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), 0.62, heat, 0.38, 0)
    tx1, ty1, tx2, ty2 = [int(round(value)) for value in target]
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 255, 0), 3)
    cv2.rectangle(
        overlay,
        (int(peak[0]), int(peak[1])),
        (int(peak[2]), int(peak[3])),
        (255, 255, 255),
        3,
    )
    cv2.putText(
        overlay,
        f"id {image_id}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    occlusion_tiles.append(cv2.resize(overlay, (256, 256), interpolation=cv2.INTER_AREA))

occlusion_df = pd.DataFrame(occlusion_rows)
occlusion_df.to_csv(OUT / "occlusion_results.csv", index=False)
montage_rows = []
for start in range(0, len(occlusion_tiles), 5):
    row_tiles = occlusion_tiles[start : start + 5]
    while len(row_tiles) < 5:
        row_tiles.append(np.zeros_like(occlusion_tiles[0]))
    montage_rows.append(np.hstack(row_tiles))
cv2.imwrite(str(OUT / "occlusion_montage.png"), np.vstack(montage_rows))

report["occlusion"] = {
    "n": int(len(occlusion_df)),
    "peak_overlaps_target_rate": float(occlusion_df.peak_overlaps_target.mean()),
    "peak_target_coverage_median": float(occlusion_df.peak_target_coverage.median()),
    "max_drop_median": float(occlusion_df.max_drop.median()),
    "median_nonoverlap_drop_median": float(occlusion_df.median_nonoverlap_drop.median()),
    "peak_center_distance_normalized_median": float(
        occlusion_df.peak_center_distance_normalized.median()
    ),
    "guard": (
        "Occlusion is causal for the modified region but a replacement can introduce its own "
        "distribution shift. Agreement with saliency, removal, and transplant tests is required."
    ),
}
(OUT / "model_forensic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log(json.dumps(report, indent=2))
log("Research run complete")
