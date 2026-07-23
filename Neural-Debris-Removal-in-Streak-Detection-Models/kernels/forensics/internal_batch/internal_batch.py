# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RetinaNet internal forensic batch - P3.01 to P3.08
#
# Eight rule-safe checks run from one model load:
#
# 1. classification head
# 2. regression head
# 3. FPN P3-P7
# 4. early backbone
# 5. late backbone
# 6. normalization state
# 7. individual channels
# 8. paired poison/control feature embeddings
#
# Only the 20 public unlearn images and the supplied poisoned model are used.

# %%
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
UNLEARN = ROOT / "unlearn_set"
WEIGHTS = ROOT / "poisoned_model/poisoned_model.pth"
OUT = Path("/kaggle/working/internal_forensics")
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
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
predictor = DefaultPredictor(cfg)
model = predictor.model
model.eval()
log(f"Model ready on {cfg.MODEL.DEVICE}")


def control_box(target, height=1024, width=1024):
    box_width, box_height = target[2] - target[0], target[3] - target[1]
    candidates = [
        np.asarray([32, 32, 32 + box_width, 32 + box_height], np.float32),
        np.asarray([width - 32 - box_width, 32, width - 32, 32 + box_height], np.float32),
        np.asarray([32, height - 32 - box_height, 32 + box_width, height - 32], np.float32),
        np.asarray(
            [width - 32 - box_width, height - 32 - box_height, width - 32, height - 32],
            np.float32,
        ),
    ]
    target_center = (target[:2] + target[2:]) / 2
    return max(
        candidates,
        key=lambda candidate: np.linalg.norm((candidate[:2] + candidate[2:]) / 2 - target_center),
    )


def roi_vector(feature, box, input_height, input_width):
    _, channels, height, width = feature.shape
    x1 = int(np.floor(box[0] / input_width * width))
    y1 = int(np.floor(box[1] / input_height * height))
    x2 = int(np.ceil(box[2] / input_width * width))
    y2 = int(np.ceil(box[3] / input_height * height))
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(x1 + 1, min(x2, width)), max(y1 + 1, min(y2, height))
    region = feature[0, :, y1:y2, x1:x2]
    return region.mean(dim=(1, 2)).detach().cpu().numpy()


def roi_scalar_map(feature, box, input_height, input_width):
    _, channels, height, width = feature.shape
    x1 = int(np.floor(box[0] / input_width * width))
    y1 = int(np.floor(box[1] / input_height * height))
    x2 = int(np.ceil(box[2] / input_width * width))
    y2 = int(np.ceil(box[3] / input_height * height))
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(x1 + 1, min(x2, width)), max(y1 + 1, min(y2, height))
    return feature[0, :, y1:y2, x1:x2]


captured = {}
hooks = []
bottom_up = model.backbone.bottom_up
for layer_name in ["stem", "res2", "res3", "res4", "res5"]:
    module = getattr(bottom_up, layer_name)

    def save_output(_module, _inputs, output, name=layer_name):
        captured[name] = output

    hooks.append(module.register_forward_hook(save_output))

layer_poison = defaultdict(list)
layer_control = defaultdict(list)
head_rows = []

for image_id in tqdm(sorted(images), desc="P3.01-P3.08"):
    image = images[image_id]
    transform = predictor.aug.get_transform(image)
    resized = transform.apply_image(image)
    poison_box = transform.apply_box(np.asarray([boxes[image_id]], np.float32))[0]
    background_box = transform.apply_box(
        np.asarray([control_box(boxes[image_id])], np.float32)
    )[0]
    tensor = torch.as_tensor(
        np.ascontiguousarray(resized.transpose(2, 0, 1)),
        dtype=torch.float32,
        device=model.device,
    )

    captured.clear()
    with torch.no_grad():
        image_list = model.preprocess_image([{"image": tensor}])
        feature_dict = model.backbone(image_list.tensor)
        input_height, input_width = image_list.tensor.shape[-2:]

        for layer_name, activation in captured.items():
            layer_poison[f"backbone_{layer_name}"].append(
                roi_vector(activation, poison_box, input_height, input_width)
            )
            layer_control[f"backbone_{layer_name}"].append(
                roi_vector(activation, background_box, input_height, input_width)
            )

        for level_name in model.head_in_features:
            feature = feature_dict[level_name]
            layer_poison[f"fpn_{level_name}"].append(
                roi_vector(feature, poison_box, input_height, input_width)
            )
            layer_control[f"fpn_{level_name}"].append(
                roi_vector(feature, background_box, input_height, input_width)
            )

            cls_feature = model.head.cls_subnet(feature)
            reg_feature = model.head.bbox_subnet(feature)
            cls_logits = model.head.cls_score(cls_feature)
            bbox_offsets = model.head.bbox_pred(reg_feature)
            layer_poison[f"cls_{level_name}"].append(
                roi_vector(cls_feature, poison_box, input_height, input_width)
            )
            layer_control[f"cls_{level_name}"].append(
                roi_vector(cls_feature, background_box, input_height, input_width)
            )
            layer_poison[f"reg_{level_name}"].append(
                roi_vector(reg_feature, poison_box, input_height, input_width)
            )
            layer_control[f"reg_{level_name}"].append(
                roi_vector(reg_feature, background_box, input_height, input_width)
            )

            poison_logits = roi_scalar_map(
                cls_logits, poison_box, input_height, input_width
            ).sigmoid()
            control_logits = roi_scalar_map(
                cls_logits, background_box, input_height, input_width
            ).sigmoid()
            poison_offsets = roi_scalar_map(
                bbox_offsets, poison_box, input_height, input_width
            )
            control_offsets = roi_scalar_map(
                bbox_offsets, background_box, input_height, input_width
            )
            head_rows.append(
                {
                    "image_id": image_id,
                    "level": level_name,
                    "poison_cls_max": float(poison_logits.max().cpu()),
                    "control_cls_max": float(control_logits.max().cpu()),
                    "poison_cls_mean": float(poison_logits.mean().cpu()),
                    "control_cls_mean": float(control_logits.mean().cpu()),
                    "poison_bbox_abs_mean": float(poison_offsets.abs().mean().cpu()),
                    "control_bbox_abs_mean": float(control_offsets.abs().mean().cpu()),
                }
            )

for hook in hooks:
    hook.remove()

# %%
def grouped_auc(poison, control):
    poison = np.asarray(poison, np.float32)
    control = np.asarray(control, np.float32)
    features = np.concatenate([poison, control])
    labels = np.r_[np.ones(len(poison)), np.zeros(len(control))]
    groups = np.r_[np.arange(len(poison)), np.arange(len(control))]
    components = max(1, min(10, features.shape[1], len(features) - 2))
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=components, random_state=SEED),
        LogisticRegression(max_iter=4000, C=0.2),
    )
    prediction = cross_val_predict(
        pipeline,
        features,
        labels,
        groups=groups,
        cv=GroupKFold(5),
        method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(labels, prediction))


layer_rows = []
channel_rows = []
for layer_name in sorted(layer_poison):
    poison = np.asarray(layer_poison[layer_name], np.float32)
    control = np.asarray(layer_control[layer_name], np.float32)
    difference = poison - control
    effect = difference.mean(axis=0) / np.clip(difference.std(axis=0, ddof=1), 1e-6, None)
    order = np.argsort(np.abs(effect))[::-1]
    layer_rows.append(
        {
            "layer": layer_name,
            "channels": poison.shape[1],
            "poison_mean": float(poison.mean()),
            "control_mean": float(control.mean()),
            "mean_ratio": float(poison.mean() / max(abs(control.mean()), 1e-9)),
            "max_abs_channel_effect": float(np.max(np.abs(effect))),
            "top10_abs_effect_mean": float(np.mean(np.abs(effect[order[:10]]))),
            "grouped_cv_auc": grouped_auc(poison, control),
        }
    )
    for rank, channel in enumerate(order[:20], 1):
        channel_rows.append(
            {
                "layer": layer_name,
                "rank": rank,
                "channel": int(channel),
                "effect": float(effect[channel]),
                "poison_mean": float(poison[:, channel].mean()),
                "control_mean": float(control[:, channel].mean()),
            }
        )

layer_df = pd.DataFrame(layer_rows)
channel_df = pd.DataFrame(channel_rows)
head_df = pd.DataFrame(head_rows)
layer_df.to_csv(OUT / "layer_summary.csv", index=False)
channel_df.to_csv(OUT / "top_channels.csv", index=False)
head_df.to_csv(OUT / "head_summary.csv", index=False)

head_summary = (
    head_df.groupby("level")
    .agg(
        poison_cls_max=("poison_cls_max", "median"),
        control_cls_max=("control_cls_max", "median"),
        poison_bbox_abs_mean=("poison_bbox_abs_mean", "median"),
        control_bbox_abs_mean=("control_bbox_abs_mean", "median"),
    )
    .reset_index()
)
head_summary.to_csv(OUT / "head_level_summary.csv", index=False)

# P3.06 normalization inventory
normalization_counts = Counter()
normalization_state = []
for name, module in model.named_modules():
    class_name = module.__class__.__name__
    if any(token in class_name for token in ["BatchNorm", "FrozenBatchNorm", "GroupNorm"]):
        normalization_counts[class_name] += 1
        item = {"name": name, "type": class_name}
        if hasattr(module, "running_mean"):
            item["running_mean_abs_max"] = float(module.running_mean.detach().abs().max().cpu())
            item["running_var_max"] = float(module.running_var.detach().max().cpu())
        normalization_state.append(item)
(OUT / "normalization_state.json").write_text(
    json.dumps(normalization_state, indent=2), encoding="utf-8"
)

# %%
selected_layers = ["backbone_res2", "backbone_res5", "fpn_p3", "fpn_p4", "cls_p3", "cls_p4"]
figure, axes = plt.subplots(2, 3, figsize=(15, 9))
for axis, layer_name in zip(axes.ravel(), selected_layers):
    poison = np.asarray(layer_poison[layer_name], np.float32)
    control = np.asarray(layer_control[layer_name], np.float32)
    features = np.concatenate([poison, control])
    embedding = PCA(n_components=2, random_state=SEED).fit_transform(StandardScaler().fit_transform(features))
    axis.scatter(embedding[:20, 0], embedding[:20, 1], c="#ff6b6b", label="poison ROI", s=28)
    axis.scatter(embedding[20:, 0], embedding[20:, 1], c="#62e7b4", label="control ROI", s=28)
    axis.set_title(layer_name)
    axis.grid(alpha=0.2)
axes[0, 0].legend(fontsize=8)
figure.suptitle("P3.08 paired poison/control ROI embeddings")
figure.tight_layout()
figure.savefig(OUT / "embedding_panels.png", dpi=180)
plt.close(figure)

report = {
    "device": cfg.MODEL.DEVICE,
    "n_images": 20,
    "checks": {
        "P3.01_classification_head": "head_summary.csv and cls_* rows in layer_summary.csv",
        "P3.02_regression_head": "head_summary.csv and reg_* rows in layer_summary.csv",
        "P3.03_fpn_levels": "fpn_* rows in layer_summary.csv",
        "P3.04_early_backbone": "backbone_stem/res2/res3 rows",
        "P3.05_late_backbone": "backbone_res4/res5 rows",
        "P3.06_normalization": dict(normalization_counts),
        "P3.07_channels": "top_channels.csv",
        "P3.08_embeddings": "grouped_cv_auc plus embedding_panels.png",
    },
    "highest_grouped_auc_layers": (
        layer_df.sort_values("grouped_cv_auc", ascending=False)
        .head(10)
        .to_dict(orient="records")
    ),
    "head_level_medians": head_summary.set_index("level").to_dict(orient="index"),
    "guard": (
        "Poison/control ROI separability identifies suspicious representations, not uniquely "
        "poisoned parameters. Channel pruning requires causal held-out ablation before repair."
    ),
}
(OUT / "internal_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log(layer_df.sort_values("grouped_cv_auc", ascending=False).to_string(index=False))
log(json.dumps(report, indent=2))
log("P3.01-P3.08 batch complete")
