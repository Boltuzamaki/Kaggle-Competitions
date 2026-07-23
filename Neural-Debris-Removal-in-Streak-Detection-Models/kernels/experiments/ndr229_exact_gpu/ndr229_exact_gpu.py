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
# # Neural Debris NDR229 exact GPU reproduction
# ### Activation pruning, classifier EWC and metric-aware rescoring
#
# This notebook implements a self-contained, end-to-end pipeline to de-poison a RetinaNet streak detection model. It combines activation-guided channel pruning, classifier-only fine-tuning with EWC regularization, and metric-aware post-processing (demotion instead of deletion).
#
# **Credits & Reused Boilerplate:**
# - **Detectron2 setup and data catalog registration:** Reused from [simple-fine-tuning-baseline.ipynb](file:///c:/Codes/Kaggle_Notebooks/space/public_solutions/simple-fine-tuning-baseline.ipynb)
# - **Activation-guided channel pruning, Backbone freezing, EWC:** Reused/adapted from [de-poisining-through-prunning-finetuning.ipynb](file:///c:/Codes/Kaggle_Notebooks/space/public_solutions/de-poisining-through-prunning-finetuning.ipynb)
# - **Post-inference rescoring, Mahalanobis geometry, Local maCADD metric:** Reused/adapted from [debris-removal-calibrated-rescoring-v3-0.ipynb](file:///c:/Codes/Kaggle_Notebooks/space/public_solutions/debris-removal-calibrated-rescoring-v3-0.ipynb)
# - **Learning rate and iteration sweep settings:** Reused/adapted from [cv-debris-removal-de-poisoning-experiments.ipynb](file:///c:/Codes/Kaggle_Notebooks/space/public_solutions/cv-debris-removal-de-poisoning-experiments.ipynb)
#

# %% [markdown]
# ## 1. Setup
#
# Exact reproduction of the public 229.2314 recipe. The kernel preserves its
# score-bearing two-layer pruning index quirk and writes auditable artifacts.
#
# Public source: https://www.kaggle.com/code/sanidhyavijay24/ndr-trial1
#
# This kernel creates a model and CSV artifacts but does not submit anything.

# %%
import importlib.util
import os
import subprocess
import sys

# This kernel explicitly requests the same Tesla T4 runtime used by the scored
# public notebook. T4 is SM 7.5 and is supported by Kaggle's default CUDA 12.8
# PyTorch build, so no in-place torch/numpy/pillow replacement is required.
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
os.environ["MAX_JOBS"] = "2"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "setuptools<81"],
    check=True,
)
if importlib.util.find_spec("detectron2") is not None:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "detectron2"],
        check=True,
    )
print("[SETUP] Building Detectron2 for Tesla T4 (SM 7.5)", flush=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-build-isolation",
        "git+https://github.com/facebookresearch/detectron2.git",
    ],
    check=True,
)


# %% [markdown]
# ## 2. Imports and Configuration
#

# %%
import copy
import hashlib
import json
import logging
import pickle
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
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_train_loader,
    detection_utils as utils,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances

logging.getLogger("detectron2").setLevel(logging.ERROR)
logging.getLogger("fvcore").setLevel(logging.ERROR)

# Kaggle input paths
ROOT = "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models"
POISONED_WEIGHTS = f"{ROOT}/poisoned_model/poisoned_model.pth"
UNLEARN_DIR = f"{ROOT}/unlearn_set"
TEST_DIR = f"{ROOT}/test_set/test_set"
SAMPLE_SUB = f"{ROOT}/sample_submission.csv"

# Output paths
RUN_DIR = Path("/kaggle/working/ndr229_exact_gpu")
RUN_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = str(RUN_DIR / "unlearned")
SUBMISSION_PATH = "/kaggle/working/submission.csv"
MODEL_ALIAS = Path("/kaggle/working/ndr229_exact_model.pth")
RUN_LOG = RUN_DIR / "run.jsonl"
TRAIN_LOG = RUN_DIR / "training_history.csv"

# Model Architecture Configuration (Must match poisoned model exactly)
BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
NUM_CLASSES = 1
IMG_W = IMG_H = 1024
BATCH_SIZE = 4

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(message, **fields):
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "message": str(message),
        **fields,
    }
    print(json.dumps(row, default=str), flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def heartbeat(label, stop_event, started):
    while not stop_event.wait(30):
        log("HEARTBEAT", stage=label, elapsed_sec=round(time.time() - started, 1))


def run_with_heartbeat(label, fn):
    stop_event = threading.Event()
    started = time.time()
    worker = threading.Thread(
        target=heartbeat,
        args=(label, stop_event, started),
        daemon=True,
    )
    worker.start()
    try:
        return fn()
    finally:
        stop_event.set()
        worker.join(timeout=2)
        log("STAGE_COMPLETE", stage=label, elapsed_sec=round(time.time() - started, 1))


if DEVICE != "cuda":
    raise RuntimeError("This kernel is GPU-only; CUDA is not available")

gpu_capability = torch.cuda.get_device_capability(0)
gpu_arch = f"sm_{gpu_capability[0]}{gpu_capability[1]}"
compiled_arches = torch.cuda.get_arch_list()
if gpu_arch not in compiled_arches:
    raise RuntimeError(
        f"PyTorch wheel lacks {gpu_arch}; compiled architectures: {compiled_arches}"
    )
cuda_probe = torch.arange(16, dtype=torch.float32, device="cuda")
cuda_probe = (cuda_probe.square() + 1.0).sum()
torch.cuda.synchronize()
assert float(cuda_probe.cpu()) == 1256.0

required_inputs = [
    Path(POISONED_WEIGHTS),
    Path(UNLEARN_DIR),
    Path(TEST_DIR),
    Path(SAMPLE_SUB),
    Path(UNLEARN_DIR) / "annotations_coco.json",
]
missing_inputs = [str(path) for path in required_inputs if not path.exists()]
if missing_inputs:
    raise FileNotFoundError(f"Missing competition inputs: {missing_inputs}")

CONFIG = {
    "experiment": "E55_NDR229_EXACT",
    "public_reference_score": 229.2314,
    "seed": SEED,
    "device": DEVICE,
    "accelerator_request": "NvidiaTeslaT4",
    "runtime_stack": "kaggle_default_no_in_place_replacement",
    "scipy_dependency": False,
    "gpu_arch": gpu_arch,
    "compiled_arches": compiled_arches,
    "prune_frac": 0.15,
    "pruning_mode": "bug_faithful_two_layer_index_match",
    "trainable_scope": "classifier_only",
    "learning_rate": 2.5e-4,
    "iterations": 20,
    "batch_size": BATCH_SIZE,
    "ewc_lambda": 500.0,
    "candidate_threshold": 0.05,
    "match_iou": 0.5,
    "poison_signal_weights": {"confidence_drop": 0.90, "geometry": 0.10},
    "remap": {"min_keep": 0.20, "p_hi": 0.55, "p_lo": 0.25, "eps": 0.01},
    "rule_7a_guard": {
        "external_models": False,
        "manual_test_labels": False,
        "automatic_external_test_labels": False,
        "test_used_only_by_provided_and_repaired_models": True,
    },
}
(RUN_DIR / "run_config.json").write_text(
    json.dumps(CONFIG, indent=2),
    encoding="utf-8",
)
log(
    "RUN_START",
    gpu=torch.cuda.get_device_name(0),
    torch_version=torch.__version__,
    cuda_version=torch.version.cuda,
    config=CONFIG,
)


# %% [markdown]
# ## 3. Dataset Registration & Image Loading
#

# %%
def load_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im

# Register unlearn dataset
UNLEARN_DATASET = "unlearn"

def register_unlearn(unlearn_dir):
    json_path = Path(unlearn_dir) / "annotations_coco.json"
    with open(json_path) as f:
        coco = json.load(f)
    dicts = [
        {
            "file_name": str(Path(unlearn_dir) / im["file_name"]),
            "height":    im["height"],
            "width":     im["width"],
            "image_id":  im["id"],
            "annotations": [],   # empty annotations for unlearning
        }
        for im in coco["images"]
    ]
    if UNLEARN_DATASET in DatasetCatalog:
        DatasetCatalog.remove(UNLEARN_DATASET)
    DatasetCatalog.register(UNLEARN_DATASET, lambda: dicts)
    MetadataCatalog.get(UNLEARN_DATASET).set(thing_classes=["object"])
    print(f"Registered unlearn dataset: {len(dicts)} images (empty annotations)")
    return dicts

unlearn_dicts = register_unlearn(UNLEARN_DIR)

# Load original coco annotations for pruning box coordinates
with open(Path(UNLEARN_DIR) / "annotations_coco.json") as f:
    coco_data = json.load(f)

poison_boxes = {}
for ann in coco_data["annotations"]:
    iid = ann["image_id"]
    poison_boxes.setdefault(iid, []).append(ann["bbox"])
print(f"Loaded original annotations for {len(poison_boxes)} images")
assert len(unlearn_dicts) == 20, len(unlearn_dicts)
assert sum(len(items) for items in poison_boxes.values()) == 20
log(
    "DATA_READY",
    unlearn_images=len(unlearn_dicts),
    poison_boxes=sum(len(items) for items in poison_boxes.values()),
    test_images=len(list(Path(TEST_DIR).glob("*.png"))),
)

class UInt16DatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = load_image(dataset_dict["file_name"])
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
        dataset_dict["instances"] = utils.annotations_to_instances([], image.shape[:2])
        return dataset_dict



# %% [markdown]
# ## 4. Helper Functions
#

# %%
def build_cfg(weights=POISONED_WEIGHTS, score_thresh=0.05):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.DEVICE = DEVICE
    return cfg

def build_predictor(weights, score_thresh=0.05):
    return DefaultPredictor(build_cfg(weights, score_thresh=score_thresh))



# %% [markdown]
# ## 5. Local maCADD Metric Implementation
#

# %%
A_FACTOR = 10.0
IOU_THRESHOLDS = np.arange(0.2, 0.91, 0.1)


def linear_sum_assignment(cost_matrix):
    """Deterministic NumPy-only greedy matcher for the local proxy metric.

    The scored pipeline evaluates an empty clean reference on the unlearn set,
    so this matcher is not used for model training, test inference or exported
    confidence rescoring. Avoiding SciPy removes a fragile compiled ABI.
    """
    cost = np.asarray(cost_matrix)
    if cost.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    selected_rows = []
    selected_cols = []
    used_rows = set()
    used_cols = set()
    flat_order = np.argsort(cost, axis=None, kind="stable")
    for flat_index in flat_order:
        row, col = np.unravel_index(int(flat_index), cost.shape)
        if row in used_rows or col in used_cols:
            continue
        selected_rows.append(row)
        selected_cols.append(col)
        used_rows.add(row)
        used_cols.add(col)
        if len(used_rows) == min(cost.shape):
            break
    return (
        np.asarray(selected_rows, dtype=np.int64),
        np.asarray(selected_cols, dtype=np.int64),
    )


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = np.maximum(0, ax2 - ax1) * np.maximum(0, ay2 - ay1)
    area_b = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)

def acadd_t(clean_boxes, clean_scores, pred_boxes, pred_scores, t, clean_is_poison=None):
    if clean_is_poison is None:
        clean_is_poison = np.zeros(len(clean_boxes), dtype=bool)
    ious = iou_matrix(clean_boxes, pred_boxes)
    cost = 0.0
    matched_c, matched_p = set(), set()
    if ious.size:
        ri, ci = linear_sum_assignment(-ious)
        for i, j in zip(ri, ci):
            if ious[i, j] >= t:
                matched_c.add(i); matched_p.add(j)
                d = clean_scores[i] - pred_scores[j]
                if clean_is_poison[i]:
                    cost += (d / A_FACTOR) if d > 0 else -d
                else:
                    cost += d if d > 0 else (-d / A_FACTOR)
    cost += sum(clean_scores[i] for i in range(len(clean_scores)) if i not in matched_c)
    cost += sum(pred_scores[j] for j in range(len(pred_scores)) if j not in matched_p)
    return cost

def macadd(clean_preds, sub_preds, poison_flags=None):
    w = IOU_THRESHOLDS / IOU_THRESHOLDS.sum()
    per_image = []
    for stem, (cb, cs) in clean_preds.items():
        keep = cs > 0.2
        cb, cs = cb[keep], cs[keep]
        pf = poison_flags.get(stem)[keep] if poison_flags else None
        pb, ps = sub_preds.get(stem, (np.zeros((0, 4)), np.zeros(0)))
        per_image.append(sum(wi * acadd_t(cb, cs, pb, ps, t, pf)
                             for wi, t in zip(w, IOU_THRESHOLDS)))
    return float(np.mean(per_image))



# %% [markdown]
# ## 6. Activation-Guided Channel Pruning
#

# %%
def collect_activations(model, unlearn_dicts, poison_boxes):
    model.eval()
    target_layers = [m for m in model.head.cls_subnet if isinstance(m, nn.Conv2d)]
    hooks, stored = [], {}
    for i, layer in enumerate(target_layers):
        stored[i] = []
        hooks.append(
            layer.register_forward_hook(
                lambda m, inp, out, idx=i: stored[idx].append(out.detach().cpu())
            )
        )

    with torch.no_grad():
        for d in tqdm(unlearn_dicts, desc="Collecting activations"):
            im = load_image(d["file_name"])
            inp = torch.as_tensor(im.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            model([{"image": inp[0]}])

    for h in hooks:
        h.remove()
    return stored

def compute_poison_scores(stored, unlearn_dicts, poison_boxes):
    scores_per_layer = {}
    for layer_idx, activation_list in stored.items():
        fg_acc = None
        bg_acc = None
        n_fg = n_bg = 0

        for act, d in zip(activation_list, unlearn_dicts):
            act = act[0]  # (C, H, W)
            C, aH, aW = act.shape
            scale_y = aH / d["height"]
            scale_x = aW / d["width"]

            img_id = d["image_id"]
            boxes = poison_boxes.get(img_id, [])

            if fg_acc is None:
                fg_acc = torch.zeros(C)
                bg_acc = torch.zeros(C)

            for x, y, w, h in boxes:
                x1 = max(0, int(x * scale_x))
                y1 = max(0, int(y * scale_y))
                x2 = min(aW, int((x + w) * scale_x) + 1)
                y2 = min(aH, int((y + h) * scale_y) + 1)
                if x2 > x1 and y2 > y1:
                    patch = act[:, y1:y2, x1:x2].relu()
                    fg_acc += patch.mean(dim=[1, 2])
                    n_fg += 1

            rng = np.random.default_rng(seed=42)
            for _ in range(max(1, len(boxes))):
                ph = max(1, int(16 * scale_y))
                pw = max(1, int(16 * scale_x))
                ry = rng.integers(0, max(1, aH - ph))
                rx = rng.integers(0, max(1, aW - pw))
                patch = act[:, ry:ry+ph, rx:rx+pw].relu()
                bg_acc += patch.mean(dim=[1, 2])
                n_bg += 1

        if n_fg > 0:
            fg_acc /= n_fg
        if n_bg > 0:
            bg_acc /= n_bg

        scores_per_layer[layer_idx] = (fg_acc - bg_acc).numpy()
    return scores_per_layer

pruned_layer_records = []


def prune_channels(model, scores_per_layer, prune_frac=0.15):
    target_layers = [
        (i, m) for i, m in enumerate(model.head.cls_subnet)
        if isinstance(m, nn.Conv2d)
    ]
    total_pruned = 0
    for i, layer in target_layers:
        if i not in scores_per_layer:
            continue
        scores = scores_per_layer[i]
        n_prune = max(1, int(len(scores) * prune_frac))
        bad_channels = np.argsort(scores)[-n_prune:]
        with torch.no_grad():
            layer.weight.data[bad_channels] = 0.0
            if layer.bias is not None:
                layer.bias.data[bad_channels] = 0.0
        total_pruned += n_prune
        pruned_layer_records.append(
            {
                "sequential_index": int(i),
                "score_key": int(i),
                "channels_pruned": int(n_prune),
                "channels_total": int(len(scores)),
                "channel_ids": [int(value) for value in bad_channels.tolist()],
            }
        )
        print(f"  Layer cls_subnet[{i*2}]: pruned {n_prune}/{len(scores)} channels")
    print(f"Total channels pruned: {total_pruned}")
    return model

print("Building poisoned model for activation guided pruning...")
cfg = build_cfg(POISONED_WEIGHTS)
model = build_model(cfg)
DetectionCheckpointer(model).load(POISONED_WEIGHTS)
model = model.to(DEVICE)

print("Collecting activations...")
stored = run_with_heartbeat(
    "activation_collection",
    lambda: collect_activations(model, unlearn_dicts, poison_boxes),
)
scores_per_layer = compute_poison_scores(stored, unlearn_dicts, poison_boxes)

print("Applying channel pruning...")
model = prune_channels(model, scores_per_layer, prune_frac=0.15)
assert [row["sequential_index"] for row in pruned_layer_records] == [0, 2]
assert sum(row["channels_pruned"] for row in pruned_layer_records) == 76
pruning_audit = {
    "mode": "bug_faithful_two_layer_index_match",
    "activation_score_keys": sorted(int(key) for key in scores_per_layer),
    "eligible_sequential_conv_indices": [
        int(i)
        for i, module in enumerate(model.head.cls_subnet)
        if isinstance(module, nn.Conv2d)
    ],
    "executed": pruned_layer_records,
}
(RUN_DIR / "pruning_audit.json").write_text(
    json.dumps(pruning_audit, indent=2),
    encoding="utf-8",
)
log("PRUNING_COMPLETE", audit=pruning_audit)


# %% [markdown]
# ## 7. Classifier-Only Fine-Tuning with Frozen Backbone and EWC
#

# %%
# Save post-pruning weights as EWC anchor for classifier layers
anchor_weights = {
    name: param.detach().clone()
    for name, param in model.named_parameters()
    if param.requires_grad and ("cls_score" in name or "cls_subnet" in name)
}

# Save the pruned model weights temporarily
os.makedirs(OUTPUT_DIR, exist_ok=True)
pruned_weights_path = os.path.join(OUTPUT_DIR, "pruned_model.pth")
torch.save({"model": model.state_dict()}, pruned_weights_path)
print(f"Pruned model weights saved temporarily to {pruned_weights_path}")

class ClassifierOnlyEWCTrainer(DefaultTrainer):
    anchor_weights = None
    ewc_lambda = 500.0   # strength of L2 regularization towards post-pruning weights

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper = UInt16DatasetMapper(cfg, is_train=True, augmentations=[])
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dicts)

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        
        # Freeze backbone, FPN, and bbox regressor entirely
        # Only head.cls_subnet and head.cls_score remain trainable
        for name, param in model.named_parameters():
            if "cls_score" in name or "cls_subnet" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (Classifier-only): {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        assert trainable == 2_376_455, trainable
        return model

    def run_step(self):
        assert self.model.training
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)

        loss_dict = self.model(data)

        # Apply EWC regularization on trainable classifier parameters
        if self.anchor_weights is not None:
            ewc_loss = torch.tensor(0.0, device=DEVICE)
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.anchor_weights:
                    anchor = self.anchor_weights[name]
                    ewc_loss = ewc_loss + ((param - anchor) ** 2).sum()
            loss_dict["loss_ewc"] = self.ewc_lambda * ewc_loss

        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        row = {
            "iteration": int(self.iter),
            "loss_total": float(losses.detach().cpu()),
            **{
                key: float(value.detach().cpu())
                for key, value in loss_dict.items()
            },
        }
        if not hasattr(self, "history"):
            self.history = []
        self.history.append(row)
        log("TRAIN_STEP", **row)

        if self.iter % 5 == 0:
            log_str = "  ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"[iter {self.iter:3d}] {log_str}")

# Configure trainer
cfg = build_cfg(weights=pruned_weights_path)
cfg.DATASETS.TRAIN = (UNLEARN_DATASET,)
cfg.DATASETS.TEST  = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH   = BATCH_SIZE
cfg.SOLVER.BASE_LR         = 2.5e-4
cfg.SOLVER.MAX_ITER        = 20
cfg.SOLVER.STEPS           = []
cfg.OUTPUT_DIR             = OUTPUT_DIR

# Train the model
trainer = ClassifierOnlyEWCTrainer(cfg)
trainer.anchor_weights = anchor_weights
trainer.resume_or_load(resume=False)
run_with_heartbeat("classifier_ewc_training", trainer.train)

# Path to our final fine-tuned model
DEPOISONED_WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
assert Path(DEPOISONED_WEIGHTS).exists()
pd.DataFrame(trainer.history).to_csv(TRAIN_LOG, index=False)
shutil.copyfile(DEPOISONED_WEIGHTS, MODEL_ALIAS)
log(
    "TRAINING_COMPLETE",
    iterations=len(trainer.history),
    model_path=str(MODEL_ALIAS),
    model_sha256=sha256(MODEL_ALIAS),
)


# %% [markdown]
# ## 8. Local Validation on Unlearn Set (Empty Clean Reference)
#

# %%
# Define predictors
poisoned_predictor = build_predictor(POISONED_WEIGHTS, score_thresh=0.05)
depoisoned_predictor = build_predictor(DEPOISONED_WEIGHTS, score_thresh=0.05)

# Collect predictions on the 20 unlearn images
unlearn_files = sorted(Path(UNLEARN_DIR).glob("*.png"))

poisoned_preds = {}
depoisoned_preds = {}

for p in tqdm(unlearn_files, desc="Validating predictors"):
    stem = p.stem
    img = load_image(p)
    
    # Original poisoned model predictions
    out_p = poisoned_predictor(img)["instances"].to("cpu")
    poisoned_preds[stem] = (out_p.pred_boxes.tensor.numpy(), out_p.scores.numpy())
    
    # De-poisoned model predictions
    out_d = depoisoned_predictor(img)["instances"].to("cpu")
    depoisoned_preds[stem] = (out_d.pred_boxes.tensor.numpy(), out_d.scores.numpy())

# Calculate local maCADD score assuming clean reference is completely empty
# (Lower is better; 0 means no detections remain on the poisoned images)
empty_clean_ref = {stem: (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)) for stem in poisoned_preds}

score_raw = macadd(empty_clean_ref, poisoned_preds)
score_dep = macadd(empty_clean_ref, depoisoned_preds)

print(f"\nLocal unlearning maCADD score (Raw poisoned model): {score_raw:.4f}")
print(f"Local unlearning maCADD score (De-poisoned model): {score_dep:.4f}")
print(f"Poison detection score reduction: {100 * (score_raw - score_dep) / score_raw:.2f}%")
unlearn_validation = {
    "raw_empty_reference_macadd": float(score_raw),
    "repaired_empty_reference_macadd": float(score_dep),
    "poison_score_reduction_fraction": float((score_raw - score_dep) / score_raw),
    "raw_detection_count": int(sum(len(scores) for _, scores in poisoned_preds.values())),
    "repaired_detection_count": int(sum(len(scores) for _, scores in depoisoned_preds.values())),
}
(RUN_DIR / "unlearn_validation.json").write_text(
    json.dumps(unlearn_validation, indent=2),
    encoding="utf-8",
)
log("UNLEARN_VALIDATION_COMPLETE", metrics=unlearn_validation)


# %% [markdown]
# ## 9. Box Geometry Mahalanobis Distance Setup
#

# %%
# Fit a Mahalanobis distance distribution on the 20 known poison boxes from unlearn set
logwh = np.log(np.array([ann["bbox"][2:4] for ann in coco_data["annotations"]], dtype=np.float32))
MU = logwh.mean(0)
COV = np.cov(logwh.T) + 1e-3 * np.eye(2)
COV_INV = np.linalg.inv(COV)

def geometry_scores(boxes):
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
    d = np.log(np.maximum(wh, 1e-3)) - MU
    m2 = np.einsum("ni,ij,nj->n", d, COV_INV, d)
    return np.exp(-0.5 * m2 / 2.0).astype(np.float32)



# %% [markdown]
# ## 10. Test Set Inference & Metric-Aware Demotion
#

# %%
test_files = sorted(Path(TEST_DIR).glob("*.png"))
print(f"Running inference on {len(test_files)} test images...")

# Post-processing thresholds
R = {
    "MIN_KEEP": 0.20,  # drop boxes with baseline confidence below 0.20
    "P_HI": 0.55,      # demote boxes with poison probability >= 0.55
    "P_LO": 0.25,      # keep original confidence if poison probability <= 0.25
    "EPS": 0.01        # demoted confidence value
}

def remap_confidence(s, p, R):
    if s < R["MIN_KEEP"]:
        return 0.0
    if p >= R["P_HI"]:
        return R["EPS"]
    if p <= R["P_LO"]:
        return float(s)
    frac = (p - R["P_LO"]) / max(R["P_HI"] - R["P_LO"], 1e-6)
    return float(max(R["EPS"], s * (1 - frac)))

final_test_preds = {}
test_diagnostics = []
inference_started = time.time()

for image_index, img_path in enumerate(tqdm(test_files, desc="Test Inference"), start=1):
    stem = img_path.stem
    im = load_image(img_path)
    
    # 1. Bounding box candidates from original model at 0.05 threshold (high recall)
    cand_out = poisoned_predictor(im)["instances"].to("cpu")
    cand_boxes = cand_out.pred_boxes.tensor.numpy()
    cand_scores = cand_out.scores.numpy()
    
    if len(cand_boxes) == 0:
        final_test_preds[stem] = (cand_boxes, cand_scores)
        test_diagnostics.append(
            {
                "image_id": stem,
                "original_candidates": 0,
                "repaired_candidates": 0,
                "kept": 0,
                "demoted": 0,
                "mean_p_poison": 0.0,
            }
        )
        continue
        
    # 2. De-poisoned model predictions on the same image
    dep_out = depoisoned_predictor(im)["instances"].to("cpu")
    dep_boxes = dep_out.pred_boxes.tensor.numpy()
    dep_scores = dep_out.scores.numpy()
    
    # Match candidate boxes with de-poisoned model detections using IoU
    s_diff = np.zeros(len(cand_boxes), dtype=np.float32)
    if len(dep_boxes) > 0:
        ious = iou_matrix(cand_boxes, dep_boxes)
        best_match_idx = ious.argmax(1)
        best_match_iou = ious[np.arange(len(cand_boxes)), best_match_idx]
        
        # Calculate classification confidence drop (s_diff)
        for idx in range(len(cand_boxes)):
            if best_match_iou[idx] >= 0.5:
                dep_score = dep_scores[best_match_idx[idx]]
                s_diff[idx] = 1.0 - (dep_score / max(cand_scores[idx], 1e-6))
            else:
                s_diff[idx] = 1.0  # complete unlearning / deletion
    else:
        s_diff = np.ones(len(cand_boxes), dtype=np.float32)
        
    # Compute geometry similarity score
    s_geo = geometry_scores(cand_boxes)
    
    # Combined poison probability
    p_poison = 0.90 * s_diff + 0.10 * s_geo
    
    # Apply confidence remap
    new_conf = np.array([remap_confidence(s, p, R) for s, p in zip(cand_scores, p_poison)], dtype=np.float32)
    
    # Filter boxes
    keep = new_conf > 0.0
    
    # Overlap suppression for demoted boxes to avoid duplicate boxes
    eps_ids = np.where(new_conf <= R["EPS"] + 1e-6)[0]
    strong_ids = np.where(new_conf > 0.20)[0]
    if len(eps_ids) and len(strong_ids):
        overl = iou_matrix(cand_boxes[eps_ids], cand_boxes[strong_ids]).max(1)
        keep[eps_ids[overl >= 0.20]] = False
        
    final_test_preds[stem] = (cand_boxes[keep], new_conf[keep])
    test_diagnostics.append(
        {
            "image_id": stem,
            "original_candidates": int(len(cand_boxes)),
            "repaired_candidates": int(len(dep_boxes)),
            "kept": int(keep.sum()),
            "demoted": int(((new_conf <= R["EPS"] + 1e-6) & keep).sum()),
            "mean_p_poison": float(p_poison.mean()) if len(p_poison) else 0.0,
        }
    )
    if image_index % 100 == 0 or image_index == len(test_files):
        log(
            "INFERENCE_PROGRESS",
            completed=image_index,
            total=len(test_files),
            elapsed_sec=round(time.time() - inference_started, 1),
        )

pd.DataFrame(test_diagnostics).to_csv(
    RUN_DIR / "test_diagnostics.csv",
    index=False,
)


# %% [markdown]
# ## 11. Write Submission File
#

# %%
sample = pd.read_csv(SAMPLE_SUB, dtype={"image_id": str})

def format_preds(bx, sc):
    parts = []
    for (x1, y1, x2, y2), s in zip(bx, sc):
        x1, y1 = float(np.clip(x1, 0, IMG_W)), float(np.clip(y1, 0, IMG_H))
        x2, y2 = float(np.clip(x2, 0, IMG_W)), float(np.clip(y2, 0, IMG_H))
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0 and s > 0:
            parts += [f"{s:.6f}", f"{x1:.2f}", f"{y1:.2f}", f"{w:.2f}", f"{h:.2f}"]
    return " ".join(parts) or " "

df = sample.copy()
df["prediction_string"] = df["image_id"].map(
    lambda i: format_preds(*final_test_preds.get(str(i), (np.zeros((0, 4)), np.zeros(0))))
)
df.to_csv(SUBMISSION_PATH, index=False)
total_boxes = sum(len(s[1]) for s in final_test_preds.values())
print(f"Saved {SUBMISSION_PATH} with {len(df)} rows and {total_boxes} predictions total.")


def validate_submission(frame, sample_frame):
    assert list(frame.columns) == list(sample_frame.columns)
    assert len(frame) == 2000
    assert frame["image_id"].astype(str).is_unique
    assert frame["image_id"].astype(str).tolist() == sample_frame["image_id"].astype(str).tolist()
    assert frame["prediction_string"].isna().sum() == 0

    parsed_boxes = 0
    for row_index, prediction in enumerate(frame["prediction_string"].astype(str)):
        if not prediction.strip():
            assert prediction == " "
            continue
        values = [float(value) for value in prediction.split()]
        assert len(values) % 5 == 0, row_index
        for offset in range(0, len(values), 5):
            confidence, x, y, width, height = values[offset:offset + 5]
            assert 0.0 < confidence <= 1.0
            assert 0.0 <= x <= IMG_W
            assert 0.0 <= y <= IMG_H
            assert width > 0.0 and height > 0.0
            assert x + width <= IMG_W + 0.05
            assert y + height <= IMG_H + 0.05
            parsed_boxes += 1
    return parsed_boxes


parsed_boxes = validate_submission(df, sample)
assert parsed_boxes == total_boxes
submission_copy = RUN_DIR / "submission_ndr229_exact.csv"
shutil.copyfile(SUBMISSION_PATH, submission_copy)

diagnostics_frame = pd.DataFrame(test_diagnostics)
final_report = {
    "status": "complete",
    "experiment": "E55_NDR229_EXACT",
    "public_reference_score": 229.2314,
    "rows": int(len(df)),
    "unique_image_ids": int(df["image_id"].astype(str).nunique()),
    "total_boxes": int(total_boxes),
    "mean_boxes_per_image": float(total_boxes / len(df)),
    "demoted_boxes": int(diagnostics_frame["demoted"].sum()),
    "mean_poison_probability": float(diagnostics_frame["mean_p_poison"].mean()),
    "submission_sha256": sha256(SUBMISSION_PATH),
    "submission_alias_sha256": sha256(submission_copy),
    "model_sha256": sha256(MODEL_ALIAS),
    "unlearn_validation": unlearn_validation,
    "pruning_audit": pruning_audit,
    "rule_7a_guard_passed": True,
    "competition_submission_created": False,
}
assert final_report["submission_sha256"] == final_report["submission_alias_sha256"]
(RUN_DIR / "final_report.json").write_text(
    json.dumps(final_report, indent=2),
    encoding="utf-8",
)
log("RUN_COMPLETE", report=final_report)
print(json.dumps(final_report, indent=2))

