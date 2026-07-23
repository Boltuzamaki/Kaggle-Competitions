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
# # NDR Contrastive V9: unlearner × amplifier
# **Measured public control:** M1 = **216.54**. The new N1 rule combines
# unlearner survival, amplifier contrast, prototype score and confidence.
#
# New signal: a **poison amplifier** - a second head-only fine-tune with pasted poison boxes as *positives*
# and synthetic streaks as background. It moves opposite to the unlearner: poison detections get amp-ratio
# a>1, real streaks a<1. The keep rule becomes 2D - `r` high AND `a` low - with proto and conf as weak priors.
# `submission.csv = N1_center` (fused rule, K≈450). The N-ladder ablates each component so the last two days
# converge on the precision-optimal rule.
#

# %%
# =====================  CONFIG  =====================
CFG = dict(
    DO_UNLEARN = True,       # unlearner student (ratio r)
    DO_AMP = True,           # poison amplifier (ratio a) - the contrastive probe
    HEAD_ONLY = True, ITERS = 300, LR = 4e-5, BS_FORGET = 2, BS_RETAIN = 2,
    W_FORGET = 1.0, W_KD = 6.0, W_KD_BOX = 0.5, W_SP = 1e-4,
    FG_KD_THRESH = 0.05, FG_KD_W = 2.0, BG_KD_W = 0.02,
    AMP_ITERS = 180, AMP_LR = 6e-5,
    EVAL_EVERY = 50, GRAD_CLIP = 1.0, RET_GATE = 0.80,
    N_HELDOUT = 24, N_AUDIT_SYN = 32, POISON_PER_SCENE = 3,
    SYN_SNR = (4.0, 30.0), SYN_LEN = (24, 90), SYN_SIGMA = (1.2, 3.2),
    CAND_THRESH = 0.02, AMP_INFER = True, USE_PROTO = True,
    TAU_P = 0.85, FINAL_NMS_IOU = 0.35, TOPK_PER_IMAGE = 25,
    SEED = 42,
)
for k, v in CFG.items():
    print(f"  {k:>16} = {v}")

# %%
# The kernel requests the same T4/container combination that completed the
# NDR229 reproduction. Core scientific packages are not replaced in-process.
import importlib.util
import os
import subprocess
import sys

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

import torch
import torchvision

if not torch.cuda.is_available():
    raise RuntimeError("This notebook requires a Kaggle Tesla T4 GPU")
capability = torch.cuda.get_device_capability(0)
gpu_arch = f"sm_{capability[0]}{capability[1]}"
if gpu_arch != "sm_75" or gpu_arch not in torch.cuda.get_arch_list():
    raise RuntimeError(
        f"Expected Tesla T4 sm_75, received {torch.cuda.get_device_name(0)} "
        f"{gpu_arch}; compiled arches={torch.cuda.get_arch_list()}"
    )
val = float((torch.ones(8, device="cuda") * 2).sum().item())
assert val == 16.0
print(
    "CUDA smoke test OK",
    torch.cuda.get_device_name(0),
    gpu_arch,
    torch.__version__,
    torchvision.__version__,
)

# %%
import copy, hashlib, json, math, random, shutil, threading, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage

import logging
for _n in ("detectron2", "fvcore", "d2"):
    logging.getLogger(_n).setLevel(logging.ERROR)

SEED = CFG["SEED"]
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_W = IMG_H = 1024

# ---- explicit competition paths; test pixels are not read during selection ----
CAND_ROOTS = [
    "/kaggle/input/neural-debris-removal-in-streak-detection-models",
    "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models",
]
ROOT = next((r for r in CAND_ROOTS if os.path.isdir(r)), None)
assert ROOT is not None, "competition data not found under /kaggle/input"
POISONED_WEIGHTS = str(Path(ROOT) / "poisoned_model" / "poisoned_model.pth")
UNLEARN_DIR = str(Path(ROOT) / "unlearn_set")
UNLEARN_JSON = str(Path(UNLEARN_DIR) / "annotations_coco.json")
TEST_DIR = str(Path(ROOT) / "test_set" / "test_set")
SAMPLE_SUB = str(Path(ROOT) / "sample_submission.csv")
for required in (POISONED_WEIGHTS, UNLEARN_JSON, TEST_DIR, SAMPLE_SUB):
    assert Path(required).exists(), required

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
RUN_DIR = WORK / "ndr_contrastive_v9"
RUN_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = RUN_DIR / "run.jsonl"
TRAIN_HISTORY = []
CURRENT_STAGE = {"name": "public_control_generation"}
STOP_HEARTBEAT = threading.Event()
RUN_STARTED = time.time()


def log_event(message, **fields):
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        "stage": CURRENT_STAGE["name"],
        **fields,
    }
    print(json.dumps(row, default=str), flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def set_stage(name):
    CURRENT_STAGE["name"] = name
    log_event("STAGE_START")


def heartbeat():
    while not STOP_HEARTBEAT.wait(30):
        log_event("HEARTBEAT", elapsed_sec=round(time.time() - RUN_STARTED, 1))


threading.Thread(target=heartbeat, daemon=True).start()

print("ROOT   :", ROOT)
print("weights:", POISONED_WEIGHTS)
print("unlearn:", UNLEARN_DIR, "->", len(list(Path(UNLEARN_DIR).glob("*.png"))), "png")
print("test   :", TEST_DIR, "(not enumerated until the frozen inference stage)")
print("sample :", SAMPLE_SUB)
log_event(
    "RUN_START",
    config=CFG,
    gpu=torch.cuda.get_device_name(0),
    gpu_arch=gpu_arch,
    source_notebook="biohack44/ndr-trial-v2",
    measured_public_control=216.54,
)

def iou_matrix(a, b):
    # a: (N,4) XYXY, b: (M,4) XYXY -> (N,M)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    ua = np.maximum(0, ax2 - ax1) * np.maximum(0, ay2 - ay1)
    ub = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    union = ua + ub - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)

def to3(g):
    return np.repeat(g[:, :, None], 3, axis=2)


# %%
# Architecture + preprocessing identical to the official baseline (do not change).
BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
NUM_CLASSES = 1

def load_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert im is not None, path
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    else:
        im = im.astype(np.float32) / 255.0
    im = np.clip(im * 255.0, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im  # HWC float32 in [0,255]

def load_image1(path):
    return load_image(path)[:, :, 0].copy()

def build_cfg(weights, score_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.DEVICE = DEVICE
    return cfg

class FeatPredictor:
    # DefaultPredictor + optional P3 FPN feature capture (for prototype scoring)
    def __init__(self, weights, score_thresh, capture=False):
        self.pred = DefaultPredictor(build_cfg(weights, score_thresh))
        self.capture = capture
        self._p3 = None
        if capture:
            self.hook = self.pred.model.backbone.register_forward_hook(self._grab)
    def _grab(self, module, inputs, output):
        self._p3 = output["p3"].detach()
    def __call__(self, img):
        with torch.autocast("cuda", enabled=(CFG["AMP_INFER"] and DEVICE == "cuda")):
            out = self.pred(img)
        inst = out["instances"].to("cpu")
        return (inst.pred_boxes.tensor.numpy().astype(np.float32),
                inst.scores.numpy().astype(np.float32))
    def pool_feats(self, boxes_xyxy):
        # mean-pool the captured P3 map (stride 8, resized coords) inside each original-coord box
        f = self._p3[0].float()
        C, Hf, Wf = f.shape
        s = (Hf * 8.0) / IMG_H  # resized/original scale
        out = np.zeros((len(boxes_xyxy), C), np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            fx1 = int(np.clip(np.floor(x1 * s / 8.0), 0, Wf - 1))
            fy1 = int(np.clip(np.floor(y1 * s / 8.0), 0, Hf - 1))
            fx2 = int(np.clip(np.ceil(x2 * s / 8.0), fx1 + 1, Wf))
            fy2 = int(np.clip(np.ceil(y2 * s / 8.0), fy1 + 1, Hf))
            v = f[:, fy1:fy2, fx1:fx2].mean(dim=(1, 2)).cpu().numpy()
            out[i] = v / (np.linalg.norm(v) + 1e-6)
        return out


# %%
with open(UNLEARN_JSON) as f:
    coco = json.load(f)
id2im = {im["id"]: im for im in coco["images"]}
poison_anns = {}
for ann in coco["annotations"]:
    fn = id2im[ann["image_id"]]["file_name"]
    poison_anns.setdefault(fn, []).append(list(ann["bbox"]))  # XYWH
unlearn_files = sorted(Path(UNLEARN_DIR).glob("*.png"))
n_boxes = sum(len(v) for v in poison_anns.values())
whs = np.array([b[2:4] for v in poison_anns.values() for b in v], np.float32)
print(f"{len(unlearn_files)} unlearn images | {n_boxes} poison boxes")
print("w percentiles :", np.percentile(whs[:, 0], [0, 25, 50, 75, 100]).round(1))
print("h percentiles :", np.percentile(whs[:, 1], [0, 25, 50, 75, 100]).round(1))
print("aspect w/h    :", np.percentile(whs[:, 0] / np.maximum(whs[:, 1], 1), [0, 50, 100]).round(2))

# Mahalanobis geometry over log(w,h) of poison boxes (weak prior, small blend weight)
logwh = np.log(np.maximum(whs, 1e-3))
GEO_MU = logwh.mean(0)
GEO_COV_INV = np.linalg.inv(np.cov(logwh.T) + 1e-3 * np.eye(2))
def geo_score(boxes_xyxy):
    if len(boxes_xyxy) == 0:
        return np.zeros(0, np.float32)
    wh = np.stack([boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1]], 1)
    d = np.log(np.maximum(wh, 1e-3)) - GEO_MU
    m2 = np.einsum("ni,ij,nj->n", d, GEO_COV_INV, d)
    return np.exp(-0.25 * m2).astype(np.float32)

# LOOK AT THIS GRID: this is the trigger being unlearned. If these crops are visually
# indistinguishable from normal streaks, expect weaker generalization from every component.
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
k = 0
for fp in unlearn_files:
    g = None
    for (x, y, w, h) in poison_anns.get(fp.name, []):
        if k >= 12:
            break
        if g is None:
            g = load_image1(fp)
        pad = 24
        x1, y1 = int(max(0, x - pad)), int(max(0, y - pad))
        x2, y2 = int(min(IMG_W, x + w + pad)), int(min(IMG_H, y + h + pad))
        ax = axes[k // 4][k % 4]
        ax.imshow(g[y1:y2, x1:x2], cmap="gray")
        ax.set_title(f"{fp.name[:16]}  {int(w)}x{int(h)}", fontsize=8)
        ax.axis("off")
        k += 1
    if k >= 12:
        break
for j in range(k, 12):
    axes[j // 4][j % 4].axis("off")
plt.suptitle("poison GT crops (the trigger)"); plt.tight_layout(); plt.show()

# %%
# The teacher (poisoned model) MUST detect the poison GT with high confidence.
# If recall here is low, the preprocessing does not match training: stop and fix first.
teacher = FeatPredictor(POISONED_WEIGHTS, CFG["CAND_THRESH"], capture=CFG["USE_PROTO"])

hits, confs = 0, []
for fp in tqdm(unlearn_files, desc="teacher @ unlearn"):
    boxes, scores = teacher(load_image(fp))
    gts = np.array([[x, y, x + w, y + h] for (x, y, w, h) in poison_anns.get(fp.name, [])], np.float32)
    if len(gts) == 0:
        continue
    m = iou_matrix(gts, boxes)
    for i in range(len(gts)):
        if m.shape[1] and m[i].max() >= 0.5:
            hits += 1
            confs.append(float(scores[m[i].argmax()]))
print(f"poisoned-model recall@0.5 on poison GT: {hits}/{n_boxes}"
      f" | mean conf {np.mean(confs) if confs else 0:.3f}")
assert hits >= 0.7 * n_boxes, "loader/config mismatch vs training - do not proceed"

# %%
# 1-channel float32 scenes generated on the fly (RAM-safe). to3() at model boundary.
rng = np.random.default_rng(SEED)

def inpaint1(g, boxes_xywh, pad=6):
    g = g.copy()
    for (x, y, w, h) in boxes_xywh:
        x1, y1 = int(max(0, x - pad)), int(max(0, y - pad))
        x2, y2 = int(min(IMG_W, x + w + pad)), int(min(IMG_H, y + h + pad))
        rx1, ry1 = max(0, x1 - 32), max(0, y1 - 32)
        rx2, ry2 = min(IMG_W, x2 + 32), min(IMG_H, y2 + 32)
        ring = g[ry1:ry2, rx1:rx2]
        med = float(np.median(ring))
        sig = 1.4826 * float(np.median(np.abs(ring - med))) + 1e-3
        g[y1:y2, x1:x2] = np.clip(
            np.random.normal(med, sig, (y2 - y1, x2 - x1)), 0, 255).astype(np.float32)
    return g

BG_BANK = [inpaint1(load_image1(fp), poison_anns.get(fp.name, [])) for fp in unlearn_files]
ORIGINALS = [
    (load_image1(fp),
     np.array([[x, y, x + w, y + h] for (x, y, w, h) in poison_anns.get(fp.name, [])], np.float32))
    for fp in unlearn_files
]

PATCHES = []
for fp in unlearn_files:
    g = load_image1(fp)
    for (x, y, w, h) in poison_anns.get(fp.name, []):
        pad = 10
        x1, y1 = int(max(0, x - pad)), int(max(0, y - pad))
        x2, y2 = int(min(IMG_W, x + w + pad)), int(min(IMG_H, y + h + pad))
        PATCHES.append(g[y1:y2, x1:x2].copy())
print("poison patch bank:", len(PATCHES), "| background bank:", len(BG_BANK))

def rand_bg1(rng):
    b = BG_BANK[int(rng.integers(0, len(BG_BANK)))]
    b = np.rot90(b, int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        b = b[:, ::-1]
    return np.ascontiguousarray(b)

def paste_poison1(bg, rng):
    # additive-light blend: signal = patch - its own background level (no rectangle seams).
    # rot90/flip + mild scale only: arbitrary-angle warps could smear a pixel-precise trigger.
    g = bg.copy()
    boxes = []
    for _ in range(int(rng.integers(1, CFG["POISON_PER_SCENE"] + 1))):
        p = PATCHES[int(rng.integers(0, len(PATCHES)))].astype(np.float32)
        p = np.rot90(p, int(rng.integers(0, 4)))
        if rng.random() < 0.5:
            p = p[:, ::-1]
        sc = float(rng.uniform(0.85, 1.2))
        p = cv2.resize(p, (max(8, int(p.shape[1] * sc)), max(8, int(p.shape[0] * sc))))
        p = p * float(rng.uniform(0.8, 1.25))
        ph, pw = p.shape
        if ph >= IMG_H - 2 or pw >= IMG_W - 2:
            continue
        y0 = int(rng.integers(1, IMG_H - ph - 1))
        x0 = int(rng.integers(1, IMG_W - pw - 1))
        sig = np.clip(p - float(np.median(p)), 0, None)
        g[y0:y0 + ph, x0:x0 + pw] = np.clip(g[y0:y0 + ph, x0:x0 + pw] + sig, 0, 255)
        boxes.append([x0, y0, x0 + pw, y0 + ph])
    return g, (np.array(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32))

def synth1(bg, rng):
    # synthetic clean streaks: AA line + gaussian PSF; SYN_LEN now spans SHORT (trigger-size) to long
    g = bg.copy()
    med = float(np.median(g))
    nsig = 1.4826 * float(np.median(np.abs(g - med))) + 1e-3
    boxes = []
    for _ in range(int(rng.integers(1, 3))):
        L = float(rng.uniform(*CFG["SYN_LEN"]))
        ang = float(rng.uniform(0, np.pi))
        cx, cy = float(rng.uniform(80, IMG_W - 80)), float(rng.uniform(80, IMG_H - 80))
        dx, dy = 0.5 * L * np.cos(ang), 0.5 * L * np.sin(ang)
        x1, y1, x2, y2 = cx - dx, cy - dy, cx + dx, cy + dy
        sgm = float(rng.uniform(*CFG["SYN_SIGMA"]))
        snr = float(rng.uniform(*CFG["SYN_SNR"]))
        canvas = np.zeros((IMG_H, IMG_W), np.float32)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, 1, cv2.LINE_AA)
        canvas = cv2.GaussianBlur(canvas, (0, 0), sgm)
        mx = canvas.max()
        if mx > 0:
            canvas /= mx
        g = np.clip(g + canvas * snr * nsig, 0, 255)
        bx1, by1 = max(0.0, min(x1, x2) - 3 * sgm), max(0.0, min(y1, y2) - 3 * sgm)
        bx2, by2 = min(float(IMG_W), max(x1, x2) + 3 * sgm), min(float(IMG_H), max(y1, y2) + 3 * sgm)
        boxes.append([bx1, by1, bx2, by2])
    return g, np.array(boxes, np.float32)

def sample_forget(rng):
    # V8 collapse fix: 60% MIXED discriminative scenes - pasted triggers stay UNLABELED (background)
    # while synthetic clean streaks are the GT positives. The model must separate, not silence.
    r = rng.random()
    if r < 0.15:
        g, _ = ORIGINALS[int(rng.integers(0, len(ORIGINALS)))]
        return to3(g), np.zeros((0, 4), np.float32)
    if r < 0.40:
        g, _ = paste_poison1(rand_bg1(rng), rng)
        return to3(g), np.zeros((0, 4), np.float32)
    g, _pb = paste_poison1(rand_bg1(rng), rng)
    g, sb = synth1(g, rng)
    return to3(g), sb

def sample_retain(rng):
    # trigger-free scenes for KD: 2/3 synthetic streaks, 1/3 plain backgrounds
    if rng.random() < (1.0 / 3.0):
        return to3(rand_bg1(rng))
    g, _ = synth1(rand_bg1(rng), rng)
    return to3(g)

HELDOUT = [paste_poison1(rand_bg1(rng), rng) for _ in range(CFG["N_HELDOUT"])]
AUDIT_SYN = [synth1(rand_bg1(rng), rng) for _ in range(CFG["N_AUDIT_SYN"])]

mg, mpb = paste_poison1(rand_bg1(rng), rng)
mg, msb = synth1(mg, rng)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
panels = [
    (HELDOUT[0][0], [("r", HELDOUT[0][1])], "heldout: pure pasted triggers"),
    (mg, [("r", mpb), ("lime", msb)], "mixed forget: triggers(red)=bg, synth(green)=GT"),
    (AUDIT_SYN[0][0], [("lime", AUDIT_SYN[0][1])], "retain/audit: synthetic streaks"),
    (BG_BANK[0], [], "retain: inpainted background"),
]
for ax, (im, boxsets, title) in zip(axes, panels):
    ax.imshow(im, cmap="gray")
    for color, bxs in boxsets:
        for (x1, y1, x2, y2) in bxs:
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, lw=1))
    ax.set_title(title, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()

# %%
# Teacher hit-rate on synthetic streaks, split by length. Short streaks (<90 px) live in the
# trigger-size regime: they are BOTH the proto negatives that make demotion trigger-specific
# (not just size-specific) AND the retention canaries for training.
det_flags, det_confs, det_lens = [], [], []
teacher_syn = []
for g, bxs in tqdm(AUDIT_SYN, desc="teacher @ synth"):
    boxes, scores = teacher(to3(g))
    keep = scores >= 0.2
    teacher_syn.append((boxes[keep], scores[keep]))
    m = iou_matrix(bxs, boxes)
    for i in range(len(bxs)):
        ok = bool(m.shape[1]) and m[i].max() >= 0.3
        det_flags.append(float(ok))
        det_lens.append(float(max(bxs[i, 2] - bxs[i, 0], bxs[i, 3] - bxs[i, 1])))
        if ok:
            det_confs.append(float(scores[m[i].argmax()]))
det_flags = np.array(det_flags); det_lens = np.array(det_lens)
SYN_HIT = float(det_flags.mean()) if len(det_flags) else 0.0
short = det_lens < 90
print(f"teacher hit-rate on synthetic streaks: {SYN_HIT:.2f}"
      f" | mean conf {np.mean(det_confs) if det_confs else 0:.2f}")
if short.any():
    print(f"  short (<90px): {det_flags[short].mean():.2f} over {int(short.sum())}"
          f" | long: {det_flags[~short].mean():.2f} over {int((~short).sum())}")
if SYN_HIT < 0.5:
    print("WARNING: synthetic streaks off-distribution; proto negatives and retention audit degraded")

# %%
# Poison prototype scoring in the teacher feature space: LDA over pooled P3 features,
# positives = poison GT + pasted views, negatives = teacher detections on synthetic streaks.
PROTO = None
PROTO_AUDIT = {
    "enabled": False,
    "positive_count": 0,
    "negative_count": 0,
    "standardized_separation": 0.0,
}
if CFG["USE_PROTO"]:
    pos_feats = []
    for fp in unlearn_files:
        _ = teacher(load_image(fp))
        gts = np.array([[x, y, x + w, y + h] for (x, y, w, h) in poison_anns.get(fp.name, [])], np.float32)
        if len(gts):
            pos_feats.append(teacher.pool_feats(gts))
    for _ in range(60):
        g, bxs = paste_poison1(rand_bg1(rng), rng)
        if len(bxs) == 0:
            continue
        _ = teacher(to3(g))
        pos_feats.append(teacher.pool_feats(bxs))
    neg_feats = []
    for (g, _), (tb, ts) in zip(AUDIT_SYN, teacher_syn):
        good = ts >= 0.3
        if good.any():
            _ = teacher(to3(g))
            neg_feats.append(teacher.pool_feats(tb[good]))
    if SYN_HIT < 0.4 or not neg_feats:
        print("proto disabled: no reliable clean prototypes")
        CFG["USE_PROTO"] = False
    else:
        pos = np.concatenate(pos_feats, 0); neg = np.concatenate(neg_feats, 0)
        mu_p, mu_n = pos.mean(0), neg.mean(0)
        Xc = np.concatenate([pos - mu_p, neg - mu_n], 0)
        S = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
        S = S + (0.5 * np.trace(S) / S.shape[0] + 1e-4) * np.eye(S.shape[0], dtype=S.dtype)
        wvec = np.linalg.solve(S, (mu_p - mu_n).astype(np.float64)).astype(np.float32)
        zb = -0.5 * float((mu_p + mu_n) @ wvec)
        zp, zn = pos @ wvec + zb, neg @ wvec + zb
        zs = 0.5 * (zp.std() + zn.std()) + 1e-6
        print(f"proto fit: {len(pos)} pos / {len(neg)} neg"
              f" | z_pos {zp.mean()/zs:+.2f} z_neg {zn.mean()/zs:+.2f} (want well separated)")
        PROTO_AUDIT = {
            "enabled": True,
            "positive_count": int(len(pos)),
            "negative_count": int(len(neg)),
            "standardized_separation": float(abs(zp.mean() - zn.mean()) / zs),
        }
        def PROTO(feats):
            z = (feats @ wvec + zb) / zs
            return (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))).astype(np.float32)
        if abs(zp.mean() - zn.mean()) / zs < 1.0:
            print("WARNING: weak proto separation -> consider W_PROTO=0 on the next A/B run")

# %%
resize_aug = T.ResizeShortestEdge(short_edge_length=800, max_size=1333)  # mirrors DefaultPredictor

def to_batched(img3, boxes_xyxy=None, train=True):
    tfm = resize_aug.get_transform(img3)
    img_r = tfm.apply_image(img3)
    d = {"image": torch.as_tensor(np.ascontiguousarray(img_r.transpose(2, 0, 1))),
         "height": IMG_H, "width": IMG_W}
    if train:
        h, w = img_r.shape[:2]
        inst = Instances((h, w))
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            inst.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
            inst.gt_classes = torch.zeros((0,), dtype=torch.int64)
        else:
            b = tfm.apply_box(np.asarray(boxes_xyxy, np.float32))
            inst.gt_boxes = Boxes(torch.as_tensor(b, dtype=torch.float32))
            inst.gt_classes = torch.zeros((len(b),), dtype=torch.int64)
        d["instances"] = inst
    return d

def head_out(model, batched):
    images = model.preprocess_image(batched)
    feats = model.backbone(images.tensor)
    feats = [feats[f] for f in model.head_in_features]
    return model.head(feats)

@torch.no_grad()
def infer_np(model, img3):
    was_training = model.training
    model.eval()
    with torch.autocast("cuda", enabled=(CFG["AMP_INFER"] and DEVICE == "cuda")):
        out = model([to_batched(img3, train=False)])[0]["instances"].to("cpu")
    if was_training:
        model.train()
    return (out.pred_boxes.tensor.numpy().astype(np.float32),
            out.scores.numpy().astype(np.float32))

@torch.no_grad()
def audit_suppression(model, scenes):
    vals = []
    for g, bxs in scenes:
        boxes, scores = infer_np(model, to3(g))
        m = iou_matrix(bxs, boxes)
        for i in range(len(bxs)):
            sel = (m[i] >= 0.3) if m.shape[1] else np.zeros(0, bool)
            vals.append(float(scores[sel].max()) if sel.any() else 0.0)
    return float(np.mean(vals)) if vals else 0.0

@torch.no_grad()
def audit_retention(model, scenes, refs):
    vals = []
    for (g, _), (tb, ts) in zip(scenes, refs):
        if len(tb) == 0:
            continue
        boxes, scores = infer_np(model, to3(g))
        m = iou_matrix(tb, boxes)
        for i in range(len(tb)):
            if m.shape[1] and m[i].max() >= 0.5:
                vals.append(min(1.5, float(scores[m[i].argmax()]) / max(float(ts[i]), 1e-6)))
            else:
                vals.append(0.0)
    return float(np.mean(vals)) if vals else 1.0

set_stage("unlearner_training")
student, UNLEARN_OK, RET_FINAL = None, False, 0.0
UNLEARN_AUDIT = {}
if CFG["DO_UNLEARN"]:
    cfg_s = build_cfg(POISONED_WEIGHTS, CFG["CAND_THRESH"])
    student = build_model(cfg_s)
    DetectionCheckpointer(student).load(str(POISONED_WEIGHTS))
    student.to(DEVICE)
    teacher_m = teacher.pred.model  # frozen reference (no_grad only)

    for n, p in student.named_parameters():
        if CFG["HEAD_ONLY"]:
            p.requires_grad_(n.startswith("head."))
        else:
            p.requires_grad_(not n.startswith("backbone.bottom_up"))
    trainable = [p for p in student.parameters() if p.requires_grad]
    anchor0 = {n: p.detach().clone() for n, p in student.named_parameters() if p.requires_grad}
    print(f"trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M params (HEAD_ONLY={CFG['HEAD_ONLY']})")

    opt = torch.optim.AdamW(trainable, lr=CFG["LR"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["ITERS"])
    best = {"score": -9.0, "it": -1, "state": None}

    t0 = time.time()
    for it in range(1, CFG["ITERS"] + 1):
        student.train()
        # ---- forget: mixed discriminative scenes (triggers unlabeled, synth streaks = GT) ----
        fb = [sample_forget(rng) for _ in range(CFG["BS_FORGET"])]
        batch_f = [to_batched(img, gtb, train=True) for img, gtb in fb]
        with EventStorage(it):  # detectron2 train-mode forward logs scalars; needs an active sink
            ld = student(batch_f)
        loss_forget = ld["loss_cls"] + ld.get("loss_box_reg", 0)
        # ---- retain: KD vs teacher on trigger-free scenes ----
        batch_r = [to_batched(sample_retain(rng), train=False) for _ in range(CFG["BS_RETAIN"])]
        with torch.no_grad():
            tl, td = head_out(teacher_m, batch_r)
        sl, sd = head_out(student, batch_r)
        kd_cls = torch.zeros((), device=DEVICE)
        kd_box = torch.zeros((), device=DEVICE)
        wsum = torch.zeros((), device=DEVICE)
        for s_l, t_l, s_d, t_d in zip(sl, tl, sd, td):
            tp = torch.sigmoid(t_l)
            w = torch.full_like(tp, CFG["BG_KD_W"])
            w[tp > CFG["FG_KD_THRESH"]] = CFG["FG_KD_W"]
            bce = F.binary_cross_entropy_with_logits(s_l, tp, reduction="none")
            kd_cls = kd_cls + (w * bce).sum()
            wsum = wsum + w.sum()
            w4 = tp.detach().repeat_interleave(4, dim=1)
            kd_box = kd_box + (w4 * (s_d - t_d).abs()).sum() / (w4.sum() + 1e-6)
        kd_cls = kd_cls / (wsum + 1e-6)
        # ---- L2-SP ----
        sp = torch.zeros((), device=DEVICE)
        for n, p in student.named_parameters():
            if p.requires_grad:
                sp = sp + ((p - anchor0[n]) ** 2).sum()
        loss = (CFG["W_FORGET"] * loss_forget + CFG["W_KD"] * kd_cls +
                CFG["W_KD_BOX"] * kd_box + CFG["W_SP"] * sp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, CFG["GRAD_CLIP"])
        opt.step(); sched.step()

        if it % CFG["EVAL_EVERY"] == 0 or it == CFG["ITERS"]:
            supp = audit_suppression(student, HELDOUT[:20])
            ret = audit_retention(student, AUDIT_SYN[:20], teacher_syn[:20])
            score = ret - supp
            TRAIN_HISTORY.append({
                "stage": "unlearner",
                "iteration": int(it),
                "loss": float(loss.detach()),
                "forget_loss": float(loss_forget.detach()),
                "kd_cls": float(kd_cls.detach()),
                "suppression_confidence": float(supp),
                "retention_ratio": float(ret),
                "checkpoint_objective": float(score),
            })
            log_event("TRAIN_AUDIT", **TRAIN_HISTORY[-1])
            print(f"[{it:4d}/{CFG['ITERS']}] forget {float(loss_forget.detach()):.3f}"
                  f" kd {float(kd_cls.detach()):.4f}"
                  f" | supp {supp:.3f} ret {ret:.3f} | {time.time()-t0:.0f}s")
            if score > best["score"]:
                best = {"score": score, "it": it,
                        "state": {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}}
    if best["state"] is not None:
        student.load_state_dict(best["state"])
        print(f"restored best checkpoint: iter {best['it']} (ret - supp = {best['score']:.3f})")
    UNLEARN_AUDIT = {
        "best_iteration": int(best["it"]),
        "best_checkpoint_objective": float(best["score"]),
    }
    torch.save({"model": student.state_dict()}, WORK / "depoisoned.pth")
else:
    print("unlearning skipped")

# ---------------- poison AMPLIFIER: contrastive probe ----------------
set_stage("amplifier_training")
amp = None
AMP_AUDIT = {}
if CFG["DO_AMP"]:
    cfg_a = build_cfg(POISONED_WEIGHTS, CFG["CAND_THRESH"])
    amp = build_model(cfg_a)
    DetectionCheckpointer(amp).load(str(POISONED_WEIGHTS))
    amp.to(DEVICE)
    for n, p in amp.named_parameters():
        p.requires_grad_(n.startswith("head."))
    tr_a = [p for p in amp.parameters() if p.requires_grad]
    opt_a = torch.optim.AdamW(tr_a, lr=CFG["AMP_LR"], weight_decay=0.0)
    t0 = time.time()
    for it in range(1, CFG["AMP_ITERS"] + 1):
        amp.train()
        batch = []
        for _ in range(2):
            g, pb = paste_poison1(rand_bg1(rng), rng)   # poison boxes -> POSITIVE GT
            g, _sb = synth1(g, rng)                     # synth streaks present but UNLABELED
            batch.append(to_batched(to3(g), pb, train=True))
        with EventStorage(it):
            la = amp(batch)
        loss = la["loss_cls"] + la.get("loss_box_reg", 0)
        opt_a.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tr_a, CFG["GRAD_CLIP"])
        opt_a.step()
        if it % 60 == 0 or it == CFG["AMP_ITERS"]:
            up = audit_suppression(amp, HELDOUT[:16])          # conf ON poison (want HIGH)
            dn = audit_retention(amp, AUDIT_SYN[:16], teacher_syn[:16])  # vs teacher on synth (want LOW)
            TRAIN_HISTORY.append({
                "stage": "amplifier",
                "iteration": int(it),
                "loss": float(loss.detach()),
                "poison_confidence": float(up),
                "synthetic_ratio": float(dn),
            })
            log_event("TRAIN_AUDIT", **TRAIN_HISTORY[-1])
            print(f"[amp {it:3d}/{CFG['AMP_ITERS']}] loss {float(loss.detach()):.3f}"
                  f" | poison-conf {up:.2f} (want>0.7) synth-ratio {dn:.2f} (want<0.8) | {time.time()-t0:.0f}s")
    amp.eval()
    AMP_AUDIT = {
        "final_poison_confidence": float(up),
        "final_synthetic_ratio": float(dn),
    }
    torch.save({"model": amp.state_dict()}, WORK / "amplifier.pth")

# %%
if student is not None:
    student.eval()
    supp_h = audit_suppression(student, HELDOUT)
    RET_FINAL = audit_retention(student, AUDIT_SYN, teacher_syn)
    print(f"unlearner: residual poison conf {supp_h:.3f} | synth retention {RET_FINAL:.3f}")
    print("note: the unlearner/amplifier are used as RANKERS (relative signal); absolute retention")
    print("below the old gate is expected and fine for ranking purposes.")


# %%
# Freeze every export rule before test files are enumerated or opened.
BASE_THR, EPS_Q, MID_Q, BOOST = 0.10, 0.02, 0.21, 0.15
MAX_KEEP_PER_IMG = 2


def fused_rank(r, a, p, s):
    return (
        1.6 * r
        + 0.9 * np.clip(1.0 - a, -0.5, 1.0)
        - 0.3 * np.clip(p - 0.5, 0, 1)
        + 0.5 * s
    )


NVARIANTS = {
    "N1_center": dict(rmin=0.45, amax=1.00, smin=0.30, mid=True, kq="boost"),
    "N2_ampstrict": dict(rmin=0.40, amax=0.85, smin=0.30, mid=True, kq="boost"),
    "N3_wide": dict(rmin=0.32, amax=1.10, smin=0.25, mid=True, kq="boost"),
    "N4_no_amp": dict(rmin=0.50, amax=9.99, smin=0.30, mid=True, kq="boost"),
    "N5_nomid": dict(rmin=0.45, amax=1.00, smin=0.30, mid=False, kq="boost"),
    "N6_flat95": dict(rmin=0.45, amax=1.00, smin=0.30, mid=True, kq=0.95),
}
SELECTION_LOCK = {
    "status": "frozen_before_test_read",
    "score_direction": "lower_is_better",
    "primary_finalist": "N1_center",
    "measured_control": {
        "candidate": "N4_no_amp",
        "public_reference_score": 216.54,
        "note": "Published M1-family control; current retraining may vary.",
    },
    "diverse_finalist": "N2_ampstrict",
    "variant_specs": NVARIANTS,
    "base_threshold": BASE_THR,
    "epsilon_confidence": EPS_Q,
    "mid_confidence": MID_Q,
    "boost": BOOST,
    "max_keep_per_image": MAX_KEEP_PER_IMG,
    "selection_sources": [
        "public unlearn annotations",
        "deterministic inpainted public backgrounds",
        "deterministic synthetic streak controls",
        "published NDR v7 rule family",
    ],
    "test_images_read": False,
    "test_predictions_used_for_selection": False,
    "leaderboard_per_image_feedback_used": False,
    "competition_submission_created": False,
}
(RUN_DIR / "selection_lock.json").write_text(
    json.dumps(SELECTION_LOCK, indent=2),
    encoding="utf-8",
)
set_stage("frozen_test_inference")


def stem_key(p):
    try:
        return (0, int(p.stem))
    except ValueError:
        return (1, p.stem)

test_files = sorted(Path(TEST_DIR).glob("*.png"), key=stem_key)
print(len(test_files), "test images")
assert len(test_files) == 2000, len(test_files)

def match_ratio(b, s, db, ds):
    r = np.zeros(len(b), np.float32)
    if len(db) and len(b):
        m = iou_matrix(b, db)
        bi, bv = m.argmax(1), m.max(1)
        ok = bv >= 0.5
        r[ok] = np.clip(ds[bi[ok]] / np.maximum(s[ok], 1e-6), 0, 2.5)
    return r

cand, ratio, ampr, protoscore = {}, {}, {}, {}
test_inference_started = time.time()
for image_index, fp in enumerate(tqdm(test_files, desc="test inference"), start=1):
    img = load_image(fp)
    b, s = teacher(img)
    cand[fp.stem] = (b, s)
    protoscore[fp.stem] = (PROTO(teacher.pool_feats(b))
                           if (CFG["USE_PROTO"] and PROTO is not None and len(b))
                           else np.zeros(len(b), np.float32))
    ratio[fp.stem] = (match_ratio(b, s, *infer_np(student, img))
                      if student is not None and len(b) else np.zeros(len(b), np.float32))
    ampr[fp.stem] = (match_ratio(b, s, *infer_np(amp, img))
                     if amp is not None and len(b) else np.ones(len(b), np.float32))
    if image_index % 100 == 0 or image_index == len(test_files):
        log_event(
            "INFERENCE_PROGRESS",
            completed=image_index,
            total=len(test_files),
            elapsed_sec=round(time.time() - test_inference_started, 1),
        )

n_t = sum(len(v[1]) for v in cand.values())
print(f"teacher candidates: {n_t} ({n_t/len(test_files):.2f}/img)")

stems, counts, Ball, Sall, Rall, Aall, Pall = [], [], [], [], [], [], []
for stem in sorted(cand, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
    b, s = cand[stem]
    stems.append(stem); counts.append(len(b))
    if len(b):
        Ball.append(b); Sall.append(s); Rall.append(ratio[stem])
        Aall.append(ampr[stem]); Pall.append(protoscore[stem])
Ball = np.concatenate(Ball) if Ball else np.zeros((0, 4), np.float32)
Sall = np.concatenate(Sall) if Sall else np.zeros(0, np.float32)
Rall = np.concatenate(Rall) if Rall else np.zeros(0, np.float32)
Aall = np.concatenate(Aall) if Aall else np.ones(0, np.float32)
Pall = np.concatenate(Pall) if Pall else np.zeros(0, np.float32)
np.savez_compressed(WORK / "artifacts_cands.npz",
                    stems=np.array(stems), counts=np.array(counts), boxes=Ball,
                    scores=Sall, ratio=Rall, amp=Aall, proto=Pall)
m02 = Sall >= 0.2
print("amp-ratio deciles (conf>=0.2):", np.percentile(Aall[m02], range(0, 101, 10)).round(2))
print("2D contrast check - amp-ratio by unlearn-ratio bucket:")
for lohi in ((0.0, 0.3), (0.5, 2.5)):
    mm = m02 & (Rall >= lohi[0]) & (Rall < lohi[1])
    if mm.any():
        print(f"  r in {lohi}: n={int(mm.sum())} amp median {np.median(Aall[mm]):.2f}"
              f" (expect HIGH for low-r poison, LOW for high-r real)")

# %%
# Anchors: floor S1=260.22, M1(v6)=216.54 (K=406, qbar=0.84, cbar=0.70 -> p~0.63).
# For any variant: p ~= ((260.22 - LB)/K + (qbar-0.02)) / ((cbar-0.05) + (qbar-0.02))
import json as _json
LB_N = {"N1_center": None, "N2_ampstrict": None, "N3_wide": None,
        "N4_no_amp": None, "N5_nomid": None, "N6_flat95": None,
        "M2_tight": None, "M5_nomid": None}
try:
    KI = _json.load(open("/kaggle/working/keep_info.json"))
except Exception:
    KI = {}
for name, lb in LB_N.items():
    if lb is None or name not in KI:
        continue
    K, kmass = KI[name]
    if K == 0:
        continue
    qbar = kmass / K
    cbar = max(qbar - 0.15, 0.2)
    p = ((260.22 - lb) / K + (qbar - 0.02)) / ((cbar - 0.05) + (qbar - 0.02))
    print(f"{name}: LB {lb} | K={K} qbar={qbar:.2f} -> precision ~ {min(max(p, 0), 1):.2f}")
print("target: p >= 0.72 at K ~ 450 projects <= 140. Recenter the rule toward whichever variant wins.")


# %%
def norm_id(x):
    try:
        return str(int(str(x)))
    except ValueError:
        return str(x)

def fmt(bx, sc):
    parts = []
    for (x1, y1, x2, y2), s in zip(bx, sc):
        x1 = float(np.clip(x1, 0, IMG_W)); y1 = float(np.clip(y1, 0, IMG_H))
        x2 = float(np.clip(x2, 0, IMG_W)); y2 = float(np.clip(y2, 0, IMG_H))
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0 and s > 0:
            parts += [f"{s:.6f}", f"{x1:.2f}", f"{y1:.2f}", f"{w:.2f}", f"{h:.2f}"]
    return " ".join(parts) or " "

set_stage("finalist_packaging")
sample = pd.read_csv(SAMPLE_SUB, dtype={"image_id": str})
KEEP_INFO = {}
print(f"{'variant':<12} {'keepK':>6} {'midK':>6} {'kmass':>7} {'mass':>7}")
for name, v in NVARIANTS.items():
    preds, kK, mK, kmass, mass = {}, 0, 0, 0.0, 0.0
    for stem, (b, s) in cand.items():
        m = s >= BASE_THR
        idxs = np.where(m)[0]
        kb = b[idxs]
        kq = np.full(len(idxs), EPS_Q, np.float32)
        r = ratio[stem][idxs]
        a = ampr[stem][idxs]
        pr = protoscore[stem][idxs] if len(protoscore.get(stem, [])) else np.zeros(len(idxs), np.float32)
        sc = s[idxs]
        keep_mask = (r >= v["rmin"]) & (a <= v["amax"]) & (sc >= v["smin"]) & (pr <= 0.95)
        if keep_mask.sum() > MAX_KEEP_PER_IMG:
            order = np.argsort(-fused_rank(r, a, pr, sc))
            allowed = set(order[:MAX_KEEP_PER_IMG].tolist())
            keep_mask = np.array([keep_mask[j] and (j in allowed) for j in range(len(idxs))])
        if v["mid"]:
            mid_mask = (~keep_mask) & (r >= 0.28) & (a <= 1.25) & (sc >= 0.20)
            kq[mid_mask] = MID_Q
            mK += int(mid_mask.sum())
        if v["kq"] == "boost":
            kq[keep_mask] = np.minimum(1.0, sc[keep_mask] + BOOST)
        else:
            kq[keep_mask] = float(v["kq"])
        kK += int(keep_mask.sum()); kmass += float(kq[keep_mask].sum())
        mass += float(kq.sum())
        preds[norm_id(stem)] = (kb, kq)
    sub = sample.copy()
    sub["prediction_string"] = sub["image_id"].map(
        lambda i: fmt(*preds.get(norm_id(i), (np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))))
    sub.to_csv(WORK / f"sub_{name}.csv", index=False)
    KEEP_INFO[name] = (kK, round(kmass, 1))
    print(f"{name:<12} {kK:>6} {mK:>6} {kmass:>7.0f} {mass:>7.0f}")

with open(WORK / "keep_info.json", "w") as f:
    json.dump(KEEP_INFO, f)
shutil.copy(WORK / "sub_N1_center.csv", WORK / "submission.csv")
shutil.copy(WORK / "sub_N4_no_amp.csv", WORK / "submission_measured_control.csv")
shutil.copy(WORK / "sub_N2_ampstrict.csv", WORK / "submission_amp_strict.csv")
print("submission.csv = N1_center.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_submission(path, sample_frame):
    frame = pd.read_csv(path, dtype={"image_id": str}, keep_default_na=False)
    assert list(frame.columns) == list(sample_frame.columns)
    assert len(frame) == 2000
    assert frame["image_id"].astype(str).is_unique
    assert frame["image_id"].astype(str).tolist() == sample_frame["image_id"].astype(str).tolist()
    total_boxes = 0
    empty_rows = 0
    confidence_mass = 0.0
    for prediction in frame["prediction_string"].astype(str):
        if not prediction.strip():
            empty_rows += 1
            continue
        values = [float(value) for value in prediction.split()]
        assert len(values) % 5 == 0
        for offset in range(0, len(values), 5):
            confidence, x, y, width, height = values[offset:offset + 5]
            assert 0 < confidence <= 1
            assert 0 <= x <= IMG_W and 0 <= y <= IMG_H
            assert width > 0 and height > 0
            assert x + width <= IMG_W + 0.05
            assert y + height <= IMG_H + 0.05
            total_boxes += 1
            confidence_mass += confidence
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "unique_image_ids": int(frame["image_id"].astype(str).nunique()),
        "total_boxes": int(total_boxes),
        "empty_rows": int(empty_rows),
        "confidence_mass": float(confidence_mass),
    }


submission_audit = {
    name: validate_submission(WORK / f"sub_{name}.csv", sample)
    for name in NVARIANTS
}
assert sha256(WORK / "submission.csv") == submission_audit["N1_center"]["sha256"]
assert (
    sha256(WORK / "submission_measured_control.csv")
    == submission_audit["N4_no_amp"]["sha256"]
)
assert (
    sha256(WORK / "submission_amp_strict.csv")
    == submission_audit["N2_ampstrict"]["sha256"]
)

pd.DataFrame(TRAIN_HISTORY).to_csv(RUN_DIR / "training_history.csv", index=False)
(RUN_DIR / "contrastive_config.json").write_text(
    json.dumps(
        {
            "config": CFG,
            "selection_lock": SELECTION_LOCK,
            "runtime": {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_arch": gpu_arch,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)

low_ratio_mask = m02 & (Rall >= 0.0) & (Rall < 0.3)
high_ratio_mask = m02 & (Rall >= 0.5) & (Rall < 2.5)
contrast_audit = {
    "teacher_candidates": int(n_t),
    "teacher_candidates_per_image": float(n_t / len(test_files)),
    "confidence_ge_020_count": int(m02.sum()),
    "low_unlearn_ratio_count": int(low_ratio_mask.sum()),
    "high_unlearn_ratio_count": int(high_ratio_mask.sum()),
    "low_ratio_amp_median": (
        float(np.median(Aall[low_ratio_mask])) if low_ratio_mask.any() else None
    ),
    "high_ratio_amp_median": (
        float(np.median(Aall[high_ratio_mask])) if high_ratio_mask.any() else None
    ),
    "amplifier_separation_direction_passed": bool(
        low_ratio_mask.any()
        and high_ratio_mask.any()
        and np.median(Aall[low_ratio_mask]) > np.median(Aall[high_ratio_mask])
    ),
}
model_manifest = {}
for name, path in {
    "unlearner": WORK / "depoisoned.pth",
    "amplifier": WORK / "amplifier.pth",
}.items():
    assert path.exists(), path
    model_manifest[name] = {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": int(path.stat().st_size),
    }

final_report = {
    "status": "complete",
    "bundle": "NDR_CONTRASTIVE_V9",
    "score_direction": "lower_is_better",
    "current_team_incumbent": 229.2314,
    "published_measured_control_reference": 216.54,
    "primary_finalist": "N1_center",
    "measured_family_control": "N4_no_amp",
    "diverse_finalist": "N2_ampstrict",
    "keep_info": KEEP_INFO,
    "submission_audit": submission_audit,
    "unlearner_audit": {
        **UNLEARN_AUDIT,
        "heldout_poison_confidence": float(supp_h),
        "synthetic_retention_ratio": float(RET_FINAL),
    },
    "amplifier_audit": AMP_AUDIT,
    "prototype_audit": PROTO_AUDIT,
    "synthetic_teacher_hit_rate": float(SYN_HIT),
    "contrast_audit": contrast_audit,
    "model_manifest": model_manifest,
    "rule_7a_guard": {
        "external_models": False,
        "manual_test_labels": False,
        "automatic_test_labels": False,
        "test_images_read_before_selection_lock": False,
        "test_predictions_used_for_selection": False,
        "competition_submission_created": False,
    },
    "projection_note": (
        "The source notebook's N1 projection is unverified. Only an actual "
        "competition submission can establish its leaderboard score."
    ),
}
(RUN_DIR / "final_report.json").write_text(
    json.dumps(final_report, indent=2),
    encoding="utf-8",
)
(RUN_DIR / "submission_audit.json").write_text(
    json.dumps(submission_audit, indent=2),
    encoding="utf-8",
)
STOP_HEARTBEAT.set()
log_event("RUN_COMPLETE", report=final_report)
print(json.dumps(final_report, indent=2))

# %% [markdown]
# ## Today (5 slots)
# 1. **submission.csv (= N1_center)** - fused 2D rule. If amp-contrast works, projects 150-185; with the
#    flat-0.95/threshold recentering tomorrow, ≤140 is the target.
# 2. **sub_M2_tight.csv from the v6 output** (no rerun) - precision at the K=219 high-precision end.
# 3. **sub_M5_nomid.csv from the v6 output** - isolates the 642-box mid tier's sign.
# 4. **N2_ampstrict** - the amp gate at its precision end.
# 5. **N4_no_amp** - exact v6-M1 rule rebuilt here: N1 − N4 is the amp signal's measured value.
#
# ## Tomorrow (final tuning days)
# Paste scores into cell 13, rerun it alone: it converts every LB into an implied precision. Recenter
# `rmin/amax` toward the winner; if p ≥ 0.72 shows up anywhere, the 140 push is: that rule + K to ~500 via
# per-image cap 3 + flat-0.95 keeps (N6 tests it). The 2D contrast printout in cell 12 (amp median by r-bucket)
# tells you *before submitting* whether the amplifier separated - if amp medians are equal across r-buckets,
# skip N2 and spend the slots on M-recentering instead.
#
