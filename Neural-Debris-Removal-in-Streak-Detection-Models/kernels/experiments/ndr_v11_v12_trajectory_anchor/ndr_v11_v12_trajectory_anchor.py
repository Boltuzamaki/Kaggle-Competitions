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
# # NDR V11 - exact V12/211 anchor + training-trajectory rankers
# This run first reproduces the public **211.3036** V12 recipe without changing its M1
# selection or scoring. It additionally saves head checkpoints and measures their survivor
# ratios on the test candidates. All trajectory fusions are frozen before test enumeration.
#
# `submission.csv` is always the exact M1 anchor. No competition submission is created.
#

# %%
# =====================  CONFIG  =====================
CFG = dict(
    DO_UNLEARN = True,
    DO_AMP = False,
    HEAD_ONLY = True, ITERS = 400, LR = 4e-5, BS_FORGET = 2, BS_RETAIN = 2,
    W_FORGET = 1.0, W_KD = 5.0, W_KD_BOX = 0.5, W_SP = 1e-4,
    FG_KD_THRESH = 0.05, FG_KD_W = 2.0, BG_KD_W = 0.02,
    AMP_ITERS = 180, AMP_LR = 6e-5,
    EVAL_EVERY = 50, GRAD_CLIP = 1.0, RET_GATE = 0.80,
    N_HELDOUT = 24, N_AUDIT_SYN = 32, POISON_PER_SCENE = 3,
    SYN_SNR = (4.0, 30.0), SYN_LEN = (24, 90), SYN_SIGMA = (1.2, 3.2),
    CAND_THRESH = 0.02, AMP_INFER = True, USE_PROTO = True,
    TAU_P = 0.85, FINAL_NMS_IOU = 0.35, TOPK_PER_IMAGE = 25,
    TRAJECTORY_STEPS = (100, 150, 200, 250, 300),
    PUBLIC_V12_SCORE = 211.3036,
    PUBLIC_V12_CSV_SHA256 = "c4eaa3df750879a6999a752ddbd0e3f6a9d473b85e8ada323ccf663b8c321cdd",
    SEED = 42,
)
for k, v in CFG.items():
    print(f"  {k:>16} = {v}")

# %%
# Kaggle's current image ships torch 2.10+cu128, whose wheels dropped Pascal (sm_60) kernels.
# P100 = sm_60 -> "no kernel image is available". T4 = sm_75 -> fine as-is.
# This cell detects the GPU BEFORE importing torch and, on Pascal, downgrades to
# torch 2.6.0+cu124 (last widely-used Kaggle pairing that ships sm_60 kernels),
# then builds detectron2 against whichever torch is active.
import subprocess

def sh(cmd, note=""):
    if note:
        print(note)
    print(">>", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    print("   " + "; ".join(tail[-4:]) if tail else "   (no output)")
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-3000:])
        raise RuntimeError(f"command failed: {cmd}")

q = subprocess.run("nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader",
                   shell=True, capture_output=True, text=True).stdout.strip().splitlines()
cap_s, gpu_name = ("7.5", "unknown")
if q:
    parts = [p.strip() for p in q[0].split(",")]
    cap_s = parts[0]
    gpu_name = parts[1] if len(parts) > 1 else "unknown"
try:
    CAP = float(cap_s)
except ValueError:
    CAP = 6.0 if "P100" in gpu_name.upper() else 7.5
print(f"GPU: {gpu_name} | compute capability {CAP}")

sh("pip install -q 'setuptools<81'")
if CAP < 7.0:
    sh("pip install -q torch==2.6.0 torchvision==0.21.0"
       " --index-url https://download.pytorch.org/whl/cu124",
       note="Pascal GPU: downgrading torch to a build with sm_60 kernels (~2.5 GB, 3-6 min)")
sh("pip install -q --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'",
   note="building detectron2 against the active torch (~5-8 min)")

import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| arch list:", torch.cuda.get_arch_list())
try:
    val = float((torch.ones(8, device="cuda") * 2).sum().item())  # real kernel launch
    print(f"CUDA smoke test OK ({val}) on", torch.cuda.get_device_name(0))
except Exception as e:
    raise SystemExit(f"CUDA smoke test FAILED on {torch.cuda.get_device_name(0)}: {e}."
                     " Switch Settings -> Accelerator -> GPU T4 x2 and rerun.")

# %%
import copy, glob, hashlib, json, math, os, random, threading, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
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

# ---- robust competition-path resolution ----
CAND_ROOTS = [
    "/kaggle/input/neural-debris-removal-in-streak-detection-models",
    "/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models",
]
ROOT = next((r for r in CAND_ROOTS if os.path.isdir(r)), None)
if ROOT is None:
    hits = glob.glob("/kaggle/input/**/annotations_coco.json", recursive=True)
    assert hits, "competition data not found under /kaggle/input"
    ROOT = str(Path(hits[0]).parents[1])

pth_hits = sorted(glob.glob(str(Path(ROOT) / "**" / "*.pth"), recursive=True))
assert pth_hits, "no .pth under ROOT"
POISONED_WEIGHTS = next((p for p in pth_hits if "poison" in p.lower()), pth_hits[0])
UNLEARN_JSON = sorted(glob.glob(str(Path(ROOT) / "**" / "annotations_coco.json"), recursive=True))[0]
UNLEARN_DIR = str(Path(UNLEARN_JSON).parent)
png_dirs = {}
for p in glob.glob(str(Path(ROOT) / "**" / "*.png"), recursive=True):
    d = str(Path(p).parent); png_dirs[d] = png_dirs.get(d, 0) + 1
TEST_DIR = max(png_dirs, key=png_dirs.get)
SAMPLE_SUB = sorted(glob.glob(str(Path(ROOT) / "**" / "sample_submission.csv"), recursive=True))[0]
WORK = Path("/kaggle/working"); WORK.mkdir(parents=True, exist_ok=True)
RUN_DIR = WORK / "ndr_v11_v12_trajectory"
RUN_DIR.mkdir(parents=True, exist_ok=True)
TRAJECTORY_DIR = RUN_DIR / "trajectory_checkpoints"
TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = RUN_DIR / "run.jsonl"
RUN_STAGE = {"name": "initialization"}
STOP_HEARTBEAT = threading.Event()

def log_event(event, **payload):
    record = {"ts": time.time(), "event": event, "stage": RUN_STAGE["name"], **payload}
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    print("[LOG]", json.dumps(record, default=str), flush=True)

def set_stage(name):
    RUN_STAGE["name"] = name
    log_event("STAGE", name=name)

def heartbeat():
    while not STOP_HEARTBEAT.wait(60):
        log_event("HEARTBEAT")

threading.Thread(target=heartbeat, daemon=True).start()
log_event("CONFIG_FROZEN", config=CFG)

print("ROOT   :", ROOT)
print("weights:", POISONED_WEIGHTS)
print("unlearn:", UNLEARN_DIR, "->", len(glob.glob(UNLEARN_DIR + "/*.png")), "png")
print("test   :", TEST_DIR, "->", png_dirs[TEST_DIR], "png")
print("sample :", SAMPLE_SUB)

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

student, UNLEARN_OK, RET_FINAL = None, False, 0.0
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
    trajectory_manifest = []

    set_stage("v12_anchor_training")
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
            print(f"[{it:4d}/{CFG['ITERS']}] forget {float(loss_forget.detach()):.3f}"
                  f" kd {float(kd_cls.detach()):.4f}"
                  f" | supp {supp:.3f} ret {ret:.3f} | {time.time()-t0:.0f}s")
            log_event(
                "TRAIN_EVAL", iteration=it, forget=float(loss_forget.detach()),
                kd=float(kd_cls.detach()), suppression=supp, retention=ret,
                objective=score, elapsed_seconds=time.time() - t0,
            )
            if it in CFG["TRAJECTORY_STEPS"]:
                checkpoint_path = TRAJECTORY_DIR / f"head_iter_{it:03d}.pth"
                head_state = {
                    k: v.detach().cpu().clone()
                    for k, v in student.state_dict().items()
                    if k.startswith("head.")
                }
                torch.save(
                    {
                        "head_state": head_state,
                        "iteration": it,
                        "suppression": supp,
                        "retention": ret,
                        "objective": score,
                    },
                    checkpoint_path,
                )
                trajectory_manifest.append(
                    {
                        "iteration": it,
                        "file": checkpoint_path.name,
                        "suppression": supp,
                        "retention": ret,
                        "objective": score,
                    }
                )
            if score > best["score"]:
                best = {"score": score, "it": it,
                        "state": {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}}
    if best["state"] is not None:
        student.load_state_dict(best["state"])
        print(f"restored best checkpoint: iter {best['it']} (ret - supp = {best['score']:.3f})")
    torch.save({"model": student.state_dict()}, WORK / "depoisoned.pth")
    (RUN_DIR / "trajectory_manifest.json").write_text(
        json.dumps(trajectory_manifest, indent=2), encoding="utf-8"
    )
    log_event("BEST_CHECKPOINT", iteration=best["it"], objective=best["score"])
else:
    print("unlearning skipped")

# ---------------- poison AMPLIFIER: contrastive probe ----------------
amp = None
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
            print(f"[amp {it:3d}/{CFG['AMP_ITERS']}] loss {float(loss.detach()):.3f}"
                  f" | poison-conf {up:.2f} (want>0.7) synth-ratio {dn:.2f} (want<0.8) | {time.time()-t0:.0f}s")
    amp.eval()
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
BASE_THR, EPS_Q, MID_Q, BOOST = 0.10, 0.02, 0.21, 0.15
MAX_KEEP_PER_IMG = 2
MVARIANTS = {
    "M1_center":  dict(keep=dict(r=0.50, s=0.30, pmax=0.92), mid=dict(r=0.30, s=0.20), kq="boost", ratio="best"),
    "M2_tight":   dict(keep=dict(r=0.70, s=0.35, pmax=0.90), mid=None,                  kq="boost", ratio="best"),
    "M3_wide":    dict(keep=dict(r=0.35, s=0.25, pmax=0.95), mid=dict(r=0.20, s=0.15), kq="boost", ratio="best"),
    "M4_rawkeep": dict(keep=dict(r=0.50, s=0.30, pmax=0.92), mid=dict(r=0.30, s=0.20), kq="raw",   ratio="best"),
    "M5_nomid":   dict(keep=dict(r=0.50, s=0.30, pmax=0.92), mid=None,                  kq="boost", ratio="best"),
    "M6_flat9":   dict(keep=dict(r=0.50, s=0.30, pmax=0.92), mid=dict(r=0.30, s=0.20), kq=0.90,    ratio="best"),
    # New candidates. Their formulas are frozen here, before test file enumeration.
    "T1_median":  dict(keep=dict(r=0.50, s=0.30, pmax=0.92), mid=dict(r=0.30, s=0.20), kq="boost", ratio="median"),
    "T2_q25":     dict(keep=dict(r=0.42, s=0.30, pmax=0.92), mid=dict(r=0.25, s=0.20), kq="boost", ratio="q25"),
    "T3_stable":  dict(keep=dict(r=0.48, s=0.30, pmax=0.92), mid=dict(r=0.28, s=0.20), kq="boost", ratio="stable"),
}
SELECTION_LOCK = {
    "status": "frozen_before_test_enumeration",
    "score_direction": "lower_is_better",
    "anchor": "M1_center",
    "public_reference_score": CFG["PUBLIC_V12_SCORE"],
    "public_reference_csv_sha256": CFG["PUBLIC_V12_CSV_SHA256"],
    "trajectory_steps": list(CFG["TRAJECTORY_STEPS"]),
    "candidates": MVARIANTS,
    "selection_data": ["public_unlearn_images", "public_unlearn_annotations", "deterministic_synthetic_controls"],
    "test_pixels_used_for_selection": False,
    "leaderboard_used_for_candidate_selection": False,
    "competition_submission_created": False,
}
(RUN_DIR / "selection_lock.json").write_text(
    json.dumps(SELECTION_LOCK, indent=2), encoding="utf-8"
)
set_stage("frozen_test_inference")

def stem_key(p):
    try:
        return (0, int(p.stem))
    except ValueError:
        return (1, p.stem)

test_files = sorted(Path(TEST_DIR).glob("*.png"), key=stem_key)
print(len(test_files), "test images")

def match_ratio(b, s, db, ds):
    r = np.zeros(len(b), np.float32)
    if len(db) and len(b):
        m = iou_matrix(b, db)
        bi, bv = m.argmax(1), m.max(1)
        ok = bv >= 0.5
        r[ok] = np.clip(ds[bi[ok]] / np.maximum(s[ok], 1e-6), 0, 2.5)
    return r

cand, ratio, ampr, protoscore = {}, {}, {}, {}
for fp in tqdm(test_files, desc="test inference"):
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

n_t = sum(len(v[1]) for v in cand.values())
print(f"teacher candidates: {n_t} ({n_t/len(test_files):.2f}/img)")

# Measure the same fixed teacher boxes with every saved training-time ranker.
# This creates test-time signals only; no labels, pseudo-labels, or selection are produced.
trajectory_ratio = {}
for meta in trajectory_manifest:
    it = int(meta["iteration"])
    payload = torch.load(
        TRAJECTORY_DIR / meta["file"], map_location="cpu", weights_only=False
    )
    student.load_state_dict(payload["head_state"], strict=False)
    student.eval()
    per_stem = {}
    t0 = time.time()
    for fp in tqdm(test_files, desc=f"trajectory iter {it}"):
        b, s = cand[fp.stem]
        per_stem[fp.stem] = (
            match_ratio(b, s, *infer_np(student, load_image(fp)))
            if len(b) else np.zeros(0, np.float32)
        )
    trajectory_ratio[it] = per_stem
    log_event("TRAJECTORY_INFERENCE_COMPLETE", iteration=it, elapsed_seconds=time.time() - t0)

# Restore the exact best checkpoint after auxiliary measurements.
student.load_state_dict(
    torch.load(WORK / "depoisoned.pth", map_location="cpu", weights_only=False)["model"]
)
student.to(DEVICE).eval()

ratio_sources = {"best": ratio}
for stem in cand:
    stack = np.stack(
        [trajectory_ratio[it][stem] for it in CFG["TRAJECTORY_STEPS"]], axis=0
    ) if len(cand[stem][0]) else np.zeros((len(CFG["TRAJECTORY_STEPS"]), 0), np.float32)
    ratio_sources.setdefault("median", {})[stem] = np.median(stack, axis=0)
    ratio_sources.setdefault("q25", {})[stem] = np.quantile(stack, 0.25, axis=0)
    med = np.median(stack, axis=0)
    spread = np.quantile(stack, 0.75, axis=0) - np.quantile(stack, 0.25, axis=0)
    ratio_sources.setdefault("stable", {})[stem] = np.clip(med - 0.5 * spread, 0, 2.5)

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
trajectory_payload = {}
for it in CFG["TRAJECTORY_STEPS"]:
    trajectory_payload[f"ratio_iter_{it}"] = np.concatenate(
        [trajectory_ratio[it][stem] for stem in stems]
    ) if len(Rall) else np.zeros(0, np.float32)
for source in ("median", "q25", "stable"):
    trajectory_payload[f"ratio_{source}"] = np.concatenate(
        [ratio_sources[source][stem] for stem in stems]
    ) if len(Rall) else np.zeros(0, np.float32)
np.savez_compressed(
    RUN_DIR / "trajectory_signals.npz",
    stems=np.array(stems), counts=np.array(counts), boxes=Ball, scores=Sall,
    proto=Pall, ratio_best=Rall, **trajectory_payload,
)
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

sample = pd.read_csv(SAMPLE_SUB, dtype={"image_id": str})
KEEP_INFO = {}
print(f"{'variant':<12} {'keepK':>6} {'midK':>6} {'boxes':>6} {'mass':>7}")
for name, v in MVARIANTS.items():
    preds, kK, mK, nb, mass = {}, 0, 0, 0, 0.0
    for stem, (b, s) in cand.items():
        m = s >= BASE_THR
        idxs = np.where(m)[0]
        kb = b[idxs]
        kq = np.full(len(idxs), EPS_Q, np.float32)
        r = ratio_sources[v["ratio"]][stem][idxs]
        pr = protoscore[stem][idxs] if len(protoscore.get(stem, [])) else np.zeros(len(idxs), np.float32)
        sc = s[idxs]
        krule = v["keep"]
        keep_mask = (r >= krule["r"]) & (sc >= krule["s"]) & (pr <= krule["pmax"])
        if keep_mask.sum() > MAX_KEEP_PER_IMG:
            rank = 2.0 * r - 0.6 * pr + sc
            order = np.argsort(-rank)
            allowed = set(order[:MAX_KEEP_PER_IMG].tolist())
            keep_mask = np.array([keep_mask[j] and (j in allowed) for j in range(len(idxs))])
        if v["mid"] is not None:
            mid_mask = (~keep_mask) & (r >= v["mid"]["r"]) & (sc >= v["mid"]["s"])
            kq[mid_mask] = MID_Q
            mK += int(mid_mask.sum())
        if v["kq"] == "boost":
            kq[keep_mask] = np.minimum(1.0, sc[keep_mask] + BOOST)
        elif v["kq"] == "raw":
            kq[keep_mask] = sc[keep_mask]
        else:
            kq[keep_mask] = float(v["kq"])
        kK += int(keep_mask.sum())
        nb += len(kb)
        mass += float(kq.sum())
        preds[norm_id(stem)] = (kb, kq)
    sub = sample.copy()
    sub["prediction_string"] = sub["image_id"].map(
        lambda i: fmt(*preds.get(norm_id(i), (np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))))
    sub.to_csv(WORK / f"sub_{name}.csv", index=False)
    KEEP_INFO[name] = (kK, round(mass, 1))
    print(f"{name:<12} {kK:>6} {mK:>6} {nb:>6} {mass:>7.0f}")

with open(WORK / "keep_info.json", "w") as f:
    json.dump(KEEP_INFO, f)
import shutil
shutil.copy(WORK / "sub_M1_center.csv", WORK / "submission.csv")
shutil.copy(WORK / "sub_M1_center.csv", RUN_DIR / "submission_anchor_M1.csv")
for name in MVARIANTS:
    shutil.copy(WORK / f"sub_{name}.csv", RUN_DIR / f"sub_{name}.csv")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

submission_audit = {}
for name in MVARIANTS:
    path = WORK / f"sub_{name}.csv"
    df = pd.read_csv(path, dtype={"image_id": str})
    submission_audit[name] = {
        "rows": len(df),
        "unique_ids": int(df["image_id"].nunique()),
        "sha256": sha256(path),
        "keep_info": KEEP_INFO[name],
    }
(RUN_DIR / "submission_audit.json").write_text(
    json.dumps(submission_audit, indent=2), encoding="utf-8"
)
final_report = {
    "status": "complete",
    "bundle": "NDR_V11_V12_TRAJECTORY_ANCHOR",
    "anchor": "M1_center",
    "public_v12_reference_score": CFG["PUBLIC_V12_SCORE"],
    "public_v12_reference_csv_sha256": CFG["PUBLIC_V12_CSV_SHA256"],
    "generated_anchor_csv_sha256": submission_audit["M1_center"]["sha256"],
    "anchor_hash_exact_match": submission_audit["M1_center"]["sha256"] == CFG["PUBLIC_V12_CSV_SHA256"],
    "best_training_iteration": int(best["it"]),
    "best_training_objective": float(best["score"]),
    "trajectory_candidates": ["T1_median", "T2_q25", "T3_stable"],
    "rule_7a": {
        "selection_frozen_before_test_enumeration": True,
        "test_labels_or_pseudo_labels_created": False,
        "test_predictions_used_for_selection": False,
        "competition_submission_created": False,
    },
}
(RUN_DIR / "final_report.json").write_text(
    json.dumps(final_report, indent=2), encoding="utf-8"
)
STOP_HEARTBEAT.set()
log_event("RUN_COMPLETE", report=final_report)
print("submission.csv = exact M1 anchor; no competition submission was created.")
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
