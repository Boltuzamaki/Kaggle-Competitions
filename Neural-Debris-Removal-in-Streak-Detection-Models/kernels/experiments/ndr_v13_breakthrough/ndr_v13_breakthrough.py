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
# # NDR V13 breakthrough bundle
#
# Three deliberately different hypotheses share one GPU pass:
#
# 1. **Poison task-vector reversal** estimates the local direction produced by
#    additional poison training, protects high-Fisher parameters, and subtracts
#    that direction from the supplied checkpoint.
# 2. **Activation + physical ranker** learns a small regularised poison-vs-clean
#    classifier from public unlearn boxes and predeclared synthetic streak
#    renderers. It uses P3/P4 ROI activations plus line-profile morphology.
# 3. **Conditional feature-subspace projection** derives a poison direction in
#    P3/P4 and removes it only at spatial locations whose activation resembles
#    the public poison concept.
#
# Candidate selection and confidence mappings are frozen before the test folder
# is enumerated. Test pixels are used only for deterministic inference. This
# kernel writes model/diagnostic/CSV artifacts and never calls the Kaggle
# submission API.

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
print("[SETUP] Building Detectron2 for Tesla T4 (SM 7.5)", flush=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation",
     "git+https://github.com/facebookresearch/detectron2.git"],
    check=True,
)

# %%
import copy
import gc
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
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage

SEED = 260721
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "V13 is a GPU experiment"

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
if not ROOT.exists():
    ROOT = Path("/kaggle/input/neural-debris-removal-in-streak-detection-models")
POISONED_WEIGHTS = ROOT / "poisoned_model" / "poisoned_model.pth"
UNLEARN_DIR = ROOT / "unlearn_set"
UNLEARN_JSON = UNLEARN_DIR / "annotations_coco.json"
TEST_DIR = ROOT / "test_set" / "test_set"
SAMPLE_SUB = ROOT / "sample_submission.csv"
for p in (POISONED_WEIGHTS, UNLEARN_JSON, TEST_DIR, SAMPLE_SUB):
    assert p.exists(), p

OUT = Path("/kaggle/working/ndr_v13")
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "run.jsonl"
IMG_W = IMG_H = 1024
BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ASPECTS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
SIZES = [[16], [32], [64], [128], [256]]
CAND_THRESH = 0.05
MATCH_IOU = 0.50

def log(message, **payload):
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "message": message, **payload}
    print(json.dumps(row, default=str), flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def heartbeat(label, stop):
    started = time.time()
    while not stop.wait(45):
        log("HEARTBEAT", stage=label, elapsed_min=round((time.time() - started) / 60, 1))

class Heartbeat:
    def __init__(self, label): self.label = label
    def __enter__(self):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=heartbeat, args=(self.label, self.stop), daemon=True)
        self.thread.start(); log("STAGE_START", stage=self.label)
        return self
    def __exit__(self, typ, value, tb):
        self.stop.set(); self.thread.join(timeout=2)
        log("STAGE_END", stage=self.label, ok=typ is None)

# Selection policy is committed before test enumeration.
VARIANTS = {
    "V13_A_ranker_strict": {"signal": "rank", "lo": 0.22, "hi": 0.72, "keep": 0.92, "mid": 0.22},
    "V13_B_taskvector": {"signal": "task", "lo": 0.22, "hi": 0.68, "keep": 0.90, "mid": 0.22},
    "V13_C_projection": {"signal": "projection", "lo": 0.22, "hi": 0.68, "keep": 0.90, "mid": 0.22},
    "V13_D_consensus": {"signal": "consensus", "lo": 0.24, "hi": 0.66, "keep": 0.94, "mid": 0.20},
    "V13_E_unanimous": {"signal": "unanimous", "lo": 0.20, "hi": 0.62, "keep": 0.96, "mid": 0.28},
}
SELECTION_LOCK = {
    "experiment": "V13_BREAKTHROUGH_BUNDLE",
    "seed": SEED,
    "methods": ["fisher_task_vector_reversal", "activation_physical_ranker",
                "conditional_p3_p4_projection"],
    "task_vector_alphas": [0.5, 1.0, 2.0, 4.0],
    "projection_strengths": [0.25, 0.5, 0.75, 1.0],
    "ranker_gate": {"oof_auc_min": 0.72, "worst_clean_family_fpr_max": 0.40},
    "variants": VARIANTS,
    "alias": "V13_D_consensus",
    "rule_7a": {
        "selection_sources": "public unlearn set plus deterministic synthetic controls only",
        "test_pixels_used_for_training_or_selection": False,
        "test_enumerated_after_lock": True,
        "competition_submission_created": False,
    },
}
(OUT / "selection_lock.json").write_text(json.dumps(SELECTION_LOCK, indent=2), encoding="utf-8")
log("SELECTION_LOCK_WRITTEN", lock=SELECTION_LOCK)

# %% [markdown]
# ## Shared model, image and geometry helpers

# %%
def build_cfg(weights=POISONED_WEIGHTS, thresh=CAND_THRESH):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(thresh)
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ASPECTS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = SIZES
    cfg.MODEL.DEVICE = DEVICE
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    return cfg

def load_image(path):
    g = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert g is not None, path
    if g.dtype == np.uint16:
        g = g.astype(np.float32) / 65535.0 * 255.0
    elif g.dtype == np.uint8:
        g = g.astype(np.float32)
    else:
        g = g.astype(np.float32)
        if float(g.max()) <= 1.0: g *= 255.0
    g = np.clip(g, 0, 255).astype(np.float32)
    if g.ndim == 3: g = g[:, :, 0]
    return np.repeat(g[:, :, None], 3, axis=2)

def iou_matrix(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    aa = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    bb = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-6)

def match_scores(ref_boxes, pred_boxes, pred_scores, iou=MATCH_IOU):
    out = np.zeros(len(ref_boxes), np.float32)
    if len(ref_boxes) and len(pred_boxes):
        m = iou_matrix(ref_boxes, pred_boxes)
        for i in range(len(ref_boxes)):
            valid = np.where(m[i] >= iou)[0]
            if len(valid): out[i] = float(pred_scores[valid[np.argmax(m[i, valid])]])
    return out

class CapturePredictor:
    def __init__(self, weights=POISONED_WEIGHTS, thresh=CAND_THRESH):
        self.pred = DefaultPredictor(build_cfg(weights, thresh))
        self.features = None
        self.hook = self.pred.model.backbone.register_forward_hook(self._capture)
    def _capture(self, module, inputs, output):
        self.features = {k: v.detach() for k, v in output.items() if k in ("p3", "p4")}
    def __call__(self, image):
        with torch.inference_mode(), torch.autocast("cuda", enabled=True):
            out = self.pred(image)["instances"].to("cpu")
        return out.pred_boxes.tensor.numpy().astype(np.float32), out.scores.numpy().astype(np.float32)
    def pool_level(self, boxes, level):
        fmap = self.features[level][0].float(); c, hf, wf = fmap.shape
        stride = 8 if level == "p3" else 16
        scale = (hf * stride) / IMG_H
        pooled = np.zeros((len(boxes), c), np.float32)
        for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes)):
            fx1 = int(np.clip(np.floor(x1 * scale / stride), 0, wf - 1))
            fy1 = int(np.clip(np.floor(y1 * scale / stride), 0, hf - 1))
            fx2 = int(np.clip(np.ceil(x2 * scale / stride), fx1 + 1, wf))
            fy2 = int(np.clip(np.ceil(y2 * scale / stride), fy1 + 1, hf))
            pooled[i] = fmap[:, fy1:fy2, fx1:fx2].mean((1, 2)).cpu().numpy()
        return pooled
    def pooled(self, boxes):
        return np.concatenate([self.pool_level(boxes, "p3"), self.pool_level(boxes, "p4")], 1)

with UNLEARN_JSON.open() as f: coco = json.load(f)
id2name = {int(x["id"]): x["file_name"] for x in coco["images"]}
poison_by_name = {}
for ann in coco["annotations"]:
    x, y, w, h = map(float, ann["bbox"])
    poison_by_name.setdefault(id2name[int(ann["image_id"])], []).append([x, y, x + w, y + h])
unlearn_files = sorted(UNLEARN_DIR.glob("*.png"))
assert sum(map(len, poison_by_name.values())) == 20

def inpaint_boxes(g, boxes, pad=8):
    out = g.copy(); gray = out[:, :, 0]
    rng = np.random.default_rng(SEED + 17)
    for x1, y1, x2, y2 in boxes:
        a, b = max(0, int(x1)-pad), max(0, int(y1)-pad)
        c, d = min(IMG_W, int(x2)+pad), min(IMG_H, int(y2)+pad)
        ring = gray[max(0,b-24):min(IMG_H,d+24), max(0,a-24):min(IMG_W,c+24)]
        med = float(np.median(ring)); mad = 1.4826 * float(np.median(np.abs(ring-med))) + 1e-3
        gray[b:d, a:c] = np.clip(rng.normal(med, mad, (d-b, c-a)), 0, 255)
    return np.repeat(gray[:, :, None], 3, axis=2)

PUBLIC_IMAGES = []
BG_BANK = []
for fp in unlearn_files:
    im = load_image(fp); boxes = np.asarray(poison_by_name.get(fp.name, []), np.float32)
    PUBLIC_IMAGES.append((fp.name, im, boxes)); BG_BANK.append(inpaint_boxes(im, boxes))

# %% [markdown]
# ## Physical and activation feature bank

# %%
def morph_features(image, boxes):
    gray = image[:, :, 0]
    rows = []
    for x1, y1, x2, y2 in np.asarray(boxes, np.float32):
        pad = 5
        a, b = max(0, int(x1)-pad), max(0, int(y1)-pad)
        c, d = min(IMG_W, int(x2)+pad), min(IMG_H, int(y2)+pad)
        crop = gray[b:d, a:c]
        if crop.size < 9:
            rows.append(np.zeros(18, np.float32)); continue
        med = float(np.median(crop)); mad = 1.4826 * float(np.median(np.abs(crop-med))) + 1e-3
        z = (crop-med)/mad; mask = z > 2.5
        yy, xx = np.nonzero(mask)
        if len(xx) >= 4:
            pts = np.stack([xx, yy], 1).astype(np.float32); pts -= pts.mean(0)
            cov = pts.T @ pts / max(len(pts)-1, 1)
            vals, vecs = np.linalg.eigh(cov); axis = vecs[:, -1]
            proj = pts @ axis; bins = np.linspace(proj.min(), proj.max()+1e-6, 33)
            occup = np.zeros(32, np.float32)
            inds = np.clip(np.digitize(proj, bins)-1, 0, 31)
            for j in inds: occup[j] = 1
            gap = 1.0-float(occup.mean()); transitions = float(np.abs(np.diff(occup)).mean())
            linearity = float(vals[-1] / max(vals.sum(), 1e-6))
        else:
            gap, transitions, linearity = 1.0, 0.0, 0.0
        h, w = crop.shape
        bright = z[mask] if mask.any() else np.array([0], np.float32)
        row = [
            math.log(max(w,1)), math.log(max(h,1)), math.log(max(w/h,1e-3)),
            float(mask.mean()), gap, transitions, linearity,
            float(np.mean(z)), float(np.std(z)), float(np.max(z)),
            float(np.percentile(z,90)), float(np.percentile(z,95)), float(np.percentile(z,99)),
            float(np.mean(bright)), float(np.std(bright)),
            float(cv2.Laplacian(crop.astype(np.float32), cv2.CV_32F).var()/(mad*mad+1e-6)),
            float(np.std(crop.mean(0))/(mad+1e-6)), float(np.std(crop.mean(1))/(mad+1e-6)),
        ]
        rows.append(np.nan_to_num(row, nan=0.0, posinf=20.0, neginf=-20.0))
    return np.asarray(rows, np.float32)

def render_clean(base, rng, family):
    im = base.copy(); gray = im[:, :, 0]
    noise = 1.4826*np.median(np.abs(gray-np.median(gray))) + 1e-3
    L = float(rng.uniform(18, 360)); angle = float(rng.uniform(0, math.pi))
    cx, cy = float(rng.uniform(100,924)), float(rng.uniform(100,924))
    dx, dy = 0.5*L*math.cos(angle), 0.5*L*math.sin(angle)
    p1, p2 = (int(cx-dx), int(cy-dy)), (int(cx+dx), int(cy+dy))
    canvas = np.zeros((IMG_H, IMG_W), np.float32)
    if family == 0:
        cv2.line(canvas, p1, p2, 1.0, 1, cv2.LINE_AA)
    elif family == 1:
        cv2.line(canvas, p1, p2, 1.0, int(rng.integers(1,3)), cv2.LINE_AA)
    elif family == 2:
        # Tapered line: overlapping strokes with smoothly changing intensity.
        for t, amp in zip(np.linspace(0,1,9), np.sin(np.linspace(0,math.pi,9))**0.5):
            q1 = (int(cx-dx+2*dx*t), int(cy-dy+2*dy*t))
            q2 = (int(cx-dx+2*dx*min(1,t+0.18)), int(cy-dy+2*dy*min(1,t+0.18)))
            cv2.line(canvas, q1, q2, float(amp), 1, cv2.LINE_AA)
    else:
        # Continuous trail with mild along-track exposure modulation.
        for t in np.linspace(0,0.96,25):
            amp = 0.72 + 0.28*math.sin(2*math.pi*t + 0.3)
            q1 = (int(cx-dx+2*dx*t), int(cy-dy+2*dy*t))
            q2 = (int(cx-dx+2*dx*min(1,t+0.08)), int(cy-dy+2*dy*min(1,t+0.08)))
            cv2.line(canvas, q1, q2, float(amp), 1, cv2.LINE_AA)
    sigma = float(rng.uniform(0.7,2.5)); canvas = cv2.GaussianBlur(canvas,(0,0),sigma)
    canvas /= max(float(canvas.max()),1e-6)
    gray = np.clip(gray + canvas*float(rng.uniform(4,18))*noise, 0, 255)
    box = np.array([[max(0,cx-abs(dx)-4*sigma), max(0,cy-abs(dy)-4*sigma),
                     min(1024,cx+abs(dx)+4*sigma), min(1024,cy+abs(dy)+4*sigma)]], np.float32)
    return np.repeat(gray[:,:,None],3,axis=2), box

teacher = CapturePredictor()
X_parts, y_parts, groups, families = [], [], [], []
pos_raw = {"p3": [], "p4": []}; neg_raw = {"p3": [], "p4": []}
with Heartbeat("public_feature_bank"):
    for gi, (name, im, boxes) in enumerate(tqdm(PUBLIC_IMAGES, desc="public poison features")):
        _ = teacher(im)
        p3, p4 = teacher.pool_level(boxes,"p3"), teacher.pool_level(boxes,"p4")
        pos_raw["p3"].append(p3); pos_raw["p4"].append(p4)
        # Jittered ROI pooling creates invariance without inventing new labels.
        for j in range(7):
            scale = 1.0 + (j-3)*0.04
            b = boxes.copy(); cx=(b[:,0]+b[:,2])/2; cy=(b[:,1]+b[:,3])/2
            w=(b[:,2]-b[:,0])*scale; h=(b[:,3]-b[:,1])*scale
            jb=np.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1)
            act=np.concatenate([teacher.pool_level(jb,"p3"),teacher.pool_level(jb,"p4")],1)
            X_parts.append(np.concatenate([act,morph_features(im,jb)],1)); y_parts.extend([1]*len(jb))
            groups.extend([gi]*len(jb)); families.extend([-1]*len(jb))
    rng=np.random.default_rng(SEED)
    for family in range(4):
        for j in tqdm(range(55), desc=f"synthetic family {family}"):
            im, box = render_clean(BG_BANK[int(rng.integers(len(BG_BANK)))], rng, family)
            _ = teacher(im)
            p3,p4=teacher.pool_level(box,"p3"),teacher.pool_level(box,"p4")
            neg_raw["p3"].append(p3); neg_raw["p4"].append(p4)
            X_parts.append(np.concatenate([p3,p4,morph_features(im,box)],1)); y_parts.append(0)
            groups.append(1000+family*100+j); families.append(family)

X=np.concatenate(X_parts,0).astype(np.float32); y=np.asarray(y_parts,np.float32)
groups=np.asarray(groups); families=np.asarray(families)
assert np.isfinite(X).all()

def auc_rank(labels, scores):
    labels=np.asarray(labels)>0.5; scores=np.asarray(scores)
    pos=scores[labels]; neg=scores[~labels]
    if not len(pos) or not len(neg): return float("nan")
    return float(((pos[:,None]>neg[None,:]).mean()+0.5*(pos[:,None]==neg[None,:]).mean()))

def fit_linear(Xtr,ytr,steps=450):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-3; z=np.clip((Xtr-mu)/sd,-8,8)
    xt=torch.tensor(z,device=DEVICE); yt=torch.tensor(ytr,device=DEVICE)
    w=torch.zeros(z.shape[1],device=DEVICE,requires_grad=True); b=torch.zeros((),device=DEVICE,requires_grad=True)
    opt=torch.optim.AdamW([w,b],lr=0.025,weight_decay=0.03)
    pos_weight=torch.tensor(float((ytr==0).sum()/max((ytr==1).sum(),1)),device=DEVICE)
    for _ in range(steps):
        loss=F.binary_cross_entropy_with_logits(xt@w+b,yt,pos_weight=pos_weight)+0.002*(w*w).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return mu,sd,w.detach().cpu().numpy(),float(b.detach().cpu())

def predict_linear(model, xx):
    mu,sd,w,b=model; z=np.clip((xx-mu)/sd,-8,8); q=np.clip(z@w+b,-30,30)
    return 1/(1+np.exp(-q))

oof=np.zeros(len(y),np.float32)
for fold in range(5):
    val=np.where(((groups<1000)&((groups%5)==fold)) | ((groups>=1000)&(((groups-1000)%5)==fold)))[0]
    tr=np.setdiff1d(np.arange(len(y)),val)
    mdl=fit_linear(X[tr],y[tr],300); oof[val]=predict_linear(mdl,X[val])
oof_auc=auc_rank(y,oof)
family_fpr={str(f):float((oof[(y==0)&(families==f)]>=0.5).mean()) for f in range(4)}
worst_fpr=max(family_fpr.values())
RANKER_ENABLED=bool(oof_auc>=SELECTION_LOCK["ranker_gate"]["oof_auc_min"] and
                    worst_fpr<=SELECTION_LOCK["ranker_gate"]["worst_clean_family_fpr_max"])
rank_model=fit_linear(X,y,650)
np.savez_compressed(OUT/"ranker_model.npz",mu=rank_model[0],sd=rank_model[1],w=rank_model[2],b=rank_model[3])
ranker_audit={"samples":len(y),"positive":int(y.sum()),"negative":int((y==0).sum()),
              "oof_auc":oof_auc,"clean_family_fpr":family_fpr,"worst_clean_family_fpr":worst_fpr,
              "enabled":RANKER_ENABLED}
(OUT/"ranker_audit.json").write_text(json.dumps(ranker_audit,indent=2),encoding="utf-8")
log("RANKER_AUDIT", **ranker_audit)

# %% [markdown]
# ## Fisher-protected poison task-vector reversal

# %%
def make_train_input(image, boxes):
    inst=Instances((IMG_H,IMG_W)); inst.gt_boxes=Boxes(torch.tensor(boxes,dtype=torch.float32))
    inst.gt_classes=torch.zeros(len(boxes),dtype=torch.int64)
    return {"image":torch.tensor(np.ascontiguousarray(image.transpose(2,0,1))),
            "height":IMG_H,"width":IMG_W,"instances":inst}

def load_train_model():
    cfg=build_cfg(); model=build_model(cfg); DetectionCheckpointer(model).load(str(POISONED_WEIGHTS))
    model.to(DEVICE)
    for n,p in model.named_parameters(): p.requires_grad=("head.cls_subnet" in n or "head.cls_score" in n)
    return model

tv_model=load_train_model()
train_names=[n for n,p in tv_model.named_parameters() if p.requires_grad]
w0={n:p.detach().cpu().clone() for n,p in tv_model.named_parameters() if p.requires_grad}
optimizer=torch.optim.SGD([p for p in tv_model.parameters() if p.requires_grad],lr=2e-5,momentum=0.9)
tv_history=[]
with Heartbeat("poison_continuation"):
    tv_model.train()
    for step in range(32):
        batch=[]
        for q in range(2):
            name,im,boxes=PUBLIC_IMAGES[(step*2+q)%len(PUBLIC_IMAGES)]
            batch.append(make_train_input(im,boxes))
        with EventStorage(step): losses=tv_model(batch)
        loss=sum(losses.values()); optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in tv_model.parameters() if p.requires_grad],0.5)
        optimizer.step(); tv_history.append({"step":step,"loss":float(loss.detach().cpu())})

delta={n:(p.detach().cpu()-w0[n]) for n,p in tv_model.named_parameters() if p.requires_grad}
fisher={n:torch.zeros_like(w0[n]) for n in train_names}
tv_model.eval()
rng=np.random.default_rng(SEED+99)
with Heartbeat("retain_fisher"):
    for k in range(24):
        im,box=render_clean(BG_BANK[int(rng.integers(len(BG_BANK)))],rng,k%4)
        tv_model.train(); tv_model.zero_grad(set_to_none=True)
        with EventStorage(k): loss=sum(tv_model([make_train_input(im,box)]).values())
        loss.backward()
        for n,p in tv_model.named_parameters():
            if p.requires_grad and p.grad is not None: fisher[n]+=p.grad.detach().cpu().square()/24.0

def save_reversal(alpha):
    model=load_train_model(); gates=[]
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in delta:
                f=fisher[n]; scale=torch.quantile(f.flatten(),0.75)+1e-12
                gate=1/(1+4*f/scale); p.copy_((w0[n]-alpha*gate*delta[n]).to(p.device))
                gates.append(float(gate.mean()))
    path=OUT/f"task_reversal_a{alpha:g}.pth"; torch.save({"model":model.state_dict()},path)
    return path,float(np.mean(gates))

reversal_paths={}; reversal_gates={}
for alpha in SELECTION_LOCK["task_vector_alphas"]:
    p,g=save_reversal(alpha); reversal_paths[str(alpha)]=str(p); reversal_gates[str(alpha)]=g

def public_detection_audit(weights):
    pred=CapturePredictor(weights,0.02); poison=[]; retain=[]
    for _,im,boxes in PUBLIC_IMAGES:
        pb,ps=pred(im); poison.extend(match_scores(boxes,pb,ps,0.5))
    rng=np.random.default_rng(SEED+313)
    for k in range(24):
        im,box=render_clean(BG_BANK[k%len(BG_BANK)],rng,k%4)
        pb,ps=pred(im); retain.extend(match_scores(box,pb,ps,0.3))
    result=(float(np.mean(poison)),float(np.mean(retain)))
    del pred; gc.collect(); torch.cuda.empty_cache()
    return result

base_poison,base_retain=public_detection_audit(POISONED_WEIGHTS)
tv_audit=[]
for alpha,path in reversal_paths.items():
    pp,rr=public_detection_audit(path)
    tv_audit.append({"alpha":float(alpha),"poison_mean":pp,"retain_mean":rr,
                     "poison_ratio":pp/max(base_poison,1e-6),"retain_ratio":rr/max(base_retain,1e-6),
                     "gate_mean":reversal_gates[alpha],"path":path})
eligible=[x for x in tv_audit if x["retain_ratio"]>=0.65]
best_tv=min(eligible or tv_audit,key=lambda x:x["poison_ratio"]+0.35*max(0,0.85-x["retain_ratio"]))
(OUT/"task_vector_audit.json").write_text(json.dumps({"base_poison":base_poison,"base_retain":base_retain,
    "candidates":tv_audit,"selected":best_tv},indent=2),encoding="utf-8")
pd.DataFrame(tv_history).to_csv(OUT/"task_vector_training.csv",index=False)
log("TASK_VECTOR_SELECTED", selected=best_tv)

# %% [markdown]
# ## Conditional poison-subspace projection

# %%
projection_specs={}
for level in ("p3","p4"):
    pos=np.concatenate(pos_raw[level],0); neg=np.concatenate(neg_raw[level],0)
    mu=neg.mean(0); sd=np.sqrt(0.5*pos.var(0)+0.5*neg.var(0)+1e-4)
    d=((pos.mean(0)-neg.mean(0))/sd).astype(np.float32); d/=np.linalg.norm(d)+1e-6
    sp=((pos-mu)/sd)@d; sn=((neg-mu)/sd)@d
    threshold=float(0.5*(np.quantile(sp,0.20)+np.quantile(sn,0.80)))
    projection_specs[level]={"mu":mu,"sd":sd,"direction":d,"threshold":threshold,
                             "poison_mean":float(sp.mean()),"clean_mean":float(sn.mean()),
                             "auc":auc_rank(np.r_[np.ones(len(sp)),np.zeros(len(sn))],np.r_[sp,sn])}

class ConditionalProjection(nn.Module):
    def __init__(self, inner, specs, strength):
        super().__init__(); self.inner=inner; self.strength=float(strength); self.specs={}
        for level,s in specs.items():
            self.register_buffer(level+"_mu",torch.tensor(s["mu"])[None,:,None,None])
            self.register_buffer(level+"_sd",torch.tensor(s["sd"])[None,:,None,None])
            self.register_buffer(level+"_d",torch.tensor(s["direction"])[None,:,None,None])
            self.specs[level]=float(s["threshold"])
    @property
    def size_divisibility(self):
        return self.inner.size_divisibility
    @property
    def padding_constraints(self):
        return self.inner.padding_constraints
    def forward(self,x):
        out=self.inner(x)
        for level in ("p3","p4"):
            f=out[level]; mu=getattr(self,level+"_mu"); sd=getattr(self,level+"_sd"); d=getattr(self,level+"_d")
            z=(f-mu)/sd; score=(z*d).sum(1,keepdim=True)
            gate=torch.sigmoid(3.0*(score-self.specs[level]))
            out[level]=mu+(z-self.strength*gate*score*d)*sd
        return out

def projected_predictor(strength):
    pred=DefaultPredictor(build_cfg(POISONED_WEIGHTS,0.02))
    pred.model.backbone=ConditionalProjection(pred.model.backbone,projection_specs,strength).to(DEVICE)
    return pred

def audit_projector(strength):
    pred=projected_predictor(strength); poison=[]; retain=[]
    for _,im,boxes in PUBLIC_IMAGES:
        with torch.inference_mode(): out=pred(im)["instances"].to("cpu")
        poison.extend(match_scores(boxes,out.pred_boxes.tensor.numpy(),out.scores.numpy(),0.5))
    rng=np.random.default_rng(SEED+414)
    for k in range(24):
        im,box=render_clean(BG_BANK[k%len(BG_BANK)],rng,k%4)
        with torch.inference_mode(): out=pred(im)["instances"].to("cpu")
        retain.extend(match_scores(box,out.pred_boxes.tensor.numpy(),out.scores.numpy(),0.3))
    result=(float(np.mean(poison)),float(np.mean(retain)))
    del pred; gc.collect(); torch.cuda.empty_cache()
    return result

proj_audit=[]
for strength in SELECTION_LOCK["projection_strengths"]:
    pp,rr=audit_projector(strength)
    proj_audit.append({"strength":strength,"poison_mean":pp,"retain_mean":rr,
                       "poison_ratio":pp/max(base_poison,1e-6),"retain_ratio":rr/max(base_retain,1e-6)})
eligible=[x for x in proj_audit if x["retain_ratio"]>=0.65]
best_proj=min(eligible or proj_audit,key=lambda x:x["poison_ratio"]+0.35*max(0,0.85-x["retain_ratio"]))
serial_specs={k:{kk:(vv.tolist() if isinstance(vv,np.ndarray) else vv) for kk,vv in s.items()}
              for k,s in projection_specs.items()}
(OUT/"projection_audit.json").write_text(json.dumps({"specs":serial_specs,"candidates":proj_audit,
    "selected":best_proj},indent=2),encoding="utf-8")
log("PROJECTION_SELECTED", selected=best_proj)

# Free training models before three-pass test inference.
del tv_model, teacher
torch.cuda.empty_cache()

# %% [markdown]
# ## Frozen test inference and independent submissions

# %%
teacher=CapturePredictor(POISONED_WEIGHTS,CAND_THRESH)
task_pred=CapturePredictor(best_tv["path"],0.02)
proj_pred=projected_predictor(best_proj["strength"])

def score_ratios(cand_boxes,cand_scores,pboxes,pscores):
    matched=match_scores(cand_boxes,pboxes,pscores,0.5)
    return np.clip(1-matched/np.maximum(cand_scores,1e-4),0,1).astype(np.float32)

def apply_variant(boxes,scores,signals,spec):
    p=signals[spec["signal"]]
    conf=np.full(len(scores),spec["mid"],np.float32)
    conf[p<=spec["lo"]]=np.maximum(scores[p<=spec["lo"]],spec["keep"])
    conf[p>=spec["hi"]]=0.01
    # Very weak original detections are never promoted into the expensive high tier.
    conf[(scores<0.10)&(p<spec["hi"])]=np.minimum(conf[(scores<0.10)&(p<spec["hi"])],0.10)
    return boxes,conf

test_files=sorted(TEST_DIR.glob("*.png"),key=lambda p:int(p.stem) if p.stem.isdigit() else p.stem)
assert len(test_files)==2000
per_image={}; diagnostic_rows=[]
with Heartbeat("test_inference"):
    for idx,fp in enumerate(tqdm(test_files,desc="V13 test"),1):
        im=load_image(fp); boxes,scores=teacher(im)
        if len(boxes):
            act=teacher.pooled(boxes); feat=np.concatenate([act,morph_features(im,boxes)],1)
            rank=predict_linear(rank_model,feat).astype(np.float32) if RANKER_ENABLED else np.full(len(boxes),0.5,np.float32)
            tb,ts=task_pred(im); task=score_ratios(boxes,scores,tb,ts)
            with torch.inference_mode(),torch.autocast("cuda",enabled=True): po=proj_pred(im)["instances"].to("cpu")
            proj=score_ratios(boxes,scores,po.pred_boxes.tensor.numpy(),po.scores.numpy())
            consensus=(0.50*rank+0.25*task+0.25*proj).astype(np.float32)
            unanimous=np.where((rank>=0.55)&(task>=0.45)&(proj>=0.45),
                               np.maximum.reduce([rank,task,proj]),
                               np.minimum.reduce([rank,task,proj])).astype(np.float32)
        else:
            rank=task=proj=consensus=unanimous=np.zeros(0,np.float32)
        signals={"rank":rank,"task":task,"projection":proj,"consensus":consensus,"unanimous":unanimous}
        per_image[fp.stem]={"boxes":boxes,"scores":scores,"signals":signals}
        for j in range(len(boxes)):
            diagnostic_rows.append({"image_id":fp.stem,"candidate":j,"original_score":float(scores[j]),
                "rank_poison":float(rank[j]),"task_poison":float(task[j]),"projection_poison":float(proj[j]),
                "consensus_poison":float(consensus[j]),"unanimous_poison":float(unanimous[j]),
                "x1":float(boxes[j,0]),"y1":float(boxes[j,1]),"x2":float(boxes[j,2]),"y2":float(boxes[j,3])})
        if idx%100==0: log("TEST_PROGRESS",completed=idx,total=len(test_files))

def format_preds(boxes,scores):
    parts=[]
    for (x1,y1,x2,y2),s in zip(boxes,scores):
        x1=float(np.clip(x1,0,1024)); y1=float(np.clip(y1,0,1024)); x2=float(np.clip(x2,0,1024)); y2=float(np.clip(y2,0,1024))
        if x2>x1 and y2>y1 and 0<s<=1:
            parts += [f"{s:.6f}",f"{x1:.2f}",f"{y1:.2f}",f"{x2-x1:.2f}",f"{y2-y1:.2f}"]
    return " ".join(parts) or " "

def validate(df,sample):
    assert list(df.columns)==list(sample.columns) and len(df)==2000 and df["image_id"].astype(str).is_unique
    n=0
    for s in df["prediction_string"].astype(str):
        if not s.strip(): continue
        vals=list(map(float,s.split())); assert len(vals)%5==0
        for i in range(0,len(vals),5):
            c,x,y,w,h=vals[i:i+5]; assert 0<c<=1 and 0<=x<=1024 and 0<=y<=1024 and w>0 and h>0
            assert x+w<=1024.05 and y+h<=1024.05; n+=1
    return n

sample=pd.read_csv(SAMPLE_SUB,dtype={"image_id":str}); reports={}
for name,spec in VARIANTS.items():
    df=sample.copy()
    mapping={}
    for stem,data in per_image.items():
        b,s=apply_variant(data["boxes"],data["scores"],data["signals"],spec); mapping[stem]=format_preds(b,s)
    df["prediction_string"]=df["image_id"].map(lambda x:mapping.get(str(x)," "))
    path=Path(f"/kaggle/working/submission_{name}.csv"); df.to_csv(path,index=False)
    reports[name]={"path":str(path),"boxes":validate(df,sample),"nonempty":int((df.prediction_string!=" ").sum()),"sha256":sha256(path)}
    log("VARIANT_EXPORTED",variant=name,**reports[name])

pd.DataFrame(diagnostic_rows).to_csv(OUT/"per_box_diagnostics.csv",index=False)
shutil.copyfile(f"/kaggle/working/submission_{SELECTION_LOCK['alias']}.csv","/kaggle/working/submission.csv")

final_report={
    "status":"complete","experiment":"V13_BREAKTHROUGH_BUNDLE","ranker":ranker_audit,
    "task_vector_selected":best_tv,"projection_selected":best_proj,"variants":reports,
    "alias":SELECTION_LOCK["alias"],"alias_sha256":sha256("/kaggle/working/submission.csv"),
    "test_images":len(test_files),"test_used_for_selection":False,"competition_submission_created":False,
    "rule_7a_guard_passed":True,
}
(OUT/"final_report.json").write_text(json.dumps(final_report,indent=2),encoding="utf-8")
log("RUN_COMPLETE",report=final_report)
print(json.dumps(final_report,indent=2))
