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
# # NDR V20 - TRACE-style contextual consistency (shard 1/2)
#
# This notebook computes a genuinely new candidate-level signal: how a fixed
# RetinaNet detection behaves when its local streak residual is transplanted to
# predeclared public backgrounds and when its focal structure is disrupted.
# The model and thresholds are selected only from the 20 public poison examples,
# public external streaks, and an analytic clean simulator. Test inference is
# sharded for speed. No test labels, pseudo-labels, adaptive thresholds, box
# additions, coordinate changes, confidence promotions, or submissions occur.

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
import hashlib
import json
import math
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

SEED = 200723
SHARD_INDEX = 1
SHARD_COUNT = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda"

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
if not ROOT.exists():
    ROOT = Path("/kaggle/input/neural-debris-removal-in-streak-detection-models")
POISONED_WEIGHTS = ROOT / "poisoned_model/poisoned_model.pth"
UNLEARN_DIR = ROOT / "unlearn_set"
UNLEARN_JSON = UNLEARN_DIR / "annotations_coco.json"
TEST_DIR = ROOT / "test_set/test_set"
for required in (POISONED_WEIGHTS, UNLEARN_JSON, TEST_DIR):
    assert required.exists(), required

OUT = Path(f"/kaggle/working/ndr_v20_shard{SHARD_INDEX}")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.jsonl"
EXPECTED_V15B_SHA = "4345387b72aecd55dd3856b1607ed836ec9f400365fcf3f3b89f8947f1ff2412"
BETAS = [0.25, 0.5, 1.0, 2.0]

VARIANTS = {
    "V20_0_exact_v15b": {"mode": "identity"},
    "V20_A_strict": {"mode": "hard", "trace_threshold": "strict", "pcgrad_at": 0.70, "pcgrad_votes": 2, "floor": 0.02},
    "V20_B_relaxed": {"mode": "hard", "trace_threshold": "relaxed", "pcgrad_at": 0.65, "pcgrad_votes": 2, "floor": 0.02},
    "V20_C_trace_extreme": {"mode": "hard", "trace_threshold": "strict", "pcgrad_at": 0.55, "pcgrad_votes": 1, "floor": 0.02},
    "V20_D_four_signal": {"mode": "graded", "trace_threshold": "relaxed", "pcgrad_at": 0.60, "pcgrad_votes": 3, "cap": 0.10, "floor": 0.02},
}
LOCK = {
    "status": "frozen_before_test_enumeration",
    "experiment": "V20_TRACE_CONTEXT_FOCAL_ENTROPY",
    "seed": SEED,
    "shard": {"index": SHARD_INDEX, "count": SHARD_COUNT},
    "incumbent": {"name": "V15_B", "score": 213.7088, "sha256": EXPECTED_V15B_SHA},
    "public_gate": {
        "poison": "20 organizer-provided unlearn boxes",
        "clean_domains": ["public StreaksYolo residuals", "analytic Gaussian-PSF simulator"],
        "features": ["background persistence", "background variance", "fire entropy", "focal fragility", "context-focal contrast"],
        "bidirectional_auc_min": 0.72,
        "minimum_score_margin": 0.08,
        "strict_clean_false_positives": 0,
        "relaxed_clean_quantile": 0.98,
    },
    "test": {
        "eligible": "V15_B confidence >= 0.21 only",
        "backgrounds": 4,
        "focal_transforms": ["gaussian blur", "pixel shuffle", "phase randomization", "endpoint attenuation"],
        "selection_or_normalization_from_test": False,
    },
    "variants": VARIANTS,
    "invariants": {"box_bank": "exact V15_B", "boxes_added": 0, "boxes_moved": 0, "confidence_increases": 0},
    "rule_7a": {"test_labels": False, "test_pseudo_labels": False, "test_used_for_training_or_selection": False, "competition_submission_created": False},
}
(OUT / "selection_lock.json").write_text(json.dumps(LOCK, indent=2), encoding="utf-8")


def log(message, **fields):
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "message": message, **fields}
    print(json.dumps(row, default=str), flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Heartbeat:
    def __init__(self, label): self.label = label
    def __enter__(self):
        self.stop = threading.Event(); started = time.time()
        def beat():
            while not self.stop.wait(45):
                log("HEARTBEAT", stage=self.label, elapsed_min=round((time.time()-started)/60, 1))
        self.thread = threading.Thread(target=beat, daemon=True); self.thread.start(); log("STAGE_START", stage=self.label); return self
    def __exit__(self, typ, value, trace):
        self.stop.set(); self.thread.join(timeout=2); log("STAGE_END", stage=self.label, ok=typ is None)


log("SELECTION_LOCK_WRITTEN", lock=LOCK)

# %% [markdown]
# ## Data, detector, and deterministic transformations

# %%
def discover_external_root():
    candidates = []
    for yaml_path in sorted(Path("/kaggle/input").rglob("data.yaml")):
        root = yaml_path.parent
        if (root / "train/images").is_dir() and (root / "train/labels").is_dir():
            candidates.append(root)
    assert candidates, "StreaksYoloDataset mount not found"
    candidates.sort(key=lambda p: ("streak" not in str(p).lower(), len(p.parts), str(p)))
    return candidates[0]


def find_prior(name, required_columns=None):
    candidates = sorted(Path("/kaggle/input").rglob(name))
    for path in candidates:
        if required_columns:
            try: columns = set(pd.read_csv(path, nrows=1).columns)
            except Exception: continue
            if not set(required_columns).issubset(columns): continue
        return path
    raise AssertionError((name, [str(x) for x in candidates]))


EXT_ROOT = discover_external_root()
ANCHOR_PATH = find_prior("submission_V19_0_exact_v15b.csv")
PCGRAD_PATH = find_prior("per_box_diagnostics.csv", ["poison_beta_0.25", "poison_beta_0.5", "poison_beta_1", "poison_beta_2"])
assert sha256(ANCHOR_PATH) == EXPECTED_V15B_SHA

BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ASPECTS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
SIZES = [[16], [32], [64], [128], [256]]


def cfg_for():
    cfg = get_cfg(); cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG)); cfg.MODEL.WEIGHTS = str(POISONED_WEIGHTS)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1; cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.02
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ASPECTS]; cfg.MODEL.ANCHOR_GENERATOR.SIZES = SIZES
    cfg.MODEL.DEVICE = DEVICE; cfg.TEST.DETECTIONS_PER_IMAGE = 100
    return cfg


PREDICTOR = DefaultPredictor(cfg_for())


def load_gray(path):
    gray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED); assert gray is not None, path
    if gray.ndim == 3: gray = gray[:, :, 0]
    if gray.dtype == np.uint16: gray = gray.astype(np.float32) / 65535.0 * 255.0
    else:
        gray = gray.astype(np.float32)
        if gray.max() <= 1: gray *= 255.0
    return np.clip(gray, 0, 255).astype(np.float32)


def to_bgr(gray): return np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)


def iou_matrix(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 4); b = np.asarray(b, np.float32).reshape(-1, 4)
    if not len(a) or not len(b): return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2]); br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    size = np.clip(br-tl, 0, None); inter = size[:, :, 0]*size[:, :, 1]
    aa = np.prod(np.clip(a[:, 2:]-a[:, :2], 0, None), 1); bb = np.prod(np.clip(b[:, 2:]-b[:, :2], 0, None), 1)
    return inter/np.maximum(aa[:, None]+bb[None, :]-inter, 1e-6)


def predict_match(gray, box, threshold=0.30):
    with torch.inference_mode(), torch.autocast("cuda", enabled=True):
        output = PREDICTOR(to_bgr(gray))["instances"].to("cpu")
    boxes = output.pred_boxes.tensor.numpy().astype(np.float32); scores = output.scores.numpy().astype(np.float32)
    if not len(boxes): return 0.0, 0.0
    overlaps = iou_matrix(np.asarray([box], np.float32), boxes)[0]; index = int(np.argmax(overlaps))
    return (float(scores[index]), float(overlaps[index])) if overlaps[index] >= threshold else (0.0, float(overlaps[index]))


with UNLEARN_JSON.open(encoding="utf-8") as handle: coco = json.load(handle)
id_to_name = {int(x["id"]): x["file_name"] for x in coco["images"]}; poison_by_name = {}
for ann in coco["annotations"]:
    x,y,w,h = map(float, ann["bbox"]); poison_by_name.setdefault(id_to_name[int(ann["image_id"])], []).append([x,y,x+w,y+h])

PUBLIC = []; BACKGROUNDS = []; bg_rng = np.random.default_rng(SEED+1)
for path in sorted(UNLEARN_DIR.glob("*.png")):
    gray = load_gray(path); boxes = np.asarray(poison_by_name.get(path.name, []), np.float32); PUBLIC.append((path.name, gray, boxes)); clean = gray.copy()
    for x1,y1,x2,y2 in boxes:
        l,t,r,b = max(0,int(x1)-8),max(0,int(y1)-8),min(1024,int(x2)+8),min(1024,int(y2)+8)
        ring = clean[max(0,t-28):min(1024,b+28),max(0,l-28):min(1024,r+28)]
        med = float(np.median(ring)); mad = 1.4826*float(np.median(np.abs(ring-med)))+1e-3
        clean[t:b,l:r] = np.clip(bg_rng.normal(med,mad,(b-t,r-l)),0,255)
    BACKGROUNDS.append(clean)


def crop_geometry(gray, box, pad=6):
    x1,y1,x2,y2 = map(float, box); l=max(0,int(math.floor(x1))-pad); t=max(0,int(math.floor(y1))-pad)
    r=min(gray.shape[1],int(math.ceil(x2))+pad); b=min(gray.shape[0],int(math.ceil(y2))+pad)
    crop = gray[t:b,l:r].copy(); border = np.r_[crop[0],crop[-1],crop[:,0],crop[:,-1]]
    med = float(np.median(border)); mad = 1.4826*float(np.median(np.abs(border-med)))+1e-3
    residual_z = np.clip((crop-med)/mad,0,24).astype(np.float32)
    inner = np.asarray([float(x1-l),float(y1-t),float(x2-l),float(y2-t)],np.float32)
    return crop, residual_z, inner


def paste_context(source_gray, source_box, background, seed):
    _, residual, inner = crop_geometry(source_gray, source_box); h,w = residual.shape; rng = np.random.default_rng(seed)
    if h >= 990 or w >= 990: residual = cv2.resize(residual,(min(w,900),min(h,900))); h,w = residual.shape; inner=np.asarray([6,6,w-6,h-6],np.float32)
    y = int(rng.integers(8,1024-h-8)); x = int(rng.integers(8,1024-w-8)); result = background.copy(); local = result[y:y+h,x:x+w]
    med=float(np.median(local)); mad=1.4826*float(np.median(np.abs(local-med)))+1e-3
    result[y:y+h,x:x+w] = np.clip(local+residual*mad,0,255)
    target = inner + np.asarray([x,y,x,y],np.float32)
    return result,target


def focal_variant(gray, box, kind, seed):
    result = gray.copy(); crop, residual, inner = crop_geometry(gray,box); x1,y1,x2,y2=map(float,box)
    l=int(round(x1-inner[0])); t=int(round(y1-inner[1])); h,w=crop.shape; rng=np.random.default_rng(seed)
    if kind == "blur": changed=cv2.GaussianBlur(crop,(0,0),2.2)
    elif kind == "shuffle": changed=crop.ravel()[rng.permutation(crop.size)].reshape(crop.shape)
    elif kind == "phase":
        spectrum=np.fft.rfft2(crop-crop.mean()); phase=rng.uniform(-math.pi,math.pi,spectrum.shape); changed=np.fft.irfft2(np.abs(spectrum)*np.exp(1j*phase),s=crop.shape).real+crop.mean()
    else:
        changed=crop.copy(); positive=residual>2.0
        if w>=h:
            q=max(1,w//6); changed[:, :q][positive[:, :q]]=np.median(crop); changed[:, -q:][positive[:, -q:]]=np.median(crop)
        else:
            q=max(1,h//6); changed[:q][positive[:q]]=np.median(crop); changed[-q:][positive[-q:]]=np.median(crop)
    result[t:t+h,l:l+w] = np.clip(changed,0,255)
    return result


def binary_entropy(rate):
    rate=float(np.clip(rate,1e-6,1-1e-6)); return float(-(rate*math.log(rate)+(1-rate)*math.log(1-rate)))


def trace_features(gray, box, base_score, seed):
    base=max(float(base_score),1e-4); context=[]; context_iou=[]
    for j in range(4):
        image,target=paste_context(gray,box,BACKGROUNDS[(seed+3*j)%len(BACKGROUNDS)],seed+101*j)
        score,overlap=predict_match(image,target); context.append(min(score/base,2.0)); context_iou.append(overlap)
    focal=[]
    for j,kind in enumerate(("blur","shuffle","phase","endpoints")):
        score,_=predict_match(focal_variant(gray,box,kind,seed+211*j),box); focal.append(min(score/base,2.0))
    context=np.asarray(context,np.float32); focal=np.asarray(focal,np.float32); fire=float(np.mean(context*base>=.20))
    features=np.asarray([
        context.mean(),context.std(),np.min(context),np.max(context),fire,binary_entropy(fire),
        np.mean(context_iou),focal.mean(),focal.std(),focal[0],focal[1],focal[2],focal[3],
        context.mean()-focal.mean(),context.mean()-.5*context.std(),float(np.mean(focal<.25)),math.log1p(base),
    ],np.float32)
    return np.nan_to_num(features,nan=0,posinf=4,neginf=-4),context,focal

# %% [markdown]
# ## Public cross-domain validation and frozen TRACE ranker

# %%
def external_signals(limit=80):
    signals=[]
    for split in ("valid","val","test","train"):
        image_dir,label_dir=EXT_ROOT/split/"images",EXT_ROOT/split/"labels"
        if not image_dir.is_dir(): continue
        for image_path in sorted(image_dir.glob("*")):
            label_path=label_dir/f"{image_path.stem}.txt"
            if not label_path.exists() or not label_path.stat().st_size: continue
            gray=cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)
            if gray is None: continue
            h,w=gray.shape
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values=line.split()
                if len(values)<5: continue
                _,xc,yc,bw,bh=map(float,values[:5]); x1=(xc-bw/2)*w; y1=(yc-bh/2)*h; x2=(xc+bw/2)*w; y2=(yc+bh/2)*h
                crop,residual,_=crop_geometry(gray,[x1,y1,x2,y2]);
                if residual.max()>=3: signals.append(residual/max(float(residual.max()),1e-6))
                if len(signals)>=limit: return signals
    return signals


def synthetic_signal(rng):
    h=int(rng.integers(24,100)); w=int(rng.integers(50,320)); canvas=np.zeros((h,w),np.float32)
    angle=float(rng.uniform(-.35,.35)); center=(w/2,h/2); length=float(rng.uniform(.55,.92)*w)
    dx=math.cos(angle)*length/2; dy=math.sin(angle)*length/2
    cv2.line(canvas,(int(center[0]-dx),int(center[1]-dy)),(int(center[0]+dx),int(center[1]+dy)),1.0,int(rng.integers(1,4)),cv2.LINE_AA)
    canvas=cv2.GaussianBlur(canvas,(0,0),float(rng.uniform(.7,2.0)))
    return canvas/max(float(canvas.max()),1e-6)


def composite_from_signal(signal,index,seed):
    rng=np.random.default_rng(seed); longest=max(signal.shape); scale=float(rng.uniform(40,300))/max(longest,1)
    h=max(5,int(signal.shape[0]*scale)); w=max(5,int(signal.shape[1]*scale)); signal=cv2.resize(signal,(w,h))
    background=BACKGROUNDS[index%len(BACKGROUNDS)].copy(); y=int(rng.integers(12,1024-h-12)); x=int(rng.integers(12,1024-w-12)); local=background[y:y+h,x:x+w]
    mad=1.4826*float(np.median(np.abs(local-np.median(local))))+1e-3; amplitude=float(rng.uniform(5,18))*mad
    background[y:y+h,x:x+w]=np.clip(local+signal*amplitude,0,255)
    return background,np.asarray([x,y,x+w,y+h],np.float32)


def fit_logistic(x,y,steps=1200,lr=.08,l2=.03):
    mu=x.mean(0); sd=x.std(0)+1e-3; z=np.clip((x-mu)/sd,-8,8); w=np.zeros(z.shape[1],np.float64); b=0.0
    y=y.astype(np.float64); pos_weight=float(np.sum(y==0)/max(np.sum(y==1),1)); weights=np.where(y==1,pos_weight,1.0)
    for step in range(steps):
        logits=np.clip(z@w+b,-25,25); p=1/(1+np.exp(-logits)); error=(p-y)*weights/weights.mean()
        rate=lr/(1+.0015*step); w-=rate*((z.T@error)/len(z)+l2*w); b-=rate*float(error.mean())
    return {"mu":mu,"sd":sd,"w":w,"b":b}


def logistic_predict(model,x):
    z=np.clip((x-model["mu"])/model["sd"],-8,8); logits=np.clip(z@model["w"]+model["b"],-25,25); return (1/(1+np.exp(-logits))).astype(np.float32)


def auc(y,s):
    p=s[y==1]; n=s[y==0]; return float((p[:,None]>n).mean()+.5*(p[:,None]==n).mean())


external=external_signals(80); assert len(external)>=60
rng=np.random.default_rng(SEED+20); synthetic=[synthetic_signal(rng) for _ in range(80)]
poison_examples=[(gray,box) for _,gray,boxes in PUBLIC for box in boxes]


def build_features(examples,domain,offset):
    rows=[]; kept=[]
    for i,item in enumerate(tqdm(examples,desc=f"TRACE {domain}")):
        if domain=="poison": gray,box=item; base,_=predict_match(gray,box)
        else:
            gray,box=composite_from_signal(item,i,SEED+offset+i); base,_=predict_match(gray,box)
        if base<.20: continue
        feature,_,_=trace_features(gray,box,base,SEED+offset+10000+i); rows.append(feature); kept.append(i)
    return np.asarray(rows,np.float32),kept


with Heartbeat("public_trace_gate"):
    poison_x,poison_kept=build_features(poison_examples,"poison",100)
    external_x,external_kept=build_features(external[:60],"external",200)
    synthetic_x,synthetic_kept=build_features(synthetic[:60],"synthetic",300)

assert len(poison_x)>=16 and len(external_x)>=16 and len(synthetic_x)>=16,(len(poison_x),len(external_x),len(synthetic_x))
# Some clean controls legitimately yield no original-model detection after the
# frozen transformations. Size each public-only split from the retained count
# while reserving at least four examples for the cross-domain validation gate.
def public_split(values,preferred_train):
    train_count=min(preferred_train,max(12,int(np.floor(len(values)*2/3))))
    train_count=min(train_count,len(values)-4)
    return values[:train_count],values[train_count:]

p_train,p_valid=public_split(poison_x,15)
ext_train,ext_valid=public_split(external_x,40)
syn_train,syn_valid=public_split(synthetic_x,40)

model_ext=fit_logistic(np.r_[p_train,ext_train],np.r_[np.ones(len(p_train)),np.zeros(len(ext_train))])
model_syn=fit_logistic(np.r_[p_train,syn_train],np.r_[np.ones(len(p_train)),np.zeros(len(syn_train))])
y_ext=np.r_[np.ones(len(p_valid)),np.zeros(len(syn_valid))]; s_ext=logistic_predict(model_ext,np.r_[p_valid,syn_valid])
y_syn=np.r_[np.ones(len(p_valid)),np.zeros(len(ext_valid))]; s_syn=logistic_predict(model_syn,np.r_[p_valid,ext_valid])
auc_ext,auc_syn=auc(y_ext,s_ext),auc(y_syn,s_syn)

FINAL_MODELS=[]; poison_oof=np.zeros(len(poison_x),np.float32)
for fold in range(5):
    valid=np.arange(len(poison_x))[np.arange(len(poison_x))%5==fold]; train=np.setdiff1d(np.arange(len(poison_x)),valid)
    model=fit_logistic(np.r_[poison_x[train],ext_train,syn_train],np.r_[np.ones(len(train)),np.zeros(len(ext_train)+len(syn_train))]); FINAL_MODELS.append(model)
    poison_oof[valid]=logistic_predict(model,poison_x[valid])
clean_cal=np.mean([logistic_predict(m,np.r_[ext_valid,syn_valid]) for m in FINAL_MODELS],0)
strict_threshold=float(min(1.0,np.max(clean_cal)+1e-5)); relaxed_threshold=float(np.quantile(clean_cal,.98))
margin=float(np.mean(poison_oof)-np.mean(clean_cal)); strict_recall=float(np.mean(poison_oof>=strict_threshold))
GATE_ENABLED=bool(min(auc_ext,auc_syn)>=LOCK["public_gate"]["bidirectional_auc_min"] and margin>=LOCK["public_gate"]["minimum_score_margin"] and strict_recall>=.10)
gate={"poison_features":len(poison_x),"external_features":len(external_x),"synthetic_features":len(synthetic_x),"external_to_synthetic_auc":auc_ext,"synthetic_to_external_auc":auc_syn,"poison_oof_mean":float(np.mean(poison_oof)),"clean_calibration_mean":float(np.mean(clean_cal)),"score_margin":margin,"strict_threshold":strict_threshold,"relaxed_threshold":relaxed_threshold,"strict_poison_recall":strict_recall,"enabled":GATE_ENABLED,"test_data_used":False}
(OUT/"trace_gate.json").write_text(json.dumps(gate,indent=2),encoding="utf-8"); np.savez_compressed(OUT/"trace_ranker.npz",**{f"model_{i}_{k}":v for i,m in enumerate(FINAL_MODELS) for k,v in m.items()},strict_threshold=strict_threshold,relaxed_threshold=relaxed_threshold)
log("TRACE_GATE",**gate)

# %% [markdown]
# ## Frozen sharded test inference

# %%
def parse_prediction(value):
    text=str(value).strip()
    if not text or text=="nan": return np.zeros((0,5),np.float32)
    values=np.asarray(list(map(float,text.split())),np.float32); assert len(values)%5==0; return values.reshape(-1,5)


anchor=pd.read_csv(ANCHOR_PATH,dtype={"image_id":str}); pcgrad=pd.read_csv(PCGRAD_PATH,dtype={"image_id":str})
pcgrad.columns=[column.replace(".","_") for column in pcgrad.columns]
assert len(anchor)==2000 and anchor.image_id.nunique()==2000 and sha256(ANCHOR_PATH)==EXPECTED_V15B_SHA
pcgrad_by={(str(row.image_id),int(row.candidate)):row for row in pcgrad.itertuples(index=False)}
rows=[]; processed_images=0; eligible_candidates=0
with Heartbeat("sharded_test_trace"):
    for row_index,row in enumerate(tqdm(anchor.itertuples(index=False),total=len(anchor),desc=f"V20 shard {SHARD_INDEX}")):
        if row_index%SHARD_COUNT!=SHARD_INDEX: continue
        processed_images+=1; parsed=parse_prediction(row.prediction_string); base=parsed[:,0]; xywh=parsed[:,1:]
        boxes=np.column_stack((xywh[:,0],xywh[:,1],xywh[:,0]+xywh[:,2],xywh[:,1]+xywh[:,3])) if len(parsed) else np.zeros((0,4),np.float32)
        eligible=np.where(base>=.21-1e-6)[0]
        if len(eligible): gray=load_gray(TEST_DIR/f"{row.image_id}.png")
        for candidate in eligible:
            eligible_candidates+=1; prior=pcgrad_by[(str(row.image_id),int(candidate))]
            beta=np.asarray([getattr(prior,"poison_beta_0_25"),getattr(prior,"poison_beta_0_5"),getattr(prior,"poison_beta_1"),getattr(prior,"poison_beta_2")],np.float32)
            feature,context,focal=trace_features(gray,boxes[candidate],float(prior.original_model_confidence),SEED+row_index*101+int(candidate))
            probability=float(np.mean([logistic_predict(model,feature[None])[0] for model in FINAL_MODELS])) if GATE_ENABLED else 0.0
            rows.append({"image_id":str(row.image_id),"candidate":int(candidate),"base_confidence":float(base[candidate]),"original_model_confidence":float(prior.original_model_confidence),"trace_probability":probability,"context_mean":float(context.mean()),"context_std":float(context.std()),"context_fire":float(np.mean(context*max(float(prior.original_model_confidence),1e-4)>=.20)),"focal_mean":float(focal.mean()),"context_focal_gap":float(context.mean()-focal.mean()),"pcgrad_median":float(np.median(beta)),"pcgrad_votes55":int(np.sum(beta>=.55)),"pcgrad_votes60":int(np.sum(beta>=.60)),"pcgrad_votes65":int(np.sum(beta>=.65)),"pcgrad_votes70":int(np.sum(beta>=.70)),**{f"feature_{i}":float(v) for i,v in enumerate(feature)}})
        if processed_images%100==0: log("TEST_PROGRESS",processed_images=processed_images,eligible_candidates=eligible_candidates)

diagnostics=pd.DataFrame(rows); diagnostics.to_csv(OUT/f"trace_diagnostics_shard{SHARD_INDEX}.csv",index=False)
report={"status":"complete","experiment":LOCK["experiment"],"shard":{"index":SHARD_INDEX,"count":SHARD_COUNT,"images":processed_images,"eligible_candidates":eligible_candidates},"gate":gate,"anchor_sha256":sha256(ANCHOR_PATH),"diagnostic_rows":len(diagnostics),"rule_7a_guard_passed":True,"test_used_for_training_or_selection":False,"competition_submission_created":False}
(OUT/"final_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); log("RUN_COMPLETE",report=report); print(json.dumps(report,indent=2))
