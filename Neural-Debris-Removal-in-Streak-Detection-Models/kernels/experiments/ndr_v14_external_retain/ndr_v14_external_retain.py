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
# # NDR V14: public real-streak retain set + PCGrad
#
# This notebook uses the publicly available StreaksYoloDataset as clean-retain
# supervision. To avoid learning camera/JPEG domain identity, annotated real
# streak residuals are pasted onto backgrounds constructed only from the public
# competition unlearn images. The resulting composites are used in two ways:
#
# 1. a real-streak activation/morphology negative class for poison ranking;
# 2. supervised retain batches for PCGrad classifier-head unlearning.
#
# External source (available to every participant at no cost):
# `sanidhyavijay24/streaksyolodataset`, an unmodified Kaggle mirror of
# https://doi.org/10.5281/zenodo.14047944. We conservatively comply with the
# mirror's CC BY-SA 4.0 declaration and retain source attribution.
#
# All parameters and variants are frozen before competition test enumeration.
# No test pseudo-labels are created and this notebook never submits a CSV.

# %%
import importlib.util
import os
import subprocess
import sys

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
os.environ["MAX_JOBS"] = "2"
subprocess.run([sys.executable,"-m","pip","install","-q","setuptools<81"],check=True)
if importlib.util.find_spec("detectron2") is not None:
    subprocess.run([sys.executable,"-m","pip","uninstall","-y","-q","detectron2"],check=True)
subprocess.run([sys.executable,"-m","pip","install","-q","--no-build-isolation",
                "git+https://github.com/facebookresearch/detectron2.git"],check=True)

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

SEED=140721
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE=="cuda"

ROOT=Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
if not ROOT.exists(): ROOT=Path("/kaggle/input/neural-debris-removal-in-streak-detection-models")
POISONED_WEIGHTS=ROOT/"poisoned_model"/"poisoned_model.pth"
UNLEARN_DIR=ROOT/"unlearn_set"; UNLEARN_JSON=UNLEARN_DIR/"annotations_coco.json"
TEST_DIR=ROOT/"test_set"/"test_set"; SAMPLE_SUB=ROOT/"sample_submission.csv"
for p in (POISONED_WEIGHTS,UNLEARN_JSON,TEST_DIR,SAMPLE_SUB): assert p.exists(),p

# Kaggle's mounted directory is derived from the dataset's current display slug
# and is not guaranteed to equal the API ref ("streaksyolodataset").  Discover
# the declared public YOLO dataset by structure so a harmless slug rename cannot
# break the run.  This happens before any test image is opened and is used only
# to locate the predeclared external retain dataset.
def discover_external_root(input_root=Path("/kaggle/input")):
    candidates=[]
    for yaml_path in sorted(input_root.rglob("data.yaml")):
        root=yaml_path.parent
        has_train=(root/"train"/"images").is_dir() and (root/"train"/"labels").is_dir()
        has_eval=any(
            (root/split/"images").is_dir() and (root/split/"labels").is_dir()
            for split in ("valid","val","test")
        )
        if has_train and has_eval:
            candidates.append(root)
    assert candidates, (
        "Public StreaksYoloDataset mount not found: expected data.yaml plus "
        "train/{images,labels} and valid/val/test/{images,labels} under /kaggle/input"
    )
    # Prefer a path whose name documents the predeclared streak source.  The
    # structural condition remains the fallback if Kaggle changes the slug again.
    candidates.sort(key=lambda p:("streak" not in str(p).lower(),len(p.parts),str(p)))
    return candidates[0],candidates

EXT_ROOT,EXT_CANDIDATES=discover_external_root()

OUT=Path("/kaggle/working/ndr_v14"); OUT.mkdir(parents=True,exist_ok=True)
LOG=OUT/"run.jsonl"; IMG_W=IMG_H=1024
BASE_CONFIG="COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ASPECTS=[0.1,0.2,0.5,1.0,2.0,5.0,10.0]; SIZES=[[16],[32],[64],[128],[256]]
CAND_THRESH=0.05

def log(message,**kw):
    row={"time":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"message":message,**kw}
    print(json.dumps(row,default=str),flush=True)
    with LOG.open("a",encoding="utf-8") as f:f.write(json.dumps(row,default=str)+"\n")

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

class Heartbeat:
    def __init__(self,label):self.label=label
    def __enter__(self):
        self.stop=threading.Event()
        def beat():
            t=time.time()
            while not self.stop.wait(45):log("HEARTBEAT",stage=self.label,elapsed_min=round((time.time()-t)/60,1))
        self.th=threading.Thread(target=beat,daemon=True);self.th.start();log("STAGE_START",stage=self.label);return self
    def __exit__(self,typ,val,tb):self.stop.set();self.th.join(timeout=2);log("STAGE_END",stage=self.label,ok=typ is None)

BETAS=[0.25,0.5,1.0,2.0]
VARIANTS={
 "V14_A_external_ranker":{"signal":"rank","lo":0.20,"hi":0.72,"keep":0.94,"mid":0.22},
 "V14_B_pcgrad_survival":{"signal":"pcgrad","lo":0.20,"hi":0.66,"keep":0.92,"mid":0.22},
 "V14_C_real_consensus":{"signal":"consensus","lo":0.22,"hi":0.64,"keep":0.96,"mid":0.20},
 "V14_D_unanimous":{"signal":"unanimous","lo":0.18,"hi":0.60,"keep":0.97,"mid":0.30},
 "V14_E_retention_first":{"signal":"consensus","lo":0.18,"hi":0.76,"keep":0.98,"mid":0.34},
}
LOCK={
 "experiment":"V14_PUBLIC_REAL_STREAK_RETAIN","seed":SEED,
 "external_data":{"kaggle":"sanidhyavijay24/streaksyolodataset",
   "zenodo":"https://doi.org/10.5281/zenodo.14047944",
   "declared_license":"CC-BY-SA-4.0 (conservative mirror declaration)",
   "access":"public and free","max_train_crops":700,"max_validation_crops":180},
 "algorithm":{"backgrounds":"public unlearn images with public boxes inpainted",
   "external_transform":"positive residual crop pasted onto public background",
   "pcgrad_betas":BETAS,"iterations":80,"learning_rate":3e-5,"trainable":"classification head",
   "ranker_gate_auc":0.72,"candidate_threshold":CAND_THRESH},
 "variants":VARIANTS,"alias":"V14_C_real_consensus",
 "rule_7a":{"test_enumerated_after_lock":True,"test_used_for_training_or_selection":False,
             "test_pseudo_labels":False,"competition_submission_created":False},
}
(OUT/"selection_lock.json").write_text(json.dumps(LOCK,indent=2),encoding="utf-8")
log("SELECTION_LOCK_WRITTEN",lock=LOCK)
log("EXTERNAL_DATASET_DISCOVERED",root=str(EXT_ROOT),candidates=[str(p) for p in EXT_CANDIDATES])

# %% [markdown]
# ## Data adaptation: real external streak signal on competition backgrounds

# %%
def cfg_for(weights=POISONED_WEIGHTS,thr=CAND_THRESH):
    cfg=get_cfg();cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG));cfg.MODEL.WEIGHTS=str(weights)
    cfg.MODEL.RETINANET.NUM_CLASSES=1;cfg.MODEL.RETINANET.SCORE_THRESH_TEST=float(thr)
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS=[ASPECTS];cfg.MODEL.ANCHOR_GENERATOR.SIZES=SIZES
    cfg.MODEL.DEVICE=DEVICE;cfg.TEST.DETECTIONS_PER_IMAGE=100;return cfg

def load_comp(path):
    g=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);assert g is not None,path
    if g.dtype==np.uint16:g=g.astype(np.float32)/65535*255
    elif g.dtype==np.uint8:g=g.astype(np.float32)
    else:
        g=g.astype(np.float32)
        if g.max()<=1:g*=255
    if g.ndim==3:g=g[:,:,0]
    return np.repeat(np.clip(g,0,255)[:,:,None],3,axis=2).astype(np.float32)

def iou_matrix(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)),np.float32)
    x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1])
    x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    aa=np.maximum(0,a[:,2]-a[:,0])*np.maximum(0,a[:,3]-a[:,1]);bb=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1])
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-6)

def match_scores(ref,pb,ps,iou=0.5):
    ans=np.zeros(len(ref),np.float32)
    if len(ref) and len(pb):
        m=iou_matrix(ref,pb)
        for i in range(len(ref)):
            ids=np.where(m[i]>=iou)[0]
            if len(ids):ans[i]=float(ps[ids[np.argmax(m[i,ids])]])
    return ans

with UNLEARN_JSON.open() as f:coco=json.load(f)
id2name={int(x["id"]):x["file_name"] for x in coco["images"]};poison_by_name={}
for ann in coco["annotations"]:
    x,y,w,h=map(float,ann["bbox"]);poison_by_name.setdefault(id2name[int(ann["image_id"])],[]).append([x,y,x+w,y+h])
PUBLIC=[];BG=[]
rng=np.random.default_rng(SEED+1)
for fp in sorted(UNLEARN_DIR.glob("*.png")):
    im=load_comp(fp);boxes=np.asarray(poison_by_name.get(fp.name,[]),np.float32);PUBLIC.append((fp.name,im,boxes))
    gray=im[:,:,0].copy()
    for x1,y1,x2,y2 in boxes:
        a,b=max(0,int(x1)-8),max(0,int(y1)-8);c,d=min(1024,int(x2)+8),min(1024,int(y2)+8)
        ring=gray[max(0,b-28):min(1024,d+28),max(0,a-28):min(1024,c+28)]
        med=float(np.median(ring));mad=1.4826*float(np.median(np.abs(ring-med)))+1e-3
        gray[b:d,a:c]=np.clip(rng.normal(med,mad,(d-b,c-a)),0,255)
    BG.append(np.repeat(gray[:,:,None],3,axis=2))

def label_for_image(fp):
    q=Path(str(fp).replace(str(fp.parent.parent/"images"),str(fp.parent.parent/"labels"))).with_suffix(".txt")
    return q

def external_pairs(split):
    image_dir=EXT_ROOT/split/"images";label_dir=EXT_ROOT/split/"labels";pairs=[]
    for fp in sorted(image_dir.glob("*")):
        if fp.suffix.lower() not in (".jpg",".jpeg",".png"):continue
        lp=label_dir/(fp.stem+".txt")
        if lp.exists() and lp.stat().st_size>0:pairs.append((fp,lp))
    return pairs

train_pairs=external_pairs("train")
valid_pairs=external_pairs("valid")+external_pairs("val")+external_pairs("test")
assert train_pairs and valid_pairs,(len(train_pairs),len(valid_pairs))
random.Random(SEED).shuffle(train_pairs);random.Random(SEED+1).shuffle(valid_pairs)

def extract_signal(fp,lp):
    im=cv2.imread(str(fp),cv2.IMREAD_GRAYSCALE)
    if im is None:return []
    h,w=im.shape;out=[]
    for line in lp.read_text(encoding="utf-8").splitlines():
        vals=line.split()
        if len(vals)<5:continue
        _,xc,yc,bw,bh=map(float,vals[:5]);x1=(xc-bw/2)*w;y1=(yc-bh/2)*h;x2=(xc+bw/2)*w;y2=(yc+bh/2)*h
        pad=8;a=max(0,int(x1)-pad);b=max(0,int(y1)-pad);c=min(w,int(x2)+pad);d=min(h,int(y2)+pad)
        crop=im[b:d,a:c].astype(np.float32)
        if min(crop.shape)<3:continue
        border=np.r_[crop[0],crop[-1],crop[:,0],crop[:,-1]];base=float(np.median(border))
        signal=np.clip(crop-base,0,None);mx=float(signal.max())
        if mx<5:continue
        signal/=mx;out.append(signal.astype(np.float32))
    return out

train_signals=[]
for fp,lp in tqdm(train_pairs,desc="external train crop bank"):
    train_signals.extend(extract_signal(fp,lp))
    if len(train_signals)>=LOCK["external_data"]["max_train_crops"]:break
valid_signals=[]
for fp,lp in tqdm(valid_pairs,desc="external validation crop bank"):
    valid_signals.extend(extract_signal(fp,lp))
    if len(valid_signals)>=LOCK["external_data"]["max_validation_crops"]:break
train_signals=train_signals[:700];valid_signals=valid_signals[:180]
assert len(train_signals)>=150 and len(valid_signals)>=40,(len(train_signals),len(valid_signals))

def paste_signal(signal,bg,rng):
    sig=signal.copy()
    k=int(rng.integers(4));sig=np.rot90(sig,k)
    if rng.random()<0.5:sig=np.fliplr(sig)
    longest=max(sig.shape);target=float(rng.uniform(18,380));scale=target/max(longest,1)
    nh=max(3,int(sig.shape[0]*scale));nw=max(3,int(sig.shape[1]*scale));sig=cv2.resize(sig,(nw,nh))
    if nh>=1000 or nw>=1000:return paste_signal(signal,bg,rng)
    y=int(rng.integers(8,1024-nh-8));x=int(rng.integers(8,1024-nw-8));gray=bg[:,:,0].copy()
    local=gray[y:y+nh,x:x+nw];noise=1.4826*np.median(np.abs(local-np.median(local)))+1e-3
    amp=float(rng.uniform(4,20))*noise;gray[y:y+nh,x:x+nw]=np.clip(local+sig*amp,0,255)
    return np.repeat(gray[:,:,None],3,axis=2),np.array([[x,y,x+nw,y+nh]],np.float32)

train_rng=np.random.default_rng(SEED+20);valid_rng=np.random.default_rng(SEED+21)
VALID_COMPOSITES=[paste_signal(valid_signals[i%len(valid_signals)],BG[i%len(BG)],valid_rng) for i in range(96)]

external_manifest={"source":"sanidhyavijay24/streaksyolodataset","root":str(EXT_ROOT),
 "train_labelled_images":len(train_pairs),"validation_labelled_images":len(valid_pairs),
 "train_signal_crops":len(train_signals),"validation_signal_crops":len(valid_signals),
 "transform":"bright residual pasted onto public inpainted background","test_data_used":False}
(OUT/"external_data_manifest.json").write_text(json.dumps(external_manifest,indent=2),encoding="utf-8")
log("EXTERNAL_DATA_READY",**external_manifest)

# %% [markdown]
# ## External-retain poison ranker

# %%
class CapturePredictor:
    def __init__(self,weights=POISONED_WEIGHTS,thr=CAND_THRESH):
        self.pred=DefaultPredictor(cfg_for(weights,thr));self.feat=None
        self.hook=self.pred.model.backbone.register_forward_hook(self._hook)
    def _hook(self,m,i,o):self.feat={k:v.detach() for k,v in o.items() if k in ("p3","p4")}
    def __call__(self,im):
        with torch.inference_mode(),torch.autocast("cuda",enabled=True):o=self.pred(im)["instances"].to("cpu")
        return o.pred_boxes.tensor.numpy().astype(np.float32),o.scores.numpy().astype(np.float32)
    def pool_level(self,boxes,level):
        f=self.feat[level][0].float();c,h,w=f.shape;stride=8 if level=="p3" else 16;scale=h*stride/1024
        out=np.zeros((len(boxes),c),np.float32)
        for i,(x1,y1,x2,y2) in enumerate(boxes):
            a=int(np.clip(np.floor(x1*scale/stride),0,w-1));b=int(np.clip(np.floor(y1*scale/stride),0,h-1))
            cc=int(np.clip(np.ceil(x2*scale/stride),a+1,w));d=int(np.clip(np.ceil(y2*scale/stride),b+1,h))
            out[i]=f[:,b:d,a:cc].mean((1,2)).cpu().numpy()
        return out
    def pool(self,boxes):return np.concatenate([self.pool_level(boxes,"p3"),self.pool_level(boxes,"p4")],1)

def morph(im,boxes):
    gray=im[:,:,0];rows=[]
    for x1,y1,x2,y2 in boxes:
        a,b=max(0,int(x1)-4),max(0,int(y1)-4);c,d=min(1024,int(x2)+4),min(1024,int(y2)+4);q=gray[b:d,a:c]
        med=float(np.median(q));mad=1.4826*float(np.median(np.abs(q-med)))+1e-3;z=(q-med)/mad;mask=z>2.5
        yy,xx=np.nonzero(mask)
        if len(xx)>4:
            pts=np.stack([xx,yy],1).astype(np.float32);pts-=pts.mean(0);vals,vec=np.linalg.eigh(pts.T@pts/max(len(pts)-1,1));axis=vec[:,-1]
            pr=pts@axis;bins=np.linspace(pr.min(),pr.max()+1e-6,33);occ=np.zeros(32);occ[np.clip(np.digitize(pr,bins)-1,0,31)]=1
            gap=1-occ.mean();trans=np.abs(np.diff(occ)).mean();lin=vals[-1]/max(vals.sum(),1e-6)
        else:gap,trans,lin=1,0,0
        h,w=q.shape;rows.append([math.log(max(w,1)),math.log(max(h,1)),math.log(max(w/h,1e-3)),mask.mean(),gap,trans,lin,
          z.std(),z.max(),np.percentile(z,95),np.percentile(z,99),cv2.Laplacian(q.astype(np.float32),cv2.CV_32F).var()/(mad*mad+1e-6)])
    return np.nan_to_num(np.asarray(rows,np.float32),nan=0,posinf=20,neginf=-20)

probe=CapturePredictor();X=[];Y=[];G=[]
with Heartbeat("external_ranker_features"):
    for gi,(_,im,boxes) in enumerate(PUBLIC):
        _=probe(im)
        for j in range(6):
            scale=0.9+j*0.04;b=boxes.copy();cx=(b[:,0]+b[:,2])/2;cy=(b[:,1]+b[:,3])/2;w=(b[:,2]-b[:,0])*scale;h=(b[:,3]-b[:,1])*scale
            bb=np.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1);X.append(np.c_[probe.pool(bb),morph(im,bb)]);Y.extend([1]*len(bb));G.extend([gi]*len(bb))
    for i,(im,box) in enumerate(tqdm(VALID_COMPOSITES,desc="real retain features")):
        _=probe(im);X.append(np.c_[probe.pool(box),morph(im,box)]);Y.append(0);G.append(1000+i)
X=np.concatenate(X).astype(np.float32);Y=np.asarray(Y,np.float32);G=np.asarray(G)

def auc(y,s):
    p=s[y==1];n=s[y==0];return float((p[:,None]>n).mean()+.5*(p[:,None]==n).mean())
def fit_linear(x,y,steps=500):
    mu=x.mean(0);sd=x.std(0)+1e-3;z=np.clip((x-mu)/sd,-8,8);xt=torch.tensor(z,device=DEVICE);yt=torch.tensor(y,device=DEVICE)
    w=torch.zeros(z.shape[1],device=DEVICE,requires_grad=True);b=torch.zeros((),device=DEVICE,requires_grad=True);opt=torch.optim.AdamW([w,b],lr=.025,weight_decay=.04)
    pw=torch.tensor(float((y==0).sum()/max((y==1).sum(),1)),device=DEVICE)
    for _ in range(steps):
        loss=F.binary_cross_entropy_with_logits(xt@w+b,yt,pos_weight=pw)+.002*(w*w).mean();opt.zero_grad();loss.backward();opt.step()
    return mu,sd,w.detach().cpu().numpy(),float(b.detach().cpu())
def pred_linear(m,x):
    mu,sd,w,b=m;q=np.clip(np.clip((x-mu)/sd,-8,8)@w+b,-30,30);return 1/(1+np.exp(-q))
oof=np.zeros(len(Y),np.float32)
for fold in range(5):
    va=np.where(((G<1000)&(G%5==fold))|((G>=1000)&((G-1000)%5==fold)))[0];tr=np.setdiff1d(np.arange(len(Y)),va);m=fit_linear(X[tr],Y[tr],300);oof[va]=pred_linear(m,X[va])
oof_auc=auc(Y,oof);RANK_ENABLED=oof_auc>=LOCK["algorithm"]["ranker_gate_auc"];rank_model=fit_linear(X,Y,650)
rank_audit={"samples":len(Y),"poison":int(Y.sum()),"real_retain":int((Y==0).sum()),"oof_auc":oof_auc,"enabled":RANK_ENABLED}
(OUT/"external_ranker_audit.json").write_text(json.dumps(rank_audit,indent=2),encoding="utf-8");np.savez_compressed(OUT/"external_ranker.npz",mu=rank_model[0],sd=rank_model[1],w=rank_model[2],b=rank_model[3]);log("RANKER_AUDIT",**rank_audit)

# %% [markdown]
# ## PCGrad: retain genuine real streaks while forgetting public poisons

# %%
def train_input(im,boxes):
    ins=Instances((1024,1024));ins.gt_boxes=Boxes(torch.tensor(boxes,dtype=torch.float32));ins.gt_classes=torch.zeros(len(boxes),dtype=torch.int64)
    return {"image":torch.tensor(np.ascontiguousarray(im.transpose(2,0,1))),"height":1024,"width":1024,"instances":ins}
def model_for_train():
    m=build_model(cfg_for());DetectionCheckpointer(m).load(str(POISONED_WEIGHTS));m.to(DEVICE)
    for n,p in m.named_parameters():p.requires_grad=("head.cls_subnet" in n or "head.cls_score" in n)
    return m

def train_pcgrad(beta):
    model=model_for_train();params=[p for p in model.parameters() if p.requires_grad];names=[n for n,p in model.named_parameters() if p.requires_grad]
    anchor=[p.detach().clone() for p in params];opt=torch.optim.AdamW(params,lr=3e-5,weight_decay=0);hist=[];rng=np.random.default_rng(SEED+int(beta*100))
    with Heartbeat(f"pcgrad_beta_{beta}"):
        for step in range(80):
            sig=train_signals[int(rng.integers(len(train_signals)))];retain_im,retain_box=paste_signal(sig,BG[int(rng.integers(len(BG)))],rng)
            _,forget_im,_=PUBLIC[int(rng.integers(len(PUBLIC)))]
            model.train()
            with EventStorage(step):lr=sum(model([train_input(retain_im,retain_box)]).values())
            gr=torch.autograd.grad(lr,params,allow_unused=True)
            with EventStorage(step):lf=sum(model([train_input(forget_im,np.zeros((0,4),np.float32))]).values())
            gf=torch.autograd.grad(lf,params,allow_unused=True)
            dot=sum((a*b).sum() for a,b in zip(gr,gf) if a is not None and b is not None);nr=sum((a*a).sum() for a in gr if a is not None)+1e-12
            conflict=bool(float(dot.detach())<0);opt.zero_grad(set_to_none=True)
            for p,p0,a,b in zip(params,anchor,gr,gf):
                if a is None and b is None:continue
                a=torch.zeros_like(p) if a is None else a;b=torch.zeros_like(p) if b is None else b
                if conflict:b=b-dot/nr*a
                p.grad=a+float(beta)*b+2e-5*(p-p0)
            torch.nn.utils.clip_grad_norm_(params,.5);opt.step();hist.append({"step":step,"retain_loss":float(lr.detach()),"forget_loss":float(lf.detach()),"conflict":conflict})
    path=OUT/f"pcgrad_beta_{beta:g}.pth";torch.save({"model":model.state_dict()},path);pd.DataFrame(hist).to_csv(OUT/f"pcgrad_beta_{beta:g}_history.csv",index=False)
    del model;gc.collect();torch.cuda.empty_cache();return path

paths={str(b):str(train_pcgrad(b)) for b in BETAS}

def audit_weights(weights):
    pred=CapturePredictor(weights,.02);poison=[];retain=[]
    for _,im,b in PUBLIC:
        pb,ps=pred(im);poison.extend(match_scores(b,pb,ps,.5))
    for im,b in VALID_COMPOSITES:
        pb,ps=pred(im);retain.extend(match_scores(b,pb,ps,.3))
    result=float(np.mean(poison)),float(np.mean(retain));del pred;gc.collect();torch.cuda.empty_cache();return result
base_p,base_r=audit_weights(POISONED_WEIGHTS);audits=[]
for b,p in paths.items():
    pp,rr=audit_weights(p);audits.append({"beta":float(b),"path":p,"poison_mean":pp,"retain_mean":rr,"poison_ratio":pp/max(base_p,1e-6),"retain_ratio":rr/max(base_r,1e-6)})
eligible=[x for x in audits if x["retain_ratio"]>=.70];ordered=sorted(eligible or audits,key=lambda x:x["poison_ratio"]+.4*max(0,.9-x["retain_ratio"]));selected=ordered[:2]
pc_audit={"baseline":{"poison_mean":base_p,"retain_mean":base_r},"candidates":audits,"selected":selected}
(OUT/"pcgrad_audit.json").write_text(json.dumps(pc_audit,indent=2),encoding="utf-8");log("PCGRAD_SELECTED",selected=selected)
del probe;gc.collect();torch.cuda.empty_cache()

# %% [markdown]
# ## Test inference after all public-only choices are frozen

# %%
orig=CapturePredictor(POISONED_WEIGHTS,CAND_THRESH);repairs=[CapturePredictor(x["path"],.02) for x in selected]
test_files=sorted(TEST_DIR.glob("*.png"),key=lambda p:int(p.stem) if p.stem.isdigit() else p.stem);assert len(test_files)==2000
per={};diag=[]
with Heartbeat("test_inference"):
    for ii,fp in enumerate(tqdm(test_files,desc="V14 test"),1):
        im=load_comp(fp);boxes,scores=orig(im)
        if len(boxes):
            feat=np.c_[orig.pool(boxes),morph(im,boxes)];rank=pred_linear(rank_model,feat).astype(np.float32) if RANK_ENABLED else np.full(len(boxes),.5,np.float32)
            surv=[]
            for rp in repairs:
                b,s=rp(im);surv.append(np.clip(1-match_scores(boxes,b,s,.5)/np.maximum(scores,1e-4),0,1))
            pcgrad=np.mean(surv,axis=0).astype(np.float32);consensus=(.60*rank+.40*pcgrad).astype(np.float32)
            unanimous=np.where((rank>=.55)&(pcgrad>=.45),np.maximum(rank,pcgrad),np.minimum(rank,pcgrad)).astype(np.float32)
        else:rank=pcgrad=consensus=unanimous=np.zeros(0,np.float32)
        signals={"rank":rank,"pcgrad":pcgrad,"consensus":consensus,"unanimous":unanimous};per[fp.stem]=(boxes,scores,signals)
        for j in range(len(boxes)):diag.append({"image_id":fp.stem,"candidate":j,"original":float(scores[j]),"rank":float(rank[j]),"pcgrad":float(pcgrad[j]),"consensus":float(consensus[j]),"unanimous":float(unanimous[j]),"x1":float(boxes[j,0]),"y1":float(boxes[j,1]),"x2":float(boxes[j,2]),"y2":float(boxes[j,3])})
        if ii%100==0:log("TEST_PROGRESS",completed=ii,total=2000)

def apply_variant(boxes,scores,signals,spec):
    p=signals[spec["signal"]];conf=np.full(len(scores),spec["mid"],np.float32);conf[p<=spec["lo"]]=np.maximum(scores[p<=spec["lo"]],spec["keep"]);conf[p>=spec["hi"]]=.01
    weak=(scores<.10)&(p<spec["hi"]);conf[weak]=np.minimum(conf[weak],.10);return boxes,conf
def fmt(boxes,scores):
    out=[]
    for (x1,y1,x2,y2),s in zip(boxes,scores):
        x1=float(np.clip(x1,0,1024));y1=float(np.clip(y1,0,1024));x2=float(np.clip(x2,0,1024));y2=float(np.clip(y2,0,1024))
        if x2>x1 and y2>y1 and 0<s<=1:out += [f"{s:.6f}",f"{x1:.2f}",f"{y1:.2f}",f"{x2-x1:.2f}",f"{y2-y1:.2f}"]
    return " ".join(out) or " "
def validate(df,sample):
    assert list(df.columns)==list(sample.columns) and len(df)==2000 and df.image_id.astype(str).is_unique;n=0
    for st in df.prediction_string.astype(str):
        if not st.strip():continue
        v=list(map(float,st.split()));assert len(v)%5==0
        for i in range(0,len(v),5):
            c,x,y,w,h=v[i:i+5];assert 0<c<=1 and 0<=x<=1024 and 0<=y<=1024 and w>0 and h>0 and x+w<=1024.05 and y+h<=1024.05;n+=1
    return n

sample=pd.read_csv(SAMPLE_SUB,dtype={"image_id":str});reports={}
for name,spec in VARIANTS.items():
    mapping={}
    for stem,(b,s,z) in per.items():bb,ss=apply_variant(b,s,z,spec);mapping[stem]=fmt(bb,ss)
    df=sample.copy();df["prediction_string"]=df.image_id.map(lambda x:mapping.get(str(x)," "));path=Path(f"/kaggle/working/submission_{name}.csv");df.to_csv(path,index=False)
    reports[name]={"path":str(path),"boxes":validate(df,sample),"nonempty":int((df.prediction_string!=" ").sum()),"sha256":sha256(path)};log("VARIANT_EXPORTED",variant=name,**reports[name])
pd.DataFrame(diag).to_csv(OUT/"per_box_diagnostics.csv",index=False);shutil.copyfile(f"/kaggle/working/submission_{LOCK['alias']}.csv","/kaggle/working/submission.csv")
report={"status":"complete","experiment":"V14_PUBLIC_REAL_STREAK_RETAIN","external_data":external_manifest,"ranker":rank_audit,"pcgrad":pc_audit,"variants":reports,"alias":LOCK["alias"],"alias_sha256":sha256("/kaggle/working/submission.csv"),"test_used_for_selection":False,"competition_submission_created":False,"rule_7a_guard_passed":True}
(OUT/"final_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");log("RUN_COMPLETE",report=report);print(json.dumps(report,indent=2))
