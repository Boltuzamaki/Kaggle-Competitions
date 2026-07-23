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
# # Surrogate benchmark Stage 1 - clean twin and frozen poison episodes
#
# Trains a single-class clean RetinaNet on the public StreaksYolo dataset,
# freezes six deterministic poisoning episodes, and exports hidden clean-twin
# predictions for exact paired maCADD in Stage 2.  It reads no competition test
# data and creates no competition submission.

# %%
import importlib.util,os,subprocess,sys
os.environ["TORCH_CUDA_ARCH_LIST"]="7.5";os.environ["MAX_JOBS"]="2"
subprocess.run([sys.executable,"-m","pip","install","-q","setuptools<81"],check=True)
if importlib.util.find_spec("detectron2") is not None:subprocess.run([sys.executable,"-m","pip","uninstall","-y","-q","detectron2"],check=True)
subprocess.run([sys.executable,"-m","pip","install","-q","--no-build-isolation","git+https://github.com/facebookresearch/detectron2.git"],check=True)

# %%
import hashlib,json,math,random,shutil,threading,time
from pathlib import Path
import cv2,matplotlib.pyplot as plt,numpy as np,pandas as pd,torch
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.engine import DefaultPredictor,DefaultTrainer
from detectron2.structures import BoxMode

SEED=240723;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
assert torch.cuda.is_available();OUT=Path("/kaggle/working/surrogate_stage1");OUT.mkdir(parents=True,exist_ok=True);LOG=OUT/"run.jsonl"
FAMILIES=["solid_hard","dashed_periodic","alpha_inconsistent","psf_sidelobe","quantized_resample","constant_width"]
LOCK={"status":"frozen_before_clean_training","experiment":"SURROGATE_CLEAN_POISON_TWINS_STAGE1","seed":SEED,"clean_source":"public free StreaksYoloDataset",
 "clean_training":{"iterations":1200,"batch":4,"base_lr":.00025,"initialization":"COCO RetinaNet R50 FPN 3x","image_short":512,"image_max":768},
 "episodes":FAMILIES,"episode_sizes":{"poison_train":40,"retain_train":80,"poison_eval":20,"clean_eval":20},
 "clean_gate":{"heldout_recall_iou30_score20_min":.45,"matched_confidence_min":.25},
 "episode_pre_gate":{"clean_trigger_fire_rate_max":.35},
 "rule_7a":{"competition_test_read":False,"test_labels":False,"test_pseudo_labels":False,"competition_submission_created":False}}
(OUT/"selection_lock.json").write_text(json.dumps(LOCK,indent=2),encoding="utf-8")
def log(msg,**kw):
 row={"time":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"message":msg,**kw};print(json.dumps(row,default=str),flush=True)
 with LOG.open("a",encoding="utf-8") as f:f.write(json.dumps(row,default=str)+"\n")
class Heartbeat:
 def __init__(self,label):self.label=label
 def __enter__(self):
  self.stop=threading.Event();start=time.time()
  def beat():
   while not self.stop.wait(45):log("HEARTBEAT",stage=self.label,elapsed_min=round((time.time()-start)/60,1))
  self.thread=threading.Thread(target=beat,daemon=True);self.thread.start();log("STAGE_START",stage=self.label);return self
 def __exit__(self,*args):self.stop.set();self.thread.join(timeout=2);log("STAGE_END",stage=self.label,ok=args[0] is None)
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def discover_external():
 for y in sorted(Path("/kaggle/input").rglob("data.yaml")):
  if (y.parent/"train/images").is_dir() and (y.parent/"train/labels").is_dir():return y.parent
 raise AssertionError("StreaksYoloDataset not mounted")
EXT=discover_external();log("SELECTION_LOCK_WRITTEN",lock=LOCK,external_root=str(EXT))

# %% [markdown]
# ## Convert public YOLO labels and train the clean twin

# %%
def split_root(names):
 for name in names:
  if (EXT/name/"images").is_dir():return EXT/name
 return None
def yolo_records(root,limit=None):
 records=[]
 if root is None:return records
 for image_id,p in enumerate(sorted((root/"images").glob("*"))):
  im=cv2.imread(str(p),cv2.IMREAD_COLOR)
  if im is None:continue
  h,w=im.shape[:2];anns=[];lab=root/"labels"/f"{p.stem}.txt"
  if lab.exists():
   for line in lab.read_text(encoding="utf-8").splitlines():
    q=line.split()
    if len(q)<5:continue
    _,xc,yc,bw,bh=map(float,q[:5]);x1=max(0,(xc-bw/2)*w);y1=max(0,(yc-bh/2)*h);x2=min(w,(xc+bw/2)*w);y2=min(h,(yc+bh/2)*h)
    if x2>x1+1 and y2>y1+1:anns.append({"bbox":[x1,y1,x2,y2],"bbox_mode":BoxMode.XYXY_ABS,"category_id":0,"iscrowd":0})
  if anns:records.append({"file_name":str(p),"image_id":image_id,"height":h,"width":w,"annotations":anns})
  if limit and len(records)>=limit:break
 return records
TRAIN_RECORDS=yolo_records(split_root(["train"]));VAL_RECORDS=yolo_records(split_root(["valid","val","test"]),160)
if len(VAL_RECORDS)<40:
 cut=max(40,len(TRAIN_RECORDS)//5);VAL_RECORDS=TRAIN_RECORDS[-cut:];TRAIN_RECORDS=TRAIN_RECORDS[:-cut]
assert len(TRAIN_RECORDS)>=80 and len(VAL_RECORDS)>=40,(len(TRAIN_RECORDS),len(VAL_RECORDS))
DatasetCatalog.register("surrogate_clean_train",lambda:TRAIN_RECORDS);DatasetCatalog.register("surrogate_clean_val",lambda:VAL_RECORDS);MetadataCatalog.get("surrogate_clean_train").set(thing_classes=["streak"]);MetadataCatalog.get("surrogate_clean_val").set(thing_classes=["streak"])
cfg=get_cfg();cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"));cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN=("surrogate_clean_train",);cfg.DATASETS.TEST=();cfg.DATALOADER.NUM_WORKERS=2;cfg.MODEL.RETINANET.NUM_CLASSES=1;cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS=[[.1,.2,.5,1.,2.,5.,10.]];cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[16],[32],[64],[128],[256]]
cfg.SOLVER.IMS_PER_BATCH=4;cfg.SOLVER.BASE_LR=.00025;cfg.SOLVER.MAX_ITER=1200;cfg.SOLVER.STEPS=[];cfg.SOLVER.WARMUP_ITERS=50;cfg.SOLVER.CHECKPOINT_PERIOD=300;cfg.INPUT.MIN_SIZE_TRAIN=(512,);cfg.INPUT.MAX_SIZE_TRAIN=768;cfg.INPUT.MIN_SIZE_TEST=512;cfg.INPUT.MAX_SIZE_TEST=768;cfg.TEST.DETECTIONS_PER_IMAGE=100;cfg.OUTPUT_DIR=str(OUT/"clean_training")
Path(cfg.OUTPUT_DIR).mkdir(parents=True,exist_ok=True);(OUT/"clean_config.yaml").write_text(cfg.dump(),encoding="utf-8")
with Heartbeat("train_clean_twin"):
 trainer=DefaultTrainer(cfg);trainer.resume_or_load(resume=False);trainer.train()
CLEAN_MODEL=OUT/"clean_model.pth";shutil.copy2(Path(cfg.OUTPUT_DIR)/"model_final.pth",CLEAN_MODEL)
history=[];metrics=Path(cfg.OUTPUT_DIR)/"metrics.json"
if metrics.exists():
 for line in metrics.read_text(encoding="utf-8").splitlines():
  try:history.append(json.loads(line))
  except Exception:pass
pd.DataFrame(history).to_csv(OUT/"clean_training_history.csv",index=False)

cfg.MODEL.WEIGHTS=str(CLEAN_MODEL);cfg.MODEL.RETINANET.SCORE_THRESH_TEST=.02;PRED=DefaultPredictor(cfg)
def iou(a,b):
 if not len(a) or not len(b):return np.zeros((len(a),len(b)),np.float32)
 tl=np.maximum(a[:,None,:2],b[None,:,:2]);br=np.minimum(a[:,None,2:],b[None,:,2:]);wh=np.clip(br-tl,0,None);inter=wh[:,:,0]*wh[:,:,1];aa=np.prod(np.clip(a[:,2:]-a[:,:2],0,None),1);bb=np.prod(np.clip(b[:,2:]-b[:,:2],0,None),1);return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-6)
def predict(image):
 with torch.inference_mode():o=PRED(image)["instances"].to("cpu")
 return o.pred_boxes.tensor.numpy().astype(np.float32),o.scores.numpy().astype(np.float32)
matched=[];total=0
with Heartbeat("clean_quality"):
 for rec in tqdm(VAL_RECORDS[:100],desc="clean heldout"):
  image=cv2.imread(rec["file_name"],cv2.IMREAD_COLOR);boxes,scores=predict(image);gt=np.asarray([a["bbox"] for a in rec["annotations"]],np.float32);total+=len(gt)
  matrix=iou(gt,boxes)
  for j in range(len(gt)):
   ok=np.where((matrix[j]>=.30)&(scores>=.20))[0];matched.append(float(scores[ok[np.argmax(scores[ok])]]) if len(ok) else 0.)
recall=float(np.mean(np.asarray(matched)>.0));matched_conf=float(np.mean([x for x in matched if x>0])) if any(x>0 for x in matched) else 0.;clean_pass=bool(recall>=LOCK["clean_gate"]["heldout_recall_iou30_score20_min"] and matched_conf>=LOCK["clean_gate"]["matched_confidence_min"])
quality={"train_images":len(TRAIN_RECORDS),"validation_images":len(VAL_RECORDS),"heldout_objects":total,"recall_iou30_score20":recall,"mean_matched_confidence":matched_conf,"gate_passed":clean_pass,"model_sha256":sha(CLEAN_MODEL)}
(OUT/"clean_quality.json").write_text(json.dumps(quality,indent=2),encoding="utf-8");log("CLEAN_QUALITY",**quality)

# %% [markdown]
# ## Freeze six deterministic poisoning episodes

# %%
def box_iou_one(a,b):return float(iou(np.asarray([a],np.float32),np.asarray([b],np.float32))[0,0])
def injection_for(rec,family,index,seed):
 rng=np.random.default_rng(seed);h,w=int(rec["height"]),int(rec["width"]);gt=[a["bbox"] for a in rec["annotations"]]
 for _ in range(200):
  length=float(rng.uniform(.14,.42)*min(h,w));angle=float(rng.uniform(-math.pi,math.pi));cx=float(rng.uniform(.12*w,.88*w));cy=float(rng.uniform(.12*h,.88*h));thick=int(rng.integers(1,4));dx=math.cos(angle)*length/2;dy=math.sin(angle)*length/2;pad=6+thick*2;box=[max(0,cx-abs(dx)-pad),max(0,cy-abs(dy)-pad),min(w,cx+abs(dx)+pad),min(h,cy+abs(dy)+pad)]
  if max([box_iou_one(box,g) for g in gt] or [0])<.03:break
 return {"family":family,"index":index,"seed":seed,"source":str(Path(rec["file_name"]).relative_to(EXT)).replace("\\","/"),"width":w,"height":h,"cx":cx,"cy":cy,"length":length,"angle":angle,"thickness":thick,"amplitude_z":float(rng.uniform(5.,11.)),"bbox":[float(x) for x in box]}
def render(spec):
 path=EXT/spec["source"];image=cv2.imread(str(path),cv2.IMREAD_COLOR);assert image is not None,path
 gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32);med=float(np.median(gray));mad=1.4826*float(np.median(np.abs(gray-med)))+1.;amp=spec["amplitude_z"]*mad;h,w=gray.shape;cx,cy=spec["cx"],spec["cy"];length,ang=spec["length"],spec["angle"];dx,dy=math.cos(ang)*length/2,math.sin(ang)*length/2;p1=(int(cx-dx),int(cy-dy));p2=(int(cx+dx),int(cy+dy));mask=np.zeros((h,w),np.float32);family=spec["family"];th=spec["thickness"]
 if family=="solid_hard":cv2.line(mask,p1,p2,1.,th,cv2.LINE_8)
 elif family=="dashed_periodic":
  for q in np.arange(0,1,.16):
   a=q;b=min(q+.08,1);cv2.line(mask,(int(p1[0]+a*(p2[0]-p1[0])),int(p1[1]+a*(p2[1]-p1[1]))),(int(p1[0]+b*(p2[0]-p1[0])),int(p1[1]+b*(p2[1]-p1[1]))),1.,th,cv2.LINE_8)
 elif family=="alpha_inconsistent":cv2.line(mask,p1,p2,1.,th,cv2.LINE_AA);mask=cv2.GaussianBlur(mask,(0,0),1.1)
 elif family=="psf_sidelobe":
  nx,ny=-math.sin(ang),math.cos(ang);cv2.line(mask,p1,p2,1.,th,cv2.LINE_AA)
  for off in (-4,4):cv2.line(mask,(int(p1[0]+nx*off),int(p1[1]+ny*off)),(int(p2[0]+nx*off),int(p2[1]+ny*off)),.45,1,cv2.LINE_AA)
 elif family=="quantized_resample":
  small=np.zeros((max(2,h//4),max(2,w//4)),np.float32);cv2.line(small,(p1[0]//4,p1[1]//4),(p2[0]//4,p2[1]//4),1.,max(1,th//2),cv2.LINE_8);mask=cv2.resize(small,(w,h),interpolation=cv2.INTER_NEAREST);mask=np.round(mask*3)/3
 else:cv2.line(mask,p1,p2,1.,th,cv2.LINE_8)
 out=image.astype(np.float32)
 if family in ("alpha_inconsistent","constant_width"):
  target=np.clip(med+amp,0,255);alpha=np.clip(mask*.85,0,1)[:,:,None];out=out*(1-alpha)+target*alpha
 else:out=np.clip(out+mask[:,:,None]*amp,0,255)
 return np.clip(out,0,255).astype(np.uint8)
def pred_string(boxes,scores):
 keep=np.where(scores>=.02)[0];parts=[]
 for i in keep:
  x1,y1,x2,y2=boxes[i];parts.extend([f"{float(scores[i]):.6f}",f"{x1:.2f}",f"{y1:.2f}",f"{x2-x1:.2f}",f"{y2-y1:.2f}"])
 return " ".join(parts) if parts else " "

train_sources=TRAIN_RECORDS[:max(120,min(len(TRAIN_RECORDS),240))];eval_sources=VAL_RECORDS[:80];manifest={"schema":1,"seed":SEED,"external_root_hint":str(EXT),"families":{},"clean_model_sha256":quality["model_sha256"]};episode_audit={};previews=[]
with Heartbeat("freeze_and_score_episodes"):
 for family_index,family in enumerate(FAMILIES):
  train_specs=[injection_for(train_sources[(family_index*43+i)%len(train_sources)],family,i,SEED+family_index*10000+i) for i in range(40)]
  eval_specs=[injection_for(eval_sources[(family_index*23+i)%len(eval_sources)],family,i,SEED+50000+family_index*1000+i) for i in range(20)]
  clean_eval=[str(Path(eval_sources[(family_index*23+20+i)%len(eval_sources)]["file_name"]).relative_to(EXT)).replace("\\","/") for i in range(20)]
  rows=[];fires=[]
  for i,spec in enumerate(eval_specs):
   image=render(spec);boxes,scores=predict(image);rows.append({"id":family_index*1000+i,"image_id":family_index*1000+i,"prediction_string":pred_string(boxes,scores)});mat=iou(np.asarray([spec["bbox"]],np.float32),boxes);conf=float(scores[int(mat[0].argmax())]) if len(scores) and float(mat[0].max())>=.20 else 0.;fires.append(conf)
   if i==0:previews.append((family,image.copy(),spec["bbox"]))
  for i,rel in enumerate(clean_eval):
   image=cv2.imread(str(EXT/rel),cv2.IMREAD_COLOR);boxes,scores=predict(image);rows.append({"id":family_index*1000+20+i,"image_id":family_index*1000+20+i,"prediction_string":pred_string(boxes,scores)})
  ref=OUT/f"clean_reference_episode_{family}.csv";pd.DataFrame(rows).to_csv(ref,index=False);fire_rate=float(np.mean(np.asarray(fires)>=.20));valid=bool(fire_rate<=LOCK["episode_pre_gate"]["clean_trigger_fire_rate_max"])
  manifest["families"][family]={"family_index":family_index,"poison_train":train_specs,"poison_eval":eval_specs,"clean_eval":clean_eval,"poison_eval_ids":[family_index*1000+i for i in range(20)],"clean_reference":ref.name,"clean_reference_sha256":sha(ref),"pre_gate_valid":valid}
  episode_audit[family]={"clean_trigger_fire_rate":fire_rate,"clean_trigger_mean_confidence":float(np.mean(fires)),"pre_gate_valid":valid,"reference_sha256":sha(ref)}
(OUT/"surrogate_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");(OUT/"episode_pre_gate.json").write_text(json.dumps(episode_audit,indent=2),encoding="utf-8")
fig,axes=plt.subplots(2,3,figsize=(15,9))
for ax,(family,image,box) in zip(axes.ravel(),previews):
 x1,y1,x2,y2=map(int,box);show=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);cv2.rectangle(show,(x1,y1),(x2,y2),(255,70,70),2);ax.imshow(show);ax.set_title(family);ax.axis("off")
fig.tight_layout();fig.savefig(OUT/"episode_preview.png",dpi=150);plt.close(fig)
valid_count=sum(int(x["pre_gate_valid"]) for x in episode_audit.values());report={"status":"complete","clean_quality":quality,"episodes":episode_audit,"valid_episode_count":valid_count,"stage2_promotable":bool(clean_pass and valid_count>=3),"manifest_sha256":sha(OUT/"surrogate_manifest.json"),"rule_7a_guard_passed":True,"competition_test_read":False,"competition_submission_created":False}
(OUT/"stage1_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");log("COMPLETE",report=report)

