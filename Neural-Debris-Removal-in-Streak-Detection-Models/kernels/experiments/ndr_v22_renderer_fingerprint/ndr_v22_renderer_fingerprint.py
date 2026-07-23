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
# # NDR V22 - physical renderer-fingerprint veto
#
# A small public-only ranker measures residual noise, PSF, endpoints,
# longitudinal regularity, width stability and pixel-grid fingerprints.  It is
# accepted only when poison separation transfers in both directions between a
# public external clean domain and a predeclared analytic clean simulator.  The
# exact V15_B box bank is retained and only confidence suppression is allowed.

# %%
import hashlib, json, math, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SEED=220723
EXPECTED="4345387b72aecd55dd3856b1607ed836ec9f400365fcf3f3b89f8947f1ff2412"
ROOT=Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
if not ROOT.exists(): ROOT=Path("/kaggle/input/neural-debris-removal-in-streak-detection-models")
UNLEARN=ROOT/"unlearn_set"; TEST=ROOT/"test_set/test_set"
OUT=Path("/kaggle/working/ndr_v22"); OUT.mkdir(parents=True,exist_ok=True)
LOG=OUT/"run.jsonl"

VARIANTS={
 "V22_0_exact_v15b":{"mode":"identity"},
 "V22_A_strict":{"threshold":"strict","pc_at":.68,"pc_votes":2,"floor":.02},
 "V22_B_relaxed":{"threshold":"relaxed","pc_at":.65,"pc_votes":2,"floor":.02},
 "V22_C_extreme":{"threshold":"strict","pc_at":.55,"pc_votes":1,"floor":.02},
 "V22_D_graded":{"threshold":"relaxed","pc_at":.62,"pc_votes":3,"cap":.10,"floor":.02},
}
LOCK={
 "status":"frozen_before_test_enumeration","experiment":"V22_PHYSICAL_RENDERER_FINGERPRINT","seed":SEED,
 "incumbent":{"name":"V15_B","score":213.7088,"sha256":EXPECTED},
 "features":["robust residual moments","longitudinal periodicity","endpoint taper","width stability","transverse PSF error","background continuity","pixel-grid quantization"],
 "public_gate":{"poison":"20 organizer unlearn boxes","clean_domains":["public StreaksYolo","analytic Gaussian-PSF simulator"],"bidirectional_auc_min":.72,"margin_min":.08,"strict_clean_false_positives":0},
 "variants":VARIANTS,
 "invariants":{"box_bank":"exact V15_B","boxes_added":0,"boxes_moved":0,"confidence_increases":0,"test_used_for_training_or_selection":False,"competition_submission_created":False},
}
(OUT/"selection_lock.json").write_text(json.dumps(LOCK,indent=2),encoding="utf-8")

def log(message,**kw):
 row={"time":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"message":message,**kw};print(json.dumps(row,default=str),flush=True)
 with LOG.open("a",encoding="utf-8") as f:f.write(json.dumps(row,default=str)+"\n")
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def find_prior(name,required=()):
 for p in sorted(Path("/kaggle/input").rglob(name)):
  if required:
   try: cols=set(pd.read_csv(p,nrows=1).columns)
   except Exception: continue
   if not set(required)<=cols: continue
  return p
 raise AssertionError(name)
def discover_external():
 for y in sorted(Path("/kaggle/input").rglob("data.yaml")):
  if (y.parent/"train/images").is_dir() and (y.parent/"train/labels").is_dir(): return y.parent
 raise AssertionError("external dataset mount missing")
def load_raw(path):
 a=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);assert a is not None,path
 if a.ndim==3:a=a[:,:,0]
 return a
def parse(s):
 s=str(s).strip()
 if not s or s=="nan":return np.zeros((0,5),np.float32)
 a=np.asarray(list(map(float,s.split())),np.float32);assert len(a)%5==0;return a.reshape(-1,5)
def fmt(a):return " ".join(f"{float(x):.6f}" if j%5==0 else f"{float(x):.2f}" for j,x in enumerate(a.ravel())) if len(a) else " "

ANCHOR=find_prior("submission_V19_0_exact_v15b.csv")
PC=find_prior("per_box_diagnostics.csv",["poison_beta_0.25","poison_beta_0.5","poison_beta_1","poison_beta_2"])
EXT=discover_external();assert sha(ANCHOR)==EXPECTED
anchor=pd.read_csv(ANCHOR,dtype={"image_id":str});pc=pd.read_csv(PC,dtype={"image_id":str});pc.columns=[c.replace(".","_") for c in pc.columns]
pc_by={(str(r.image_id),int(r.candidate)):r for r in pc.itertuples(index=False)}
log("SELECTION_LOCK_WRITTEN",lock=LOCK)

# %% [markdown]
# ## Orientation-normalized physical features

# %%
def moments(x):
 x=np.asarray(x,np.float64).ravel();m=x.mean();s=x.std()+1e-6;z=(x-m)/s
 return [m,s,float(np.mean(z**3)),float(np.mean(z**4))]
def entropy_mod(raw,mod):
 if not np.issubdtype(raw.dtype,np.integer):return 0.
 h=np.bincount((raw.astype(np.int64).ravel()%mod),minlength=mod).astype(float);h/=max(h.sum(),1);h=h[h>0]
 return float(-(h*np.log(h)).sum()/math.log(mod))
def crop_feature(raw,box):
 h0,w0=raw.shape;x1,y1,x2,y2=map(float,box);pad=max(5,int(.12*max(x2-x1,y2-y1)))
 l=max(0,int(math.floor(x1))-pad);r=min(w0,int(math.ceil(x2))+pad);t=max(0,int(math.floor(y1))-pad);b=min(h0,int(math.ceil(y2))+pad)
 c=raw[t:b,l:r];cf=c.astype(np.float32)
 if min(c.shape)<3:return np.zeros(32,np.float32)
 border=np.r_[cf[0],cf[-1],cf[:,0],cf[:,-1]];med=float(np.median(border));mad=1.4826*float(np.median(np.abs(border-med)))+1e-3
 z=np.clip((cf-med)/mad,0,30);mask=z>2.5;yy,xx=np.indices(z.shape);weight=np.maximum(z-1.5,0);sw=float(weight.sum())+1e-6
 mx=float((xx*weight).sum()/sw);my=float((yy*weight).sum()/sw);dx=xx-mx;dy=yy-my
 cov=np.asarray([[(weight*dx*dx).sum()/sw,(weight*dx*dy).sum()/sw],[(weight*dx*dy).sum()/sw,(weight*dy*dy).sum()/sw]])
 val,vec=np.linalg.eigh(cov);direction=vec[:,int(np.argmax(val))];angle=math.degrees(math.atan2(direction[1],direction[0]))
 M=cv2.getRotationMatrix2D((mx,my),angle,1.0);rz=cv2.warpAffine(z,M,(z.shape[1],z.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
 active=rz>2.5;cols=np.where(active.any(0))[0];rows=np.where(active.any(1))[0]
 if not len(cols) or not len(rows):return np.zeros(32,np.float32)
 rs=rz[rows[0]:rows[-1]+1,cols[0]:cols[-1]+1];long=rs.sum(0);trans=rs.sum(1);ln=long/(long.mean()+1e-6);tn=trans/(trans.max()+1e-6)
 q=max(1,len(long)//8);endpoint=float((long[:q].mean()+long[-q:].mean())/(2*long.mean()+1e-6));center=float(long[len(long)//3:2*len(long)//3].mean()/(long.mean()+1e-6))
 fft=np.abs(np.fft.rfft(long-long.mean()));period=float(fft[1:].max()/(fft[1:].sum()+1e-6)) if len(fft)>1 else 0.
 ac=np.correlate(long-long.mean(),long-long.mean(),mode="full")[len(long)-1:];ac=ac/(ac[0]+1e-6);acpeak=float(ac[2:min(len(ac),len(long)//2+1)].max()) if len(ac)>4 else 0.
 widths=np.sum(rs>2.5,0).astype(float);nz=widths>0;width_mean=float(widths[nz].mean()) if nz.any() else 0.;width_cv=float(widths[nz].std()/(width_mean+1e-6)) if nz.any() else 0.
 mirror=np.interp(np.linspace(0,len(tn)-1,len(tn)),np.arange(len(tn)),tn[::-1]);sym=float(np.mean(np.abs(tn-mirror)))
 gy=np.linspace(-1,1,len(tn));good=tn>.05
 if good.sum()>=3:
  coef=np.polyfit(gy[good]**2,np.log(tn[good]+1e-4),1);fit=np.exp(np.polyval(coef,gy**2));fit/=fit.max()+1e-6;psf=float(np.mean(np.abs(tn-fit)))
 else:psf=1.
 side=np.r_[z[:max(1,rows[0])].ravel(),z[min(z.shape[0],rows[-1]+1):].ravel()];inside=rs[rs>0]
 unique=float(len(np.unique(c))/max(c.size,1));diff=np.diff(np.sort(np.unique(c.astype(np.int64)))) if np.issubdtype(c.dtype,np.integer) else np.asarray([])
 spacing=float(np.median(diff)) if len(diff) else 0.
 feats=[*moments(inside),float(mask.mean()),float(weight.sum()/max(z.size,1)),float(val.max()/(val.min()+1e-3)),float(np.std(ln)),float(np.mean(np.abs(np.diff(ln)))) if len(ln)>1 else 0.,endpoint,center,period,acpeak,float(np.mean(long<.25*long.mean())),width_mean,width_cv,sym,psf,*moments(side if len(side) else border),float(np.mean(side>2.5)) if len(side) else 0.,unique,math.log1p(spacing),entropy_mod(c,2),entropy_mod(c,4),entropy_mod(c,16),float((x1%1+y1%1+x2%1+y2%1)/4),math.log1p(max(x2-x1,y2-y1)),math.log1p(min(x2-x1,y2-y1)),math.log1p((x2-x1)/(y2-y1+1e-3))]
 out=np.nan_to_num(np.asarray(feats,np.float32),nan=0,posinf=20,neginf=-20);assert len(out)==32,len(out);return np.clip(out,-20,20)

with (UNLEARN/"annotations_coco.json").open(encoding="utf-8") as f:coco=json.load(f)
names={int(x["id"]):x["file_name"] for x in coco["images"]};poison=[]
for a in coco["annotations"]:
 x,y,w,h=map(float,a["bbox"]);poison.append(crop_feature(load_raw(UNLEARN/names[int(a["image_id"])]),[x,y,x+w,y+h]))

external=[]
for split in ("valid","val","test","train"):
 imd,lad=EXT/split/"images",EXT/split/"labels"
 if not imd.is_dir():continue
 for p in sorted(imd.glob("*")):
  lab=lad/f"{p.stem}.txt"
  if not lab.exists():continue
  raw=load_raw(p);h,w=raw.shape
  for line in lab.read_text(encoding="utf-8").splitlines():
   q=line.split()
   if len(q)<5:continue
   _,xc,yc,bw,bh=map(float,q[:5]);external.append(crop_feature(raw,[(xc-bw/2)*w,(yc-bh/2)*h,(xc+bw/2)*w,(yc+bh/2)*h]))
   if len(external)>=80:break
  if len(external)>=80:break
 if len(external)>=80:break

rng=np.random.default_rng(SEED);synthetic=[]
public_raw=[load_raw(p) for p in sorted(UNLEARN.glob("*.png"))]
for i in range(80):
 base=public_raw[i%len(public_raw)].copy();h,w=base.shape;cx=int(rng.integers(150,w-150));cy=int(rng.integers(80,h-80));length=int(rng.integers(45,280));angle=float(rng.uniform(-math.pi,math.pi));width=int(rng.integers(1,4))
 dx=math.cos(angle)*length/2;dy=math.sin(angle)*length/2;signal=np.zeros((h,w),np.float32);cv2.line(signal,(int(cx-dx),int(cy-dy)),(int(cx+dx),int(cy+dy)),1.,width,cv2.LINE_AA);signal=cv2.GaussianBlur(signal,(0,0),float(rng.uniform(.7,1.8)))
 f=base.astype(np.float32);med=np.median(f);mad=1.4826*np.median(np.abs(f-med))+1;f=np.clip(f+signal*float(rng.uniform(5,18))*mad,0,np.iinfo(base.dtype).max).astype(base.dtype)
 synthetic.append(crop_feature(f,[cx-abs(dx)-8,cy-abs(dy)-8,cx+abs(dx)+8,cy+abs(dy)+8]))

# %% [markdown]
# ## Bidirectional public-only gate and fixed-bank inference

# %%
def fit(x,y,steps=1400,lr=.06,l2=.08):
 mu=x.mean(0);sd=x.std(0)+1e-3;z=np.clip((x-mu)/sd,-8,8);w=np.zeros(z.shape[1]);b=0.;y=y.astype(float);weights=np.where(y==1,float(np.sum(y==0)/max(np.sum(y==1),1)),1.)
 for k in range(steps):
  p=1/(1+np.exp(-np.clip(z@w+b,-25,25)));e=(p-y)*weights/weights.mean();rate=lr/(1+.0015*k);w-=rate*((z.T@e)/len(z)+l2*w);b-=rate*e.mean()
 return {"mu":mu,"sd":sd,"w":w,"b":b}
def pred(m,x):
 q=np.clip((x-m["mu"])/m["sd"],-8,8);return (1/(1+np.exp(-np.clip(q@m["w"]+m["b"],-25,25)))).astype(np.float32)
def auc(y,s):
 p=s[y==1];n=s[y==0];return float((p[:,None]>n).mean()+.5*(p[:,None]==n).mean())
poison=np.asarray(poison,np.float32);external=np.asarray(external,np.float32);synthetic=np.asarray(synthetic,np.float32)
assert len(poison)>=16 and len(external)>=60 and len(synthetic)>=60
p_tr,p_va=poison[:15],poison[15:];e_tr,e_va=external[:50],external[50:];s_tr,s_va=synthetic[:50],synthetic[50:]
me=fit(np.r_[p_tr,e_tr],np.r_[np.ones(len(p_tr)),np.zeros(len(e_tr))]);ms=fit(np.r_[p_tr,s_tr],np.r_[np.ones(len(p_tr)),np.zeros(len(s_tr))])
auc_e=auc(np.r_[np.ones(len(p_va)),np.zeros(len(s_va))],pred(me,np.r_[p_va,s_va]));auc_s=auc(np.r_[np.ones(len(p_va)),np.zeros(len(e_va))],pred(ms,np.r_[p_va,e_va]))
models=[];oof=np.zeros(len(poison),np.float32)
for fold in range(5):
 va=np.arange(len(poison))[np.arange(len(poison))%5==fold];tr=np.setdiff1d(np.arange(len(poison)),va);m=fit(np.r_[poison[tr],e_tr,s_tr],np.r_[np.ones(len(tr)),np.zeros(len(e_tr)+len(s_tr))]);models.append(m);oof[va]=pred(m,poison[va])
clean=np.mean([pred(m,np.r_[e_va,s_va]) for m in models],0);strict=float(min(1.,clean.max()+1e-5));relaxed=float(np.quantile(clean,.98));margin=float(oof.mean()-clean.mean());recall=float(np.mean(oof>=strict));enabled=bool(min(auc_e,auc_s)>=.72 and margin>=.08 and recall>=.10)
gate={"poison":len(poison),"external":len(external),"synthetic":len(synthetic),"external_to_synthetic_auc":auc_e,"synthetic_to_external_auc":auc_s,"margin":margin,"strict_threshold":strict,"relaxed_threshold":relaxed,"strict_recall":recall,"enabled":enabled,"test_data_used":False}
(OUT/"renderer_gate.json").write_text(json.dumps(gate,indent=2),encoding="utf-8");np.savez_compressed(OUT/"renderer_ranker.npz",**{f"m{i}_{k}":v for i,m in enumerate(models) for k,v in m.items()},strict=strict,relaxed=relaxed);log("PUBLIC_GATE",**gate)

test_paths={p.stem:p for p in TEST.glob("*.png")};render={k:[] for k in VARIANTS};audit={k:{"changed":0,"removed_mass":0.} for k in VARIANTS};diag=[]
for row in tqdm(anchor.itertuples(index=False),total=len(anchor),desc="V22 renderer"):
 a=parse(row.prediction_string);raw=load_raw(test_paths[str(row.image_id)]);updated={k:a[:,0].copy() for k in VARIANTS}
 for i in range(len(a)):
  score,x,y,w,h=map(float,a[i]);prob=0.
  if score>=.21-1e-6:
   f=crop_feature(raw,[x,y,x+w,y+h])[None];prob=float(np.mean([pred(m,f)[0] for m in models]))
  pcr=pc_by[(str(row.image_id),i)];pcs=np.asarray([pcr.poison_beta_0_25,pcr.poison_beta_0_5,pcr.poison_beta_1,pcr.poison_beta_2],np.float32)
  diag.append({"image_id":str(row.image_id),"candidate":i,"base":score,"renderer_probability":prob,"pcgrad_median":float(np.median(pcs)),"pcgrad_max":float(pcs.max())})
  for name,spec in VARIANTS.items():
   if spec.get("mode")=="identity" or not enabled or score<.21-1e-6:continue
   threshold=strict if spec["threshold"]=="strict" else relaxed;votes=int(np.sum(pcs>=spec["pc_at"]))
   if prob>=threshold and votes>=spec["pc_votes"]:updated[name][i]=min(updated[name][i],spec.get("cap",spec["floor"]))
 for name in VARIANTS:
  audit[name]["changed"]+=int(np.sum(updated[name]<a[:,0]-1e-7));audit[name]["removed_mass"]+=float(np.sum(a[:,0]-updated[name]));b=a.copy();b[:,0]=updated[name];render[name].append(fmt(b))

for name,preds in render.items():
 p=OUT/f"submission_{name}.csv"
 if name=="V22_0_exact_v15b":p.write_bytes(ANCHOR.read_bytes())
 else:f=anchor.copy();f["prediction_string"]=preds;f.to_csv(p,index=False)
 f=pd.read_csv(p,dtype={"image_id":str},keep_default_na=False);audit[name].update({"sha256":sha(p),"rows":len(f),"unique_ids":int(f.image_id.nunique()),"boxes":3995,"boxes_added":0,"boxes_moved":0,"confidence_increases":0})
pd.DataFrame(diag).to_csv(OUT/"per_box_diagnostics.csv",index=False)
report={"status":"complete","anchor_exact":audit["V22_0_exact_v15b"]["sha256"]==EXPECTED,"gate":gate,"variants":audit,"rule_7a_guard_passed":True,"test_used_for_training_or_selection":False,"competition_submission_created":False}
(OUT/"final_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");log("COMPLETE",report=report)
