"""Kaggle VALIDATION notebook (no submission): leave-one-well-out CV of the
offset-well structural-surface prior  S(X,Y)=TVT+Z.

Hidden wells are 'very close to existing wells', so a target well's hidden TVT
should be predictable from NEIGHBOR wells' S surface, calibrated on the visible
prefix. Reports pooled by-well CV (the honest metric; flat baseline ~15.9) for
several neighbor schemes. Pure spatial + interp, no PF -> fast even on 5 CPUs.
"""
import json, os

MD = """# ROGII — Offset-Well Structural-Surface Prior (CV only, no submission)

Leave-one-well-out CV of `TVT = S(X,Y) - Z + prefix_bias`, where the structural
surface `S = TVT+Z` is interpolated from NEIGHBOR wells' points (target well
excluded). Honest by-well pooled RMSE (flat ~15.9). Tests nearest-well, KNN, and
plane-fit neighbor schemes. If this beats PF (~10.6), it's the lever from 10 → 6.5."""

CODE = r'''import os, glob, numpy as np, pandas as pd
from scipy.spatial import cKDTree

def find_root():
    for r in ["/kaggle/input/rogii-wellbore-geology-prediction",
              "/kaggle/input/competitions/rogii-wellbore-geology-prediction","data"]+sorted(glob.glob("/kaggle/input/*")):
        if glob.glob(os.path.join(r,"train","*__horizontal_well.csv")): return r
    hits=glob.glob("/kaggle/input/**/train/*__horizontal_well.csv",recursive=True)
    if hits: return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("data")
DATA=find_root(); print("DATA:",DATA)

wells=sorted({os.path.basename(f).split("__")[0] for f in glob.glob(f"{DATA}/train/*__horizontal_well.csv")})
print("train wells",len(wells))

# load each well: X,Y,Z,MD,TVT,TVT_input; keep post-PS as eval, pre-PS as prefix
W={}
for w in wells:
    try: h=pd.read_csv(f"{DATA}/train/{w}__horizontal_well.csv",usecols=["MD","X","Y","Z","TVT","TVT_input"])
    except Exception: continue
    ps=int(h["TVT_input"].notna().sum())
    if ps<60 or ps>=len(h)-5: continue
    W[w]=dict(X=h["X"].to_numpy(float),Y=h["Y"].to_numpy(float),Z=h["Z"].to_numpy(float),
              TVT=h["TVT"].to_numpy(float),ps=ps)
print("usable wells",len(W))

# subsampled point cloud of the structural surface S=TVT+Z, tagged by well
STRIDE=15
pc_x=[]; pc_y=[]; pc_s=[]; pc_w=[]
for w,d in W.items():
    sl=slice(0,len(d["X"]),STRIDE)
    pc_x.append(d["X"][sl]); pc_y.append(d["Y"][sl])
    pc_s.append(d["TVT"][sl]+d["Z"][sl]); pc_w.append(np.full(len(d["X"][sl]),w))
PX=np.concatenate(pc_x); PY=np.concatenate(pc_y); PS=np.concatenate(pc_s); PW=np.concatenate(pc_w)
sc=np.array([PX.std(),PY.std()]); sc=np.where(sc<1e-6,1,sc)
tree=cKDTree(np.column_stack([PX/sc[0],PY/sc[1]]))
print("point cloud",len(PX))

def predict(w,K,FETCH=150):
    d=W[w]; ps=d["ps"]; n=len(d["X"])
    q=np.column_stack([d["X"]/sc[0],d["Y"]/sc[1]])
    dist,idx=tree.query(q,k=min(FETCH,len(PX)),workers=-1)
    dist=np.where(PW[idx]==w,np.inf,dist)          # drop self well
    order=np.argsort(dist,axis=1)[:,:K]
    dk=np.take_along_axis(dist,order,1); ik=np.take_along_axis(idx,order,1)
    valid=np.isfinite(dk); wt=np.where(valid,1.0/(dk+1.0),0.0)
    wsum=wt.sum(1); safe=np.where(wsum>0,wsum,1.0)
    S_hat=np.where(wsum>0,(PS[ik]*wt).sum(1)/safe,np.nan)
    tvt_hat=S_hat-d["Z"]
    kn=slice(0,ps); ev=slice(ps,n)
    bias=np.nanmedian(d["TVT"][kn]-tvt_hat[kn]); bias=bias if np.isfinite(bias) else 0.0
    pred=tvt_hat[ev]+bias
    pred=np.where(np.isfinite(pred),pred,d["TVT"][ps-1])   # fallback -> const
    return d["TVT"][ev], pred, np.full(n-ps,d["TVT"][ps-1])

def pooled(pairs,key):
    e=[(y-(p if key=="ow" else c))**2 for y,p,c in pairs]
    return float(np.sqrt(np.mean(np.concatenate(e))))

for K in [1,3,5,8,15]:
    pairs=[predict(w,K) for w in W]
    print(f"K={K:2d}  const {pooled(pairs,'const'):.3f}   offset-well-S {pooled(pairs,'ow'):.3f}",flush=True)
print("DONE")'''

def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
nb={"cells":[md(MD),code(CODE)],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
os.makedirs("kernels/offsetwell_cv",exist_ok=True)
json.dump(nb,open("kernels/offsetwell_cv/rogii-offsetwell-surface-cv.ipynb","w"),indent=1)
meta={"id":"boltuzamaki/rogii-offsetwell-surface-cv","title":"ROGII Offset-Well Surface CV",
      "code_file":"rogii-offsetwell-surface-cv.ipynb","language":"python","kernel_type":"notebook",
      "is_private":True,"enable_gpu":False,"enable_internet":False,"dataset_sources":[],
      "competition_sources":["rogii-wellbore-geology-prediction"],"kernel_sources":[]}
json.dump(meta,open("kernels/offsetwell_cv/kernel-metadata.json","w"),indent=2)
print("wrote kernels/offsetwell_cv/")
