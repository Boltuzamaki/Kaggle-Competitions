"""Kaggle CV notebook (no submission): offset-well DIP-FIELD prior.

Raw S=TVT+Z interpolation fails (~96) because wells reference different typewell
datums -> absolute levels don't transfer. Datum-free alternative: each neighbor
point carries a directional derivative dS/ds along its own path; fit the local
gradient field g=(dS/dx,dS/dy) from neighbors (target well excluded), integrate
g along the target's XY path from the PS anchor, add -Z.

Leave-one-well-out over all 773 wells, pooled suffix RMSE vs const (~15.9).
"""
import json, os

MD = """# ROGII — Offset-Well Dip-Field Prior (CV only)

Datum-free spatial prior: interpolate the **gradient** of the structural surface
S=TVT+Z from neighbor wells (directional derivatives along their paths), integrate
along the target path from the PS anchor. Tests the host-slide claim that dips
behave similarly in neighboring wells. LOWO CV over 773 wells, pooled suffix RMSE."""

CODE = r'''import os, glob, time
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

def find_root():
    for r in ["/kaggle/input/rogii-wellbore-geology-prediction",
              "/kaggle/input/competitions/rogii-wellbore-geology-prediction","data"]+sorted(glob.glob("/kaggle/input/*")):
        if glob.glob(os.path.join(r,"train","*__horizontal_well.csv")): return r
    hits=glob.glob("/kaggle/input/**/train/*__horizontal_well.csv",recursive=True)
    if hits: return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("data")
DATA=find_root(); print("DATA:",DATA,flush=True)

wells=sorted({os.path.basename(f).split("__")[0] for f in glob.glob(f"{DATA}/train/*__horizontal_well.csv")})
W={}
for w in wells:
    try: h=pd.read_csv(f"{DATA}/train/{w}__horizontal_well.csv",usecols=["MD","X","Y","Z","TVT","TVT_input"])
    except Exception: continue
    ps=int(h["TVT_input"].notna().sum())
    if ps<60 or ps>=len(h)-5: continue
    W[w]=dict(X=h["X"].to_numpy(float),Y=h["Y"].to_numpy(float),Z=h["Z"].to_numpy(float),
              TVT=h["TVT"].to_numpy(float),ps=ps)
print("usable wells",len(W),flush=True)

# ---- neighbor sample cloud: midpoints with directional derivative of S ----
SEG=25         # segment length in samples for a stable local slope
pxs,pys,gus,uxs,uys,wws=[],[],[],[],[],[]
for w,d in W.items():
    S=d["TVT"]+d["Z"]; X=d["X"]; Y=d["Y"]; n=len(S)
    for i in range(0,n-SEG,SEG):
        dx=X[i+SEG]-X[i]; dy=Y[i+SEG]-Y[i]
        ds_xy=np.hypot(dx,dy)
        if ds_xy<5: continue                      # too little lateral movement
        dS=S[i+SEG]-S[i]
        pxs.append((X[i]+X[i+SEG])/2); pys.append((Y[i]+Y[i+SEG])/2)
        gus.append(dS/ds_xy); uxs.append(dx/ds_xy); uys.append(dy/ds_xy); wws.append(w)
PX=np.array(pxs); PY=np.array(pys); GU=np.array(gus)
UX=np.array(uxs); UY=np.array(uys); PW=np.array(wws)
sc=np.array([PX.std(),PY.std()]); sc=np.where(sc<1e-6,1,sc)
tree=cKDTree(np.column_stack([PX/sc[0],PY/sc[1]]))
print("dip samples",len(PX),flush=True)

def predict(w,K=40,RIDGE=1e-4,CLIP=0.05):
    d=W[w]; ps=d["ps"]; n=len(d["X"])
    X=d["X"]; Y=d["Y"]; Z=d["Z"]; TVT=d["TVT"]
    # query gradient at strided points along the whole well
    ST=20
    qi=np.arange(0,n,ST)
    q=np.column_stack([X[qi]/sc[0],Y[qi]/sc[1]])
    dist,idx=tree.query(q,k=min(K+60,len(PX)),workers=-1)
    dist=np.where(PW[idx]==w,np.inf,dist)
    order=np.argsort(dist,axis=1)[:,:K]
    dk=np.take_along_axis(dist,order,1); ik=np.take_along_axis(idx,order,1)
    gx=np.zeros(len(qi)); gy=np.zeros(len(qi))
    for j in range(len(qi)):
        v=np.isfinite(dk[j])
        if v.sum()<6: continue
        ii=ik[j][v]; wt=1.0/(dk[j][v]+0.5)
        A=np.column_stack([UX[ii],UY[ii]])*wt[:,None]
        b=GU[ii]*wt
        M=A.T@A+RIDGE*np.eye(2)
        try: g=np.linalg.solve(M,A.T@b)
        except Exception: continue
        gx[j],gy[j]=np.clip(g,-CLIP,CLIP)
    # interpolate gradient to every sample, integrate along path from PS
    gx_f=np.interp(np.arange(n),qi,gx); gy_f=np.interp(np.arange(n),qi,gy)
    dX=np.diff(X,prepend=X[0]); dY=np.diff(Y,prepend=Y[0])
    dS=gx_f*dX+gy_f*dY
    S_int=np.cumsum(dS)
    S_ps=TVT[ps-1]+Z[ps-1]
    S_hat=S_ps+(S_int-S_int[ps-1])
    pred=S_hat-Z
    y=TVT[ps:]; return y,pred[ps:],np.full(n-ps,TVT[ps-1])

def pooled(pairs,key):
    e=[(y-(p if key=="dip" else c))**2 for y,p,c in pairs]
    return float(np.sqrt(np.mean(np.concatenate(e))))

t0=time.time()
for K in [15,40,80]:
    pairs=[predict(w,K=K) for w in W]
    print(f"K={K:3d}  const {pooled(pairs,'const'):.3f}   dip-field {pooled(pairs,'dip'):.3f}   ({time.time()-t0:.0f}s)",flush=True)
print("DONE")'''

def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
nb={"cells":[md(MD),code(CODE)],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
os.makedirs("kernels/dipfield_cv",exist_ok=True)
json.dump(nb,open("kernels/dipfield_cv/rogii-dipfield-cv.ipynb","w"),indent=1)
meta={"id":"boltuzamaki/rogii-dipfield-cv","title":"ROGII Dipfield CV",
      "code_file":"rogii-dipfield-cv.ipynb","language":"python","kernel_type":"notebook",
      "is_private":True,"enable_gpu":False,"enable_internet":False,"dataset_sources":[],
      "competition_sources":["rogii-wellbore-geology-prediction"],"kernel_sources":[]}
json.dump(meta,open("kernels/dipfield_cv/kernel-metadata.json","w"),indent=2)
print("wrote kernels/dipfield_cv/")
