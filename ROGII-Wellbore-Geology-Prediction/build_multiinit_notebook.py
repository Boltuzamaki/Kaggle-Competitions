import json, os

CODE = r'''import os, glob, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

def find_root():
    for r in ["/kaggle/input/rogii-wellbore-geology-prediction",
              "/kaggle/input/competitions/rogii-wellbore-geology-prediction","data"]+sorted(glob.glob("/kaggle/input/*")):
        if glob.glob(os.path.join(r,"train","*__horizontal_well.csv")): return r
    raise FileNotFoundError("data")
DATA=find_root(); print("DATA:",DATA,flush=True)
MOM=0.998
def pf_mean(md,z,gr,tt,tg,ls,ir,gs,N,NS,seed0):
    acc=np.zeros(len(md))
    for s in range(NS):
        rng=np.random.default_rng(seed0+s)
        pos=ls+2.0*rng.standard_normal(N); rate=ir+0.01*rng.standard_normal(N); w=np.ones(N)/N
        lo,hi=tt[0]-70,tt[-1]+70; pm=md[0]-1
        for i in range(len(md)):
            dm=max(md[i]-pm,1.0); rate=MOM*rate+0.002*rng.standard_normal(N); pos=pos+rate*dm+0.005*rng.standard_normal(N)
            tvt=np.clip(pos-z[i],lo,hi); pos=tvt+z[i]
            if np.isfinite(gr[i]):
                d=(gr[i]-np.interp(tvt,tt,tg))/gs; lk=np.maximum(np.exp(-0.5*np.minimum(d*d,600)),1e-300)
                w*=lk; sm=w.sum(); w=w/sm if sm>0 else np.ones(N)/N
            if 1.0/(w*w).sum()<0.5*N:
                cum=np.cumsum(w); u0=rng.uniform(0,1.0/N); idx=np.clip(np.searchsorted(cum,u0+np.arange(N)/N),0,N-1)
                pos=pos[idx]+0.1*rng.standard_normal(N); rate=rate[idx]+0.001*rng.standard_normal(N); w=np.ones(N)/N
            acc[i]+=float(np.dot(w,pos-z[i])); pm=md[i]
    return acc/NS

INITS=[-30.0,-12.0,0.0,12.0,30.0]     # structural-position init offsets -> distinct coherent modes
def build(w,NS=8,N=250):
    h=pd.read_csv(f"{DATA}/train/{w}__horizontal_well.csv",usecols=["MD","Z","GR","TVT","TVT_input"])
    tw=pd.read_csv(f"{DATA}/train/{w}__typewell.csv").dropna(subset=["TVT","GR"]).sort_values("TVT")
    ps=int(h["TVT_input"].notna().sum())
    if ps<60 or ps>=len(h)-3 or len(tw)<10: return None
    tt=tw["TVT"].to_numpy(float); tg=tw["GR"].to_numpy(float)
    kn=h.iloc[:ps]; tps=float(h["TVT"].iloc[ps-1]); zps=float(kn["Z"].iloc[-1]); mdps=float(kn["MD"].iloc[-1])
    at=np.interp(kn["TVT_input"].to_numpy(),tt,tg); gs=float(np.clip(np.nanstd(kn["GR"].fillna(0).to_numpy()-at),10.,60.))
    tail=kn.tail(30); dt=np.diff(tail["TVT_input"].to_numpy()); dz=np.diff(tail["Z"].to_numpy()); dmv=np.diff(tail["MD"].to_numpy()); m=dmv>0
    ir=float(np.median((dt+dz)[m]/dmv[m])) if m.sum()>=3 else 0.
    gr=h["GR"].interpolate(limit_direction="both").fillna(tg.mean()).to_numpy()
    z=h["Z"].to_numpy(float); md=h["MD"].to_numpy(float); idx=np.arange(ps,len(h)); nev=len(idx)
    mdp=np.concatenate([[mdps],md[idx]]); zp=np.concatenate([[zps],z[idx]]); grp=np.concatenate([[np.nan],gr[idx]])
    def twg(v): return np.interp(v,tt,tg)
    ge=gr[idx]
    paths=[]
    for j,off in enumerate(INITS):
        p=pf_mean(mdp,zp,grp,tt,tg,tps+off+zps,ir,gs,N,NS,2000+100*j)[1:]
        paths.append(p)
    paths=np.stack(paths,0)                          # (5,nev) coherent candidate paths
    kmd=kn["MD"].to_numpy(float); ktvt=kn["TVT_input"].to_numpy(float)
    sl=np.polyfit(kmd[-100:],ktvt[-100:],1)[0] if ps>=100 else 0.0; sl=float(np.clip(sl,-0.05,0.05))
    trend=tps+sl*(md[idx]-mdps)
    f={"well":w}
    grm=np.abs(ge[None,:]-twg(paths))               # (5,nev) GR residual per path
    for j in range(len(INITS)):
        f[f"d_p{j}"]=(paths[j]-tps).astype(np.float32)
        f[f"grm_p{j}"]=grm[j].astype(np.float32)
        f[f"trd_p{j}"]=(paths[j]-trend).astype(np.float32)   # trend consistency
    # GR-preferred coherent path (argmin windowed GR residual)
    grm_sm=np.stack([pd.Series(grm[j]).rolling(31,center=True,min_periods=1).mean().to_numpy() for j in range(len(INITS))],0)
    bestj=np.argmin(grm_sm,0); grpref=paths[bestj,np.arange(nev)]
    f["d_grpref"]=(grpref-tps).astype(np.float32); f["grm_best"]=grm_sm.min(0).astype(np.float32)
    f["path_spread"]=(paths.max(0)-paths.min(0)).astype(np.float32)
    f["d_mean"]=(paths.mean(0)-tps).astype(np.float32); f["d_median"]=(np.median(paths,0)-tps).astype(np.float32)
    f["d_md"]=(md[idx]-mdps).astype(np.float32); f["d_z"]=(z[idx]-zps).astype(np.float32)
    f["frac"]=(np.arange(nev)/max(nev-1,1)).astype(np.float32); f["gr"]=ge.astype(np.float32)
    f["prefix_slope"]=np.float32(sl); f["gs"]=np.float32(gs)
    f["target"]=(h["TVT"].to_numpy(float)[idx]-tps).astype(np.float32)
    return pd.DataFrame(f)

wells=[os.path.basename(f).split("__")[0] for f in sorted(glob.glob(f"{DATA}/train/*__horizontal_well.csv"))]
print("wells",len(wells),flush=True); t0=time.time()
parts=[p for p in Parallel(n_jobs=4,verbose=5)(delayed(build)(w) for w in wells) if p is not None]
D=pd.concat(parts,ignore_index=True); D.to_pickle("multiinit_feats.pkl")
print("rows",len(D),"%.0fs"%(time.time()-t0),flush=True)
DROP={"well","target"}; FEATS=[c for c in D.columns if c not in DROP]
X=D[FEATS].to_numpy(np.float32); yv=D["target"].to_numpy(np.float32); g=D["well"].to_numpy()
def rr(t,p): return float(np.sqrt(np.mean((t-p)**2)))
gkf=GroupKFold(5); oof=np.zeros(len(yv))
Pm=dict(objective="regression",n_estimators=1500,learning_rate=0.02,num_leaves=127,min_child_samples=80,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=5.0,reg_alpha=1.0,verbose=-1,n_jobs=-1)
for k,(tr,va) in enumerate(gkf.split(X,yv,g)):
    oof[va]=lgb.LGBMRegressor(**Pm).fit(X[tr],yv[tr]).predict(X[va]); print("fold",k,"%.0fs"%(time.time()-t0),flush=True)
print("const pooled %.3f"%rr(yv,0),flush=True)
print("PF-mean pooled %.3f"%rr(yv,D["d_mean"].to_numpy()),flush=True)
print("GR-pref pooled %.3f"%rr(yv,D["d_grpref"].to_numpy()),flush=True)
print("MULTIINIT-GBM pooled %.3f"%rr(yv,oof),flush=True)
imp=pd.Series(lgb.LGBMRegressor(**Pm).fit(X,yv).feature_importances_,index=FEATS).sort_values(ascending=False)
print("top15:"); print(imp.head(15).to_string())
print("DONE",flush=True)
'''
nb={"cells":[{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":CODE.splitlines(keepends=True)}],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
    "nbformat":4,"nbformat_minor":5}
os.makedirs("kernels/multiinit_cv",exist_ok=True)
json.dump(nb,open("kernels/multiinit_cv/rogii-multiinit-cv.ipynb","w"),indent=1)
meta={"id":"boltuzamaki/rogii-multiinit-cv","title":"ROGII MultiInit CV",
      "code_file":"rogii-multiinit-cv.ipynb","language":"python","kernel_type":"notebook",
      "is_private":True,"enable_gpu":False,"enable_internet":False,
      "dataset_sources":["boltuzamaki/rogii-cv-harness"],
      "competition_sources":["rogii-wellbore-geology-prediction"],"kernel_sources":[]}
json.dump(meta,open("kernels/multiinit_cv/kernel-metadata.json","w"),indent=2)
compile(CODE,"<nb>","exec")
print("built multiinit, syntax OK")
