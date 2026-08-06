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
def pf_seeds(md,z,gr,tt,tg,ls,ir,gs,N,NS):
    out=np.empty((NS,len(md)))
    for s in range(NS):
        rng=np.random.default_rng(1000+s)
        pos=ls+2.0*rng.standard_normal(N); rate=ir+0.01*rng.standard_normal(N); w=np.ones(N)/N
        lo,hi=tt[0]-60,tt[-1]+60; pm=md[0]-1
        VN=0.002*rng.uniform(.9,1.1); PN=0.005*rng.uniform(.85,1.15)
        for i in range(len(md)):
            dm=max(md[i]-pm,1.0); rate=MOM*rate+VN*rng.standard_normal(N); pos=pos+rate*dm+PN*rng.standard_normal(N)
            tvt=np.clip(pos-z[i],lo,hi); pos=tvt+z[i]
            if np.isfinite(gr[i]):
                d=(gr[i]-np.interp(tvt,tt,tg))/gs; lk=np.maximum(np.exp(-0.5*np.minimum(d*d,600)),1e-300)
                w*=lk; sm=w.sum(); w=w/sm if sm>0 else np.ones(N)/N
            if 1.0/(w*w).sum()<0.5*N:
                cum=np.cumsum(w); u0=rng.uniform(0,1.0/N); idx=np.clip(np.searchsorted(cum,u0+np.arange(N)/N),0,N-1)
                pos=pos[idx]+0.1*rng.standard_normal(N); rate=rate[idx]+0.001*rng.standard_normal(N); w=np.ones(N)/N
            out[s,i]=float(np.dot(w,pos-z[i])); pm=md[i]
    return out

def build(w,NS=32,N=250):
    h=pd.read_csv(f"{DATA}/train/{w}__horizontal_well.csv",usecols=["MD","X","Y","Z","GR","TVT","TVT_input"])
    tw=pd.read_csv(f"{DATA}/train/{w}__typewell.csv").dropna(subset=["TVT","GR"]).sort_values("TVT")
    ps=int(h["TVT_input"].notna().sum())
    if ps<60 or ps>=len(h)-3 or len(tw)<10: return None
    tt=tw["TVT"].to_numpy(float); tg=tw["GR"].to_numpy(float)
    kn=h.iloc[:ps]; tps=float(h["TVT"].iloc[ps-1]); zps=float(kn["Z"].iloc[-1]); mdps=float(kn["MD"].iloc[-1])
    at=np.interp(kn["TVT_input"].to_numpy(),tt,tg); gs=float(np.clip(np.nanstd(kn["GR"].fillna(0).to_numpy()-at),10.,60.))
    tail=kn.tail(30); dt=np.diff(tail["TVT_input"].to_numpy()); dz=np.diff(tail["Z"].to_numpy()); dmv=np.diff(tail["MD"].to_numpy()); m=dmv>0
    ir=float(np.median((dt+dz)[m]/dmv[m])) if m.sum()>=3 else 0.
    gr=h["GR"].interpolate(limit_direction="both").fillna(tg.mean()).to_numpy()
    z=h["Z"].to_numpy(float); md=h["MD"].to_numpy(float)
    idx=np.arange(ps,len(h))
    mdp=np.concatenate([[mdps],md[idx]]); zp=np.concatenate([[zps],z[idx]]); grp=np.concatenate([[np.nan],gr[idx]])
    P=pf_seeds(mdp,zp,grp,tt,tg,tps+zps,ir,gs,N,NS)[:,1:]
    nev=P.shape[1]; ge=gr[idx]
    def twg(v): return np.interp(v,tt,tg)
    pcts=np.percentile(P,[5,25,50,75,95],axis=0)
    med=np.median(P,0); low=P<med
    lowmean=np.where(low.any(0),(P*low).sum(0)/np.maximum(low.sum(0),1),pcts[2])
    highmean=np.where((~low).any(0),(P*(~low)).sum(0)/np.maximum((~low).sum(0),1),pcts[2])
    gap=highmean-lowmean; lowmass=low.mean(0)
    spread=pcts[4]-pcts[0]; iqr=pcts[3]-pcts[1]; mean=P.mean(0)
    kmd=kn["MD"].to_numpy(float); ktvt=kn["TVT_input"].to_numpy(float)
    sl=np.polyfit(kmd[-100:],ktvt[-100:],1)[0] if ps>=100 else 0.0; sl=float(np.clip(sl,-0.05,0.05))
    trend=tps+sl*(md[idx]-mdps)
    cands={"p05":pcts[0],"p25":pcts[1],"p50":pcts[2],"p75":pcts[3],"p95":pcts[4],
           "lowm":lowmean,"highm":highmean,"mean":mean,"trend":trend}
    f={"well":w,"tps":tps}
    for nm,c in cands.items():
        f[f"d_{nm}"]=(c-tps).astype(np.float32)
        f[f"grm_{nm}"]=np.abs(ge-twg(c)).astype(np.float32)
    modal=np.stack([lowmean,highmean,mean,trend],0)
    grr=np.abs(ge[None,:]-twg(modal))
    best=np.argmin(grr,0); grpref=modal[best,np.arange(nev)]
    f["d_grpref"]=(grpref-tps).astype(np.float32); f["grm_grpref"]=grr.min(0).astype(np.float32)
    f["gap"]=gap.astype(np.float32); f["lowmass"]=lowmass.astype(np.float32); f["spread"]=spread.astype(np.float32); f["iqr"]=iqr.astype(np.float32)
    f["low_vs_trend"]=(lowmean-trend).astype(np.float32); f["high_vs_trend"]=(highmean-trend).astype(np.float32)
    grs=pd.Series(gr)
    f["d_md"]=(md[idx]-mdps).astype(np.float32); f["d_z"]=(z[idx]-zps).astype(np.float32)
    f["frac"]=(np.arange(nev)/max(nev-1,1)).astype(np.float32)
    f["gr"]=ge.astype(np.float32); f["gr_m21"]=grs.rolling(21,center=True,min_periods=1).mean().to_numpy()[idx].astype(np.float32)
    f["prefix_slope"]=np.float32(sl); f["gs"]=np.float32(gs)
    f["target"]=(h["TVT"].to_numpy(float)[idx]-tps).astype(np.float32)
    twh=int(abs(hash((round(float(tt[0]),1),round(float(tt[-1]),1),round(float(np.nansum(tg[:200])),0),len(tt))))%10**9)
    f["sup"]=twh
    return pd.DataFrame(f)

wells=[os.path.basename(f).split("__")[0] for f in sorted(glob.glob(f"{DATA}/train/*__horizontal_well.csv"))]
print("wells",len(wells),flush=True); t0=time.time()
parts=[p for p in Parallel(n_jobs=4,verbose=5)(delayed(build)(w) for w in wells) if p is not None]
D=pd.concat(parts,ignore_index=True); D.to_pickle("topk_feats.pkl")
print("rows",len(D),"feats build %.0fs"%(time.time()-t0),flush=True)
DROP={"well","target","tps","sup"}; FEATS=[c for c in D.columns if c not in DROP]
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
print("TOPK-GBM pooled %.3f"%rr(yv,oof),flush=True)
imp=pd.Series(lgb.LGBMRegressor(**Pm).fit(X,yv).feature_importances_,index=FEATS).sort_values(ascending=False)
print("top15:"); print(imp.head(15).to_string())
print("DONE",flush=True)
'''

nb={"cells":[{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":CODE.splitlines(keepends=True)}],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
    "nbformat":4,"nbformat_minor":5}
os.makedirs("kernels/topk_cv",exist_ok=True)
json.dump(nb,open("kernels/topk_cv/rogii-topk-cv.ipynb","w"),indent=1)
meta={"id":"boltuzamaki/rogii-topk-cv","title":"ROGII TopK CV",
      "code_file":"rogii-topk-cv.ipynb","language":"python","kernel_type":"notebook",
      "is_private":True,"enable_gpu":False,"enable_internet":False,
      "dataset_sources":["boltuzamaki/rogii-cv-harness"],
      "competition_sources":["rogii-wellbore-geology-prediction"],"kernel_sources":[]}
json.dump(meta,open("kernels/topk_cv/kernel-metadata.json","w"),indent=2)
compile(CODE,"<nb>","exec")
print("built topk, syntax OK")
