"""General stack v2 TRAINING notebook (Kaggle, CV only, no submission).

Closes the formulation gap vs the public general branch: our physics signals
(PF/beam/NCC/tortuosity/twres) + spatial formation-plane & dense-surface FEATURES
(weak standalone, informative inside a GBM) + GR-alignment anchor-offset features
+ prefix-trend + PF seed-spread/bimodality. LGBM 5-fold GroupKFold on
dTVT = TVT - last_known. Reports pooled CV (const ~15.9; current stack 10.6).
Saves models + feature list to /kaggle/working for a later inference notebook.
"""
import json, os

PF = open("exp/pf_tracker.py", encoding="utf-8").read()
BEAM = open("exp/beam_tracker.py", encoding="utf-8").read().replace("cache=True", "cache=False")

def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}

MD = """# ROGII — General Stack v2 (training + CV, no submission)

Physics signals + spatial-imputer features + alignment offsets → LGBM GroupKFold.
Baselines: const 15.9 · PF ~10.8 · stack v1 10.62. Goal: CV ≤ 9."""

FEAT = r'''import os, glob, time, math
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

def find_root():
    for r in ["/kaggle/input/rogii-wellbore-geology-prediction",
              "/kaggle/input/competitions/rogii-wellbore-geology-prediction","data"]+sorted(glob.glob("/kaggle/input/*")):
        if glob.glob(os.path.join(r,"train","*__horizontal_well.csv")): return r
    hits=glob.glob("/kaggle/input/**/train/*__horizontal_well.csv",recursive=True)
    if hits: return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("data")
DATA=find_root(); print("DATA:",DATA,flush=True)
FORMS=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
ANCH_OFF=[-40,-20,-10,-5,0,5,10,20,40]

def ps_index(h): return int(h["TVT_input"].notna().sum())

# ---------- spatial reference models (fit on all train wells; leave-self-out at query) ----------
def build_plane_index():
    rows=[]
    for f in sorted(glob.glob(f"{DATA}/train/*__horizontal_well.csv")):
        w=os.path.basename(f).split("__")[0]
        try: df=pd.read_csv(f,usecols=["X","Y"]+FORMS).dropna()
        except Exception: continue
        if not len(df): continue
        r={"wid":w,"x":float(df["X"].median()),"y":float(df["Y"].median())}
        for c in FORMS: r[c]=float(df[c].median())
        rows.append(r)
    idx=pd.DataFrame(rows)
    xy=idx[["x","y"]].to_numpy(); scl=np.where(xy.std(0)<1e-3,1.,xy.std(0))
    return idx, cKDTree(xy/scl), scl

PIDX,PTREE,PSCL=build_plane_index()
PXA=PIDX["x"].to_numpy(); PYA=PIDX["y"].to_numpy()
PFA=PIDX[FORMS].to_numpy(np.float64); PWID={w:i for i,w in enumerate(PIDX["wid"])}

def impute_forms(xy_q, self_wid, k=10):
    q=xy_q/PSCL; nf=min(k+5,len(PIDX))
    dist,ii=PTREE.query(q,k=nf,workers=-1)
    if self_wid in PWID: dist=np.where(ii==PWID[self_wid],np.inf,dist)
    order=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
    dk=np.take_along_axis(dist,order,1); ik=np.take_along_axis(ii,order,1)
    vk=np.isfinite(dk); wt=np.where(vk,1./(dk+1e-3),0.)
    xn=PXA[ik]; yn=PYA[ik]; fn=PFA[ik]; wx=wt*xn; wy=wt*yn
    n=len(q); A=np.zeros((n,3,3))
    A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
    A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
    A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=wt.sum(1)
    for d in range(3): A[:,d,d]+=1e-9
    rhs=np.stack([(wx[:,:,None]*fn).sum(1),(wy[:,:,None]*fn).sum(1),(wt[:,:,None]*fn).sum(1)],1)
    try: coef=np.linalg.solve(A,rhs)
    except Exception: coef=np.stack([np.linalg.pinv(A[r])@rhs[r] for r in range(n)])
    pred=xy_q[:,0][:,None]*coef[:,0,:]+xy_q[:,1][:,None]*coef[:,1,:]+coef[:,2,:]
    dmin=np.where(vk,dk,np.inf).min(1)
    return pred, dmin

def multiscale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3):
    out=[]
    for hw in hws:
        win=2*hw+1; nk=len(kgr); nh=len(hgr)
        if nk<win+1 or nh==0:
            out.append((np.full(nh,ktvt[-1] if len(ktvt) else 0.,np.float32),np.zeros(nh,np.float32))); continue
        kg=pd.Series(kgr).rolling(5,center=True,min_periods=1).mean().to_numpy(np.float32)
        hg=pd.Series(hgr).rolling(5,center=True,min_periods=1).mean().to_numpy(np.float32)
        sts=np.arange(0,nk-win+1,stride,dtype=np.int32)
        if not len(sts):
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        C=kg[sts[:,None]+np.arange(win,dtype=np.int32)[None,:]]
        Cn=(C-C.mean(1,keepdims=True))/(C.std(1,keepdims=True)+1e-6)
        hp=np.pad(hg,hw,mode="edge")
        H=hp[np.arange(nh)[:,None]+np.arange(win)[None,:]]
        Hn=(H-H.mean(1,keepdims=True))/(H.std(1,keepdims=True)+1e-6)
        ncc=Hn@Cn.T/win; best=ncc.argmax(1); score=ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw,0,nk-1)].astype(np.float32),score))
    return out

def robust_slope(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.std(x[m])<1e-6: return 0.
    return float(np.polyfit(x[m],y[m],1)[0])

def seg_b(ktvt,kz,fcol):
    bv=ktvt+kz-fcol; n=len(bv)
    b_full=float(np.median(bv))
    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full
    return b_full,b_late

def build_well(path,is_train,self_excl):
    wid=os.path.basename(path).split("__")[0]; base=os.path.dirname(path)
    try:
        h=pd.read_csv(path); tw=pd.read_csv(f"{base}/{wid}__typewell.csv")
    except Exception: return None
    ps=ps_index(h)
    if ps<60 or ps>=len(h)-3: return None
    if is_train and "TVT" not in h.columns: return None
    tw_s=tw.sort_values("TVT"); tw_tvt=tw_s["TVT"].to_numpy(float)
    tw_gr=tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    if len(tw_tvt)<3: return None
    tps=float(h["TVT_input"].iloc[ps-1])
    md=h["MD"].to_numpy(float); x=h["X"].to_numpy(float); y=h["Y"].to_numpy(float); z=h["Z"].to_numpy(float)
    gr=h["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
    ev=np.arange(ps,len(h)); nev=len(ev)
    ktvt=h["TVT_input"].to_numpy(float)[:ps]; kz=z[:ps]; kmd=md[:ps]; kgr=gr[:ps]

    # physics candidate paths: two PF ensembles (disagreement = uncertainty) + beam
    pf_a=pf_predict(h,tw,n_particles=300,n_seeds=16,scale=12.0)
    pf_b=pf_predict(h,tw,n_particles=300,n_seeds=16,scale=3.0)
    bm=beam_predict(h,tw)
    ncc=multiscale_ncc(kgr,ktvt,gr[ps:])

    # spatial features
    form_ev,dmin_ev=impute_forms(np.column_stack([x[ev],y[ev]]), wid if self_excl else None)
    form_kn,_=impute_forms(np.column_stack([x[:ps],y[:ps]]), wid if self_excl else None)
    sp={}
    cands=[]
    for fi,fn in enumerate(FORMS):
        bf,bl=seg_b(ktvt,kz,form_kn[:,fi])
        c_full=(-z[ev]+form_ev[:,fi]+bf); c_late=(-z[ev]+form_ev[:,fi]+bl)
        sp[f"spF_{fn}"]=(c_full-tps).astype(np.float32)
        sp[f"spL_{fn}"]=(c_late-tps).astype(np.float32)
        kn_rmse=float(np.sqrt(np.mean((ktvt-(-kz+form_kn[:,fi]+bf))**2)))
        sp[f"spRm_{fn}"]=np.float32(kn_rmse)
        cands.append(c_full)
    cst=np.stack(cands,1)
    sp["sp_mean_d"]=(cst.mean(1)-tps).astype(np.float32)
    sp["sp_std"]=cst.std(1).astype(np.float32)
    sp["sp_dmin"]=dmin_ev.astype(np.float32)

    tw_at=lambda t: np.interp(t,tw_tvt,tw_gr)
    # typewell "supertype" fingerprint (57 unique typewells shared across wells):
    # wells sharing a typewell must stay in the same CV fold (no typewell leakage)
    twh=int(abs(hash((round(float(tw_tvt[0]),1),round(float(tw_tvt[-1]),1),
                      round(float(np.nansum(tw_gr[:200])),0),len(tw_tvt))))%10**9)
    gr_s=pd.Series(gr)
    f={"well":wid,"id":[f"{wid}_{i}" for i in ev],"last_known_tvt":tps,
       "supertype":twh,
       "pf_d":(pf_a[ev]-tps).astype(np.float32),
       "pf3_d":(pf_b[ev]-tps).astype(np.float32),
       "pf_dis":(pf_a[ev]-pf_b[ev]).astype(np.float32),
       "beam_d":(bm[ev]-tps).astype(np.float32),
       "pf_vs_beam":(pf_a[ev]-bm[ev]).astype(np.float32),
       "ncc8_d":np.clip(ncc[0][0]-tps,-80,80),"ncc8_s":ncc[0][1],
       "ncc15_d":np.clip(ncc[1][0]-tps,-80,80),"ncc15_s":ncc[1][1],
       "ncc25_d":np.clip(ncc[2][0]-tps,-80,80),"ncc25_s":ncc[2][1],
       "d_md":(md[ev]-md[ps-1]).astype(np.float32),
       "d_z":(z[ev]-z[ps-1]).astype(np.float32),
       "d_xy":np.hypot(x[ev]-x[ps-1],y[ev]-y[ps-1]).astype(np.float32),
       "dz_dmd":(np.gradient(z)/(np.gradient(md)+1e-9))[ev].astype(np.float32),
       "gr":gr[ev].astype(np.float32),
       "gr_m21":gr_s.rolling(21,center=True,min_periods=1).mean().to_numpy()[ev].astype(np.float32),
       "gr_s21":gr_s.rolling(21,center=True,min_periods=1).std().fillna(0).to_numpy()[ev].astype(np.float32),
       "frac":(np.arange(nev)/max(nev-1,1)).astype(np.float32),
       "slp_all":np.float32(robust_slope(kmd,ktvt)),
       "slp_50":np.float32(robust_slope(kmd[-50:],ktvt[-50:])),
       "ktvt_rng":np.float32(np.ptp(ktvt)),"ktvt_std":np.float32(ktvt.std()),
       "pfx_gr_rmse":np.float32(np.sqrt(np.nanmean((kgr-tw_at(ktvt))**2))),
      }
    for o in ANCH_OFF:
        f[f"tda{o}"]=(gr[ev]-tw_at(tps+o)).astype(np.float32)
        f[f"tdpf{o}"]=(gr[ev]-tw_at(pf_a[ev]+o)).astype(np.float32)
    f.update(sp)
    if is_train:
        f["target"]=(h["TVT"].to_numpy(float)[ev]-tps).astype(np.float32)
    return pd.DataFrame(f)

t0=time.time()
paths=sorted(glob.glob(f"{DATA}/train/*__horizontal_well.csv"))
res=Parallel(n_jobs=4)(delayed(build_well)(p,True,True) for p in paths)
train=pd.concat([r for r in res if r is not None],ignore_index=True)
print("train",train.shape,"wells",train["well"].nunique(),"%.0fs"%(time.time()-t0),flush=True)
train.to_parquet("train_feats.parquet") if hasattr(pd.DataFrame,"to_parquet") else train.to_pickle("train_feats.pkl")'''

TRAIN = r'''import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import joblib

DROP={"well","id","target","last_known_tvt","supertype"}
FEATS=[c for c in train.columns if c not in DROP]
print("features",len(FEATS))
print("unique typewell supertypes:",train["supertype"].nunique())
X=train[FEATS].to_numpy(np.float32); yv=train["target"].to_numpy(np.float32)
g=train["supertype"].to_numpy()      # fold by typewell supertype (57) — no typewell leakage
P=dict(objective="regression",n_estimators=1500,learning_rate=0.02,num_leaves=127,
       min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,
       reg_lambda=5.0,reg_alpha=1.0,verbose=-1,n_jobs=-1)
gkf=GroupKFold(n_splits=5); oof=np.zeros(len(yv)); models=[]
for k,(tr,va) in enumerate(gkf.split(X,yv,g)):
    m=lgb.LGBMRegressor(**P).fit(X[tr],yv[tr])
    oof[va]=m.predict(X[va]); models.append(m); print("fold",k,"done",flush=True)

def rr(t,p): return float(np.sqrt(np.mean((t-p)**2)))
print("const  pooled %.3f"%rr(yv,0))
print("pf     pooled %.3f"%rr(yv,train["pf_d"].to_numpy()))
print("stack2 pooled %.3f"%rr(yv,oof))
best=(1e9,1.0,100)
for sh in [1.0,0.95,0.9,0.85,0.8]:
    for cl in [40,60,100]:
        r=rr(yv,np.clip(oof*sh,-cl,cl))
        if r<best[0]: best=(r,sh,cl)
print("stack2 pp pooled %.3f (shrink=%.2f clip=%d)"%best)
imp=pd.Series(models[0].feature_importances_,index=FEATS).sort_values(ascending=False)
print("top20 features:"); print(imp.head(20).to_string())
np.save("oof.npy",oof); joblib.dump({"models":models,"feats":FEATS,"pp":best[1:]},"stack_v2.joblib")
print("DONE",flush=True)'''

cells=[md(MD),code("import os"),code(PF),code(BEAM),code(FEAT),code(TRAIN)]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
os.makedirs("kernels/stack_v2",exist_ok=True)
json.dump(nb,open("kernels/stack_v2/rogii-general-stack-v2.ipynb","w"),indent=1)
meta={"id":"boltuzamaki/rogii-general-stack-v2","title":"ROGII General Stack V2",
      "code_file":"rogii-general-stack-v2.ipynb","language":"python","kernel_type":"notebook",
      "is_private":True,"enable_gpu":False,"enable_internet":False,"dataset_sources":[],
      "competition_sources":["rogii-wellbore-geology-prediction"],"kernel_sources":[]}
json.dump(meta,open("kernels/stack_v2/kernel-metadata.json","w"),indent=2)
print("wrote kernels/stack_v2/")
