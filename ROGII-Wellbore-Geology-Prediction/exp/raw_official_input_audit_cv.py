"""Grouped-CV audit of overlooked raw official input families."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
H=ROOT/"exp/results/harshini_cached_xgb"
OUT=ROOT/"exp/results/raw_official_input_audit";OUT.mkdir(parents=True,exist_ok=True)
FORMS=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
d=pd.read_pickle(H/"rows.pkl")
base_features=[c for c in d if c not in ("well","flat","target")]
X0=d[base_features].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
y=d.target.to_numpy(np.float32); groups=d.well.to_numpy()
warm=(1-np.exp(-np.maximum(d.md_since.to_numpy(float),0)/85.)).astype(np.float32)
physics=(warm*d.blend_d.to_numpy(float)).astype(np.float32)
resid=y-physics

cache=OUT/"raw_families.npz"
if cache.exists():
    z=np.load(cache); fam={k:z[k] for k in z.files}
else:
    prefix=[];traj=[];marker=[];typewell=[]
    for wi,(w,q) in enumerate(d.groupby("well",sort=False)):
        h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
        t=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
        nv=int(h.TVT_input.notna().sum()); hs=h.iloc[nv:].reset_index(drop=True)
        if len(hs)!=len(q): raise ValueError((w,len(hs),len(q)))
        n=len(hs); last=h.iloc[nv-1]
        flat=float(last.TVT_input)
        # Absolute prefix anchor and paired-reference position.
        tv=t.TVT.to_numpy(float); tg=t.GR.to_numpy(float)
        geo=t.Geology.astype("string")
        gt=[float(t.loc[geo==f,"TVT"].median()) if np.any(geo==f) else np.nan for f in FORMS]
        prefix.append(np.column_stack([
            np.full(n,flat),np.full(n,float(last.MD)),
            np.full(n,float(last.X)),np.full(n,float(last.Y)),np.full(n,float(last.Z)),
            np.full(n,flat+float(last.Z)),np.full(n,flat-np.nanmin(tv)),
            np.full(n,np.nanmax(tv)-flat)]))
        # Raw complete-trajectory coordinates overlooked by row-local frame.
        md=hs.MD.to_numpy(float); x=hs.X.to_numpy(float); yy=hs.Y.to_numpy(float); zc=hs.Z.to_numpy(float)
        dx=np.gradient(x);dy=np.gradient(yy);dz=np.gradient(zc)
        traj.append(np.column_stack([md,md-last.MD,x-last.X,yy-last.Y,zc-last.Z,
            hs.X.iloc[-1]-x,hs.Y.iloc[-1]-yy,hs.Z.iloc[-1]-zc,
            dx,dy,dz,np.hypot(dx,dy),np.arctan2(dy,dx)]))
        # Exact organizer-supplied formation surfaces and layer thicknesses.
        ff=hs[FORMS].to_numpy(float)
        marker.append(np.column_stack([ff,ff-zc[:,None],np.gradient(ff,axis=0),
                                       np.diff(ff,axis=1)]))
        # Raw paired typewell global distribution and geology contact positions.
        qs=np.nanquantile(tg,[.05,.25,.5,.75,.95])
        vals=np.r_[np.nanmin(tv),np.nanmax(tv),np.ptp(tv),np.nanmean(tg),
                   np.nanstd(tg),qs,gt,np.diff(gt)]
        typewell.append(np.tile(vals,(n,1)))
        if (wi+1)%100==0: print("raw wells",wi+1,flush=True)
    fam={"prefix_absolute":np.vstack(prefix).astype(np.float32),
         "complete_trajectory_raw":np.vstack(traj).astype(np.float32),
         "formation_surface_raw":np.vstack(marker).astype(np.float32),
         "paired_typewell_raw":np.vstack(typewell).astype(np.float32)}
    np.savez_compressed(cache,**fam)

def model(seed):
    return lgb.LGBMRegressor(objective="huber",n_estimators=420,
      learning_rate=.035,num_leaves=28,max_depth=8,min_child_samples=110,
      max_bin=127,colsample_bytree=.7,subsample=.85,subsample_freq=1,
      reg_alpha=2.,reg_lambda=16.,verbosity=-1,n_jobs=12,random_state=seed)
def rmse(p,ix):return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))

# Saved baseline was produced with exactly this residual target/model protocol.
saved=np.load(ROOT/"exp/results/harshini_target_representation/oof.npz")
base=saved["direct_physics_residual"].astype(np.float32)
preds={}
folds=list(GroupKFold(5).split(X0,groups=groups))
report={}
t0=time.time()
for fi,(name,F) in enumerate(fam.items()):
    p=np.zeros(len(y),np.float32)
    for fold,(tr,va) in enumerate(folds):
        fit=tr[tr%8==fold%8]
        m=model(9000+100*fi+fold)
        m.fit(np.c_[X0[fit],F[fit]],resid[fit])
        p[va]=physics[va]+m.predict(np.c_[X0[va],F[va]])
    gains=[rmse(base,va)-rmse(p,va) for tr,va in folds]
    report[name]={"columns":F.shape[1],"rmse":rmse(p,np.arange(len(y))),
      "gain_vs_identical_base":rmse(base,np.arange(len(y)))-rmse(p,np.arange(len(y))),
      "fold_gains":gains,"fold_wins":sum(x>0 for x in gains)}
    preds[name]=p
    print(name,report[name],flush=True)
summary={"schema":{"horizontal":["MD","X","Y","Z"]+FORMS+["GR","TVT_input"],
 "excluded_target":"TVT","typewell":["TVT","GR","Geology"]},
 "protocol":"5-fold whole-well GroupKFold; stride-8 train; all suffix rows score",
 "base_rmse":rmse(base,np.arange(len(y))),"families":report,"seconds":time.time()-t0}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
np.savez_compressed(OUT/"oof.npz",base=base,y=y,**preds)
print(json.dumps(summary,indent=2))
