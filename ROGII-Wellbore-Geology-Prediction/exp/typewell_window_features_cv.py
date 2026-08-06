"""Multi-scale candidate-coordinate horizontal/typewell GR matching features."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
H=ROOT/"exp/results/harshini_cached_xgb"
OUT=ROOT/"exp/results/typewell_window_features"
OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_pickle(H/"rows.pkl")
y=d.target.to_numpy(np.float32); groups=d.well.to_numpy()
warm=1-np.exp(-np.maximum(d.md_since.to_numpy(float),0)/85.)
base=warm*d.blend_d.to_numpy(float); cand=d.flat.to_numpy(float)+base
forms=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
NF=40; W=np.zeros((len(d),NF),np.float32)

def roll(a,n,kind):
    s=pd.Series(a)
    if kind=="mean": return s.rolling(n,center=True,min_periods=3).mean().bfill().ffill().to_numpy()
    return s.rolling(n,center=True,min_periods=3).std().fillna(0).to_numpy()

for w,ix in d.groupby("well",sort=False).groups.items():
    ii=np.asarray(ix,int); q=cand[ii]
    tp=next(iter((ROOT/"data/train").glob(f"{w}__typewell*.csv")))
    t=pd.read_csv(tp).sort_values("TVT"); tv=t.TVT.to_numpy(float)
    tg=t.GR.interpolate(limit_direction="both").fillna(t.GR.median()).to_numpy(float)
    geo=t.Geology.fillna("").astype(str).to_numpy()
    hg=d.GR.to_numpy(float)[ii]; col=0
    hs={n:(roll(hg,n,"mean"),roll(hg,n,"std")) for n in (21,51,101)}
    # A candidate defines the complete typewell reference sequence. Compare
    # windows after small global TVT shifts; this keeps full curve context.
    for sh in (-8.,0.,8.):
        ref=np.interp(q+sh,tv,tg)
        for n in (21,51,101):
            hm,hstd=hs[n]; rm=roll(ref,n,"mean"); rstd=roll(ref,n,"std")
            diff=hg-ref
            drm=np.sqrt(np.maximum(0,roll(diff*diff,n,"mean")))
            # rolling correlation from moments
            cov=roll(hg*ref,n,"mean")-hm*rm
            cor=cov/(hstd*rstd+1e-3)
            W[ii,col]=drm/(hstd+rstd+3.); W[ii,col+1]=np.clip(cor,-1,1)
            col+=2
        W[ii,col]=np.gradient(hg)-np.gradient(ref)
        W[ii,col+1]=hg-ref; col+=2
    # Texture comparisons at candidate coordinate.
    for off in (2.,5.,12.):
        lo=np.interp(q-off,tv,tg); hi=np.interp(q+off,tv,tg)
        W[ii,col]=(hi-lo)/(2*off); W[ii,col+1]=np.abs(hi+lo-2*np.interp(q,tv,tg))
        col+=2
    jj=np.clip(np.searchsorted(tv,q),0,len(tv)-1); labels=geo[jj]
    for f in forms:
        W[ii,col]=(labels==f); col+=1
    # Direction and suffix coordinates, explicitly highlighted by discussion.
    dx=d.loc[ii,"dZ"].to_numpy(float)
    W[ii,col]=np.sign(np.nanmean(np.gradient(dx))); col+=1
    W[ii,col]=d.loc[ii,"s"].to_numpy(float); col+=1
    W[ii,col]=d.loc[ii,"s"].to_numpy(float)**2; col+=1
    W[ii,col]=np.log1p(d.loc[ii,"md_since"].to_numpy(float)); col+=1

core=["pf_std","pf_gap","disagree","imp_spread","pfx_rmse","tda_min",
      "tda_argmin","tda_curv","GR_dev","dip","dZ"]
X=np.c_[base,W,d[core].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)]
sample=np.arange(len(d))%8==0; si=np.flatnonzero(sample); oof=np.zeros(len(d),np.float32)
for k,(tr0,va0) in enumerate(GroupKFold(5).split(X[sample],groups=groups[sample])):
    tr=si[tr0]; vw=set(groups[si[va0]])
    va=np.fromiter((w in vw for w in groups),bool,len(groups))
    m=lgb.LGBMRegressor(objective="regression",n_estimators=360,
      learning_rate=.04,num_leaves=20,max_depth=6,min_child_samples=300,
      subsample=.8,subsample_freq=1,colsample_bytree=.75,
      reg_alpha=3,reg_lambda=35,verbosity=-1,n_jobs=8,random_state=7600+k)
    m.fit(X[tr],(y-base)[tr]);oof[va]=m.predict(X[va]);print("fold",k,flush=True)
def rmse(p):return float(np.sqrt(np.mean((p-y)**2)))
grid=[{"weight":float(a),"rmse":rmse(base+a*oof)} for a in np.linspace(0,1,21)]
best=min(grid,key=lambda q:q["rmse"])
out={"base":rmse(base),"raw":rmse(base+oof),"best":best,
 "features":X.shape[1],"rows":len(d),"wells":int(d.well.nunique()),
 "protocol":"5-fold whole-well GroupKFold; all typewell/horizontal inputs legal"}
np.save(OUT/"correction_oof.npy",oof)
pd.DataFrame(grid).to_csv(OUT/"grid.csv",index=False)
(OUT/"summary.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
