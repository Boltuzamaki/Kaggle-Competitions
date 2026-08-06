"""Fast grouped CV for raw paired-typewell candidate-coordinate features.

This tests the largest feature-family omission found in public notebooks:
Harshini uses typewell GR in the PF/global scans but does not expose the raw
typewell GR and Geology at each row's current TVT candidate to its GBDT.
"""
from pathlib import Path
import glob, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
H=ROOT/"exp/results/harshini_cached_xgb"
OUT=ROOT/"exp/results/typewell_candidate_features"
OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_pickle(H/"rows.pkl")
y=d.target.to_numpy(np.float32); groups=d.well.to_numpy()
warm=1-np.exp(-np.maximum(d.md_since.to_numpy(float),0)/85.)
base=warm*d.blend_d.to_numpy(float)
cand=d.flat.to_numpy(float)+base
forms=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
new=np.zeros((len(d),38),np.float32)

for w,ix in d.groupby("well",sort=False).groups.items():
    ii=np.asarray(ix,int)
    tp=next(iter((ROOT/"data/train").glob(f"{w}__typewell*.csv")))
    t=pd.read_csv(tp).sort_values("TVT")
    tv=t.TVT.to_numpy(float)
    tg=t.GR.interpolate(limit_direction="both").fillna(t.GR.median()).to_numpy(float)
    geo=t.Geology.fillna("").astype(str).to_numpy()
    q=cand[ii]; hgr=d.GR.to_numpy(float)[ii]
    col=0
    for sh in (-30,-15,-8,-4,0,4,8,15,30):
        rg=np.interp(q+sh,tv,tg)
        new[ii,col]=rg; new[ii,col+1]=hgr-rg; col+=2
    # Raw geology identity at candidate shifts (major formations ordinal and
    # one equality indicator at the central candidate).
    for sh in (-15,0,15):
        jj=np.clip(np.searchsorted(tv,q+sh),0,len(tv)-1)
        labels=geo[jj]
        code=np.array([forms.index(a)+1 if a in forms else 0 for a in labels])
        new[ii,col]=code; col+=1
        if sh==0:
            for f in forms:
                new[ii,col]=(labels==f); col+=1
    # Signed distance to each paired-typewell formation marker.
    for f in forms:
        vv=tv[geo==f]
        center=np.median(vv) if len(vv) else np.nan
        new[ii,col]=np.nan_to_num(q-center,nan=999.); col+=1
    # Candidate/typewell edge and local GR-gradient context.
    new[ii,col]=q-tv[0]; new[ii,col+1]=tv[-1]-q
    new[ii,col+2]=np.interp(q+2,tv,tg)-np.interp(q-2,tv,tg)
    new[ii,col+3]=np.interp(q+8,tv,tg)-np.interp(q-8,tv,tg)

core=["s","md_since","dZ","dip","GR","GR_dev","pf_std","pf_gap",
      "disagree","imp_spread","pfx_rmse","tda_min","tda_argmin","tda_curv"]
X=np.c_[base,new,d[core].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)]
sample=np.arange(len(d))%8==0; si=np.flatnonzero(sample)
oof=np.zeros(len(d),np.float32)
for k,(tr0,va0) in enumerate(GroupKFold(5).split(X[sample],groups=groups[sample])):
    tr=si[tr0]; vw=set(groups[si[va0]])
    va=np.fromiter((w in vw for w in groups),bool,len(groups))
    m=lgb.LGBMRegressor(objective="regression",n_estimators=320,
        learning_rate=.04,num_leaves=20,max_depth=6,min_child_samples=300,
        subsample=.8,subsample_freq=1,colsample_bytree=.75,
        reg_alpha=3,reg_lambda=30,verbosity=-1,n_jobs=8,random_state=7500+k)
    m.fit(X[tr],(y-base)[tr]); oof[va]=m.predict(X[va])
    print("fold",k,flush=True)
def rmse(p):return float(np.sqrt(np.mean((p-y)**2)))
grid=[{"weight":a,"rmse":rmse(base+a*oof)} for a in np.linspace(0,1,21)]
best=min(grid,key=lambda q:q["rmse"])
out={"rows":len(d),"wells":int(d.well.nunique()),"base":rmse(base),
     "raw_correction":rmse(base+oof),"best":best,
     "feature_family":"raw typewell GR differences, Geology identity, formation-boundary distances at candidate TVT",
     "protocol":"5-fold GroupKFold by whole well; candidate path and all typewell columns inference-available"}
pd.DataFrame(grid).to_csv(OUT/"grid.csv",index=False)
np.save(OUT/"correction_oof.npy",oof)
(OUT/"summary.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
