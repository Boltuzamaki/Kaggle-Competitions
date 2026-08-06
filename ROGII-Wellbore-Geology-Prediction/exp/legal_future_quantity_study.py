"""Bounded study of predictable legal future quantities.

One ExtraTrees model predicts complete-well quadratic residual parameters
(intercept/slope/curvature), future mean, and toe residual from aggregated
test-available Harshini features. Whole-well shuffled KFold; suffix labels are
targets only.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[1]
H=ROOT/"exp/results/harshini_cached_xgb"
OUT=ROOT/"exp/results/legal_future_quantity"
OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_pickle(H/"rows.pkl")
legal=[
 "s","md_since","dZ","Z","dip","z_span","GR","GR_dev","pf_std","pf_gap",
 "disagree","imp_spread","nn","pr_min","pr_med","pr_max","pr_std","pfx_rmse",
 "tda_min","tda_argmin","tda_curv","ktvt_range","ktvt_std","slp_all","slp_50",
 "dzdmd","grs21","grs51","grs101","cal_a","cal_b","blend_d","prod_d",
 "pf_mean_d","imp_raw_d"]
legal=[c for c in legal if c in d]
rows=[]; targets=[]; info=[]
for w,g in d.groupby("well",sort=True):
    feats=[]
    for c in legal:
        a=g[c].to_numpy(float); a=np.nan_to_num(a,nan=0,posinf=0,neginf=0)
        feats += [a.mean(),a.std(),np.quantile(a,.1),np.quantile(a,.9),a[0],a[-1]]
    s=g.s.to_numpy(float); y=g.target.to_numpy(float)
    co=np.polyfit(s,y,2) # curvature, slope, intercept
    targets.append(np.r_[co,y.mean(),y[-1]])
    rows.append(feats); info.append((w,g.index.to_numpy()))
X=np.asarray(rows,np.float32); T=np.asarray(targets,np.float32)
P=np.zeros_like(T)
for k,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=73126).split(X)):
    m=ExtraTreesRegressor(n_estimators=1000,max_features=.8,
        min_samples_leaf=3,n_jobs=-1,random_state=73126+k)
    m.fit(X[tr],T[tr]); P[va]=m.predict(X[va])

pred_poly=np.zeros(len(d)); pred_const=np.zeros(len(d)); pred_toe=np.zeros(len(d))
for i,(w,ix) in enumerate(info):
    s=d.loc[ix,"s"].to_numpy(float)
    pred_poly[ix]=np.polyval(P[i,:3],s)
    pred_const[ix]=P[i,3]
    pred_toe[ix]=P[i,4]*s # simple anchored toe ramp
y=d.target.to_numpy(float)
physics=(1-np.exp(-np.maximum(d.md_since.to_numpy(float),0)/85))*d.blend_d.to_numpy(float)
def rmse(p):return float(np.sqrt(np.mean((p-y)**2)))
grid=[]
for name,p in [("poly",pred_poly),("mean",pred_const),("toe_ramp",pred_toe)]:
    for a in np.linspace(0,1,21):
        grid.append({"quantity":name,"weight":float(a),
                     "rmse":rmse((1-a)*physics+a*p)})
best=min(grid,key=lambda q:q["rmse"])
out={"wells":len(info),"features":X.shape[1],"physics":rmse(physics),
 "standalone":{"poly":rmse(pred_poly),"mean":rmse(pred_const),
               "toe_ramp":rmse(pred_toe)},"best_blend":best,
 "target_parameter_mae":{
  "curvature":float(np.mean(abs(P[:,0]-T[:,0]))),
  "slope":float(np.mean(abs(P[:,1]-T[:,1]))),
  "intercept":float(np.mean(abs(P[:,2]-T[:,2]))),
  "future_mean":float(np.mean(abs(P[:,3]-T[:,3]))),
  "toe":float(np.mean(abs(P[:,4]-T[:,4])))},
 "protocol":"5-fold complete-well CV; inputs aggregated from legal full-well features"}
pd.DataFrame(grid).to_csv(OUT/"blend_grid.csv",index=False)
np.savez_compressed(OUT/"well_oof.npz",pred=P,target=T)
(OUT/"summary.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
