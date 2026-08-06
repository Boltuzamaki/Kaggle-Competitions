"""Strict whole-well cross-fitted rowwise uncertainty/residual gate.

Targets are used only for fitting on outer-training wells and final scoring.
Inputs are deployment-legal row features plus disagreement among ten OOF experts.
"""
from pathlib import Path
import contextlib, io, json, runpy
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/rowwise_uncertainty_gate";OUT.mkdir(parents=True,exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()): s=runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
common,y,wells,L=s["common"],s["y"].astype(float),s["wells"].astype(str),s["legs"]
fm=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/full_meta/oof.npz")
vf=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl").iloc[common].reset_index(drop=True)
hq=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")
base=fm["add"].astype(float); heel=L["v4_lgb7"]+hq["correction"][common]
def project(p,deg):
 out=p.copy()
 for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False):
  ix=ii.to_numpy();x=vf.d_md.to_numpy(float)[ix];x=2*(x-x.min())/max(np.ptp(x),1e-6)-1
  z=p[ix]+vf.d_z.to_numpy(float)[ix];out[ix]=np.polyval(np.polyfit(x,z,deg),x)-vf.d_z.to_numpy(float)[ix]
 return out
P=np.column_stack([base,fm["base"],heel,L["v4_lgb7"],L["har_physics"],L["har_lgb"],L["har_xgb"],
 L["pil_blend_oof_postprocessed"],project(L["v4_lgb7"].copy(),2),project(L["v4_lgb7"].copy(),3)]).astype(np.float32)

# Local evidence and uncertainty. Exclude identifiers, target and absolute predictions;
# use prediction deltas so the model learns a correction/gate rather than memorizing datum.
cols=[c for c in vf.columns if c not in {"well","id","target","supertype","last_known_tvt"}]
R=vf[cols].to_numpy(np.float32)
D=P[:,1:]-P[:,[0]]
unc=np.c_[D,D.mean(1),D.std(1),np.max(D,1),np.min(D,1),np.ptp(D,axis=1)].astype(np.float32)
X=np.c_[R,unc]
X=np.nan_to_num(X,nan=0.,posinf=0.,neginf=0.)
target=np.clip(y-base,-40,40).astype(np.float32)
wn=pd.Series(wells).drop_duplicates().to_numpy(); fold_of_well={w:k for k,(_,va) in enumerate(GroupKFold(5).split(wn,groups=wn)) for w in wn[va]}
fid=np.array([fold_of_well[w] for w in wells],np.int8);pred=np.zeros(len(y),np.float32)
rng=np.random.default_rng(8123)
for fold in range(5):
 trw=np.flatnonzero(np.array([fold_of_well[w]!=fold for w in wn]))
 # Balanced sampling prevents long wells dominating and makes this CPU-feasible.
 take=[]
 for w in wn[trw]:
  ix=np.flatnonzero(wells==w); take.append(rng.choice(ix,min(900,len(ix)),replace=False))
 tr=np.concatenate(take);va=np.flatnonzero(fid==fold)
 m=CatBoostRegressor(iterations=450,depth=7,learning_rate=.045,l2_leaf_reg=25,
  loss_function="RMSE",random_seed=9100+fold,verbose=False,
  allow_writing_files=False,thread_count=2,random_strength=.4,task_type="GPU",devices="0")
 m.fit(X[tr],target[tr]); raw=np.asarray(m.predict(X[va]))
 pred[va]=raw[:,0] if raw.ndim==2 else raw
 print(f"fold {fold} train={len(tr)} valid={len(va)}",flush=True)
def rm(q,m=None):
 if m is None:m=np.ones(len(y),bool)
 return float(np.sqrt(np.mean((q[m]-y[m])**2)))
grid=[]
for clip in (2,4,6,10,20,40):
 for a in (.1,.25,.5,.75,1.):
  q=base+a*np.clip(pred,-clip,clip); fg=[rm(base,fid==k)-rm(q,fid==k) for k in range(5)]
  grid.append({"clip":clip,"blend":a,"rmse":rm(q),"gain":rm(base)-rm(q),"fold_gains":fg,"fold_wins":sum(x>0 for x in fg)})
grid.sort(key=lambda z:z["rmse"])
summary={"baseline":rm(base),"best":grid[0],"top10":grid[:10],"features":X.shape[1],"rows":len(y),
 "protocol":"strict outer whole-well 5-fold; rowwise legal features and expert deltas; balanced train-well row sampling"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/"oof.npz",pred_correction=pred,base=base,y=y,fold=fid)
print(json.dumps(summary,indent=2))
