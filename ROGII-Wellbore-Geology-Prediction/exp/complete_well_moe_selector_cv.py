"""Nested whole-well mixture-of-experts over existing legal OOF paths."""
from pathlib import Path
import contextlib,io,json,runpy
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/complete_well_moe";OUT.mkdir(parents=True,exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    state=runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
common=state["common"];y=state["y"];wells=state["wells"];legs=state["legs"]
fm=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/full_meta/oof.npz")
if not np.array_equal(fm["global_indices"],common) or np.max(abs(fm["y"]-y))>.002:
    raise RuntimeError("accepted heel-meta identity failed")
accepted=fm["add"].astype(float)
heel_global=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")
heel=legs["v4_lgb7"]+heel_global["correction"][common]
vf=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl").iloc[common].reset_index(drop=True)
if not np.array_equal(vf.id.to_numpy(),
    pd.read_parquet(ROOT/"exp/public_artifacts/pilkwang/oof/train_gt.parquet",columns=["id"]).id.to_numpy()[common]):
    raise RuntimeError("feature identity failed")

def project(p,deg):
 out=p.copy()
 for _,ix0 in pd.Series(np.arange(len(y))).groupby(wells,sort=False):
  ix=ix0.to_numpy();x=vf.d_md.to_numpy(float)[ix]
  x=2*(x-x.min())/max(np.ptp(x),1e-6)-1
  structural=p[ix]+vf.d_z.to_numpy(float)[ix]
  out[ix]=np.polyval(np.polyfit(x,structural,deg),x)-vf.d_z.to_numpy(float)[ix]
 return out

experts={
 "accepted_heel_meta":accepted,
 "base_five_meta":fm["base"].astype(float),
 "heel_v4":heel,
 "v4_lgb7":legs["v4_lgb7"],
 "har_physics":legs["har_physics"],
 "har_lgb":legs["har_lgb"],
 "har_xgb":legs["har_xgb"],
 "pil_blend":legs["pil_blend_oof_postprocessed"],
}
experts["v4_poly2"]=project(legs["v4_lgb7"].copy(),2)
experts["v4_poly3"]=project(legs["v4_lgb7"].copy(),3)
names=list(experts);P=np.column_stack([experts[k] for k in names])

# Heel posterior diagnostics, aligned by well name (not position).
hrec={r["well"]:r for r in json.loads(
    (ROOT/"exp/results/heel_calibrated_gr_datum/costs.json").read_text())}
shifts=np.arange(-60,61,2.)
features=[]; losses=[]; wix=[]; wn=[]
for w,ix0 in pd.Series(np.arange(len(y))).groupby(wells,sort=False):
 ix=ix0.to_numpy();g=vf.iloc[ix]; q=hrec[w]
 C=np.asarray(q["cauchy_costs"],float);C=C-C.min()
 post=np.exp(-C/.2);post/=post.sum()
 # Compact legal input/prediction diagnostics only.
 z=[
  np.log1p(len(ix)),float(np.ptp(g.d_md)),float(np.ptp(g.d_z)),
  float(np.ptp(g.d_xy)),float(g.dz_dmd.mean()),float(g.dz_dmd.std()),
  float(g.gr.mean()),float(g.gr.std()),float(g.gr_s21.mean()),
  float(g.gr_s21.std()),float(g.pfx_gr_rmse.iloc[0]),
  float(g.pf_dis.mean()),float(g.pf_dis.std()),
  float(g.pf_vs_beam.abs().mean()),float(g.sp_std.mean()),
  float(g.sp_dmin.mean()),float(g.ncc8_s.mean()),float(g.ncc15_s.mean()),
  float(g.ncc25_s.mean()),float(q["alpha"]),float(q["beta"]/100),
  float(q["scale"]),float(C.min()),float(np.partition(C,1)[1]),
  float(-(post*np.log(post+1e-12)).sum()),float(post@shifts),
  float(np.sqrt(post@(shifts**2)-(post@shifts)**2)),
 ]
 # Expert path morphology/disagreement is legal and deployment-reproducible.
 for j in range(P.shape[1]):
  pj=P[ix,j];z += [float(pj.mean()),float(pj.std()),
   float(pj[-1]-pj[0]),float(np.sqrt(np.mean((pj-accepted[ix])**2)))]
 features.append(z);losses.append(np.sqrt(np.mean((P[ix]-y[ix,None])**2,axis=0)))
 wix.append(ix);wn.append(w)
X=np.nan_to_num(np.asarray(features,np.float32),nan=0,posinf=0,neginf=0)
L=np.asarray(losses);wn=np.asarray(wn);nrow=np.array([len(i) for i in wix])
np.savez_compressed(OUT/"well_table.npz",X=X,L=L,wn=wn,nrow=nrow)

# Strict outer GroupKFold. Each expert-loss regressor sees training wells only.
hard=np.zeros(len(y));soft=np.zeros(len(y));fold_id=np.full(len(y),-1)
fold_info=[]
for fold,(tr,va) in enumerate(GroupKFold(5).split(X,groups=wn)):
 predL=np.zeros((len(va),len(names)))
 for j in range(len(names)):
  m=CatBoostRegressor(iterations=320,depth=4,learning_rate=.035,
    loss_function="RMSE",l2_leaf_reg=20,random_seed=1200+fold*20+j,
    verbose=False,allow_writing_files=False,thread_count=8)
  m.fit(X[tr],L[tr,j],sample_weight=np.sqrt(nrow[tr]))
  predL[:,j]=m.predict(X[va])
 for qi,i in enumerate(va):
  ix=wix[i];hard[ix]=P[ix,np.argmin(predL[qi])]
  # Fixed conservative error-scale posterior; accepted expert gets a prior.
  score=predL[qi].copy();score[0]-=.20
  wt=np.exp(np.clip(-(score-score.min())/1.5,-20,0));wt/=wt.sum()
  soft[ix]=P[ix]@wt;fold_id[ix]=fold
 fold_info.append({"fold":fold,"wells":len(va)})
 print("fold",fold,flush=True)

def rmse(a,m=None):
 if m is None:m=np.ones(len(y),bool)
 return float(np.sqrt(np.mean((a[m]-y[m])**2)))
grid=[]
for method,p in (("hard",hard),("soft",soft)):
 for a in (0,.1,.2,.3,.4,.5,.7,1):
  q=(1-a)*accepted+a*p
  grid.append({"method":method,"blend":a,"rmse":rmse(q),
   "fold_gains":[rmse(accepted,fold_id==k)-rmse(q,fold_id==k) for k in range(5)]})
grid=sorted(grid,key=lambda q:q["rmse"]);best=grid[0]
summary={"rows":len(y),"wells":len(wn),"features":X.shape[1],
 "experts":names,"accepted_baseline":rmse(accepted),
 "expert_rmse":{k:rmse(v) for k,v in experts.items()},
 "oracle_family":float(np.sqrt(np.sum(nrow*np.min(L,axis=1)**2)/nrow.sum())),
 "hard":rmse(hard),"soft":rmse(soft),"best":best,
 "accept":bool(best["rmse"]<rmse(accepted) and all(x>0 for x in best["fold_gains"])),
 "folds":fold_info,"protocol":"outer 5-fold whole-well loss prediction; legal compact diagnostics"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
np.savez_compressed(OUT/"oof.npz",hard=hard,soft=soft,base=accepted,y=y,fold=fold_id)
print(json.dumps(summary,indent=2))
