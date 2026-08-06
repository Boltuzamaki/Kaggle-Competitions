"""Grouped audit of legal TVT/trajectory target coordinate formulations."""
from pathlib import Path
import json,joblib
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/target_coordinate_formulation";OUT.mkdir(parents=True,exist_ok=True)
f=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl");drop={"well","id","target","last_known_tvt","supertype"}
base_cols=[c for c in f if c not in drop];X0=f[base_cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
y=f.target.to_numpy(np.float32);g=f.well.to_numpy();n=len(f);meta=np.zeros((n,13),np.float32)
for wi,(w,ix0) in enumerate(f.groupby("well",sort=False).indices.items()):
 ix=np.asarray(ix0);h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv");t=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv")
 nv=int(h.TVT_input.notna().sum());p=nv-1;md=h.MD.to_numpy(float);z=h.Z.to_numpy(float);x=h.X.to_numpy(float);yy=h.Y.to_numpy(float)
 rr=np.array([int(v.rsplit("_",1)[1]) for v in f.id.iloc[ix]],int);dx=x[-1]-x[p];dy=yy[-1]-yy[p];norm=np.hypot(dx,dy)+1e-6
 ux,uy=dx/norm,dy/norm;rx=x[rr]-x[p];ry=yy[rr]-yy[p]
 take=slice(max(0,nv-250),nv);tvp=h.TVT_input.iloc[take].to_numpy(float);mp=md[take]
 tvsl=np.polyfit(mp,tvp,1)[0];usl=np.polyfit(mp,tvp+z[take],1)[0]
 tt=t.TVT.to_numpy(float);last=float(h.TVT_input.iloc[p])
 meta[ix]=np.column_stack([np.full(len(ix),last),np.full(len(ix),z[p]),np.full(len(ix),md[p]),
  np.full(len(ix),tt.min()),np.full(len(ix),tt.max()),np.full(len(ix),np.ptp(tt)),
  np.full(len(ix),tvsl),np.full(len(ix),usl),np.full(len(ix),np.sign(dx)),np.full(len(ix),np.sign(dy)),
  ux*rx+uy*ry,-uy*rx+ux*ry,(md[rr]-md[p])/(md[-1]-md[p]+1)])
# Target-independent pilot keeps the representation audit computationally
# comparable while another full-row experiment shares the machine.
rng=np.random.RandomState(2608);keep=np.sort(rng.choice(np.array(sorted(f.well.unique())),240,False))
mask=f.well.isin(keep).to_numpy();f=f.loc[mask].reset_index(drop=True);X0=X0[mask];y=y[mask];g=g[mask];meta=meta[mask];n=len(f)
X=np.c_[X0,meta];last=meta[:,0];zps=meta[:,1];twmin=meta[:,3];twspan=np.maximum(meta[:,5],1)
dmd=np.maximum(f.d_md.to_numpy(np.float32),20);dz=f.d_z.to_numpy(np.float32);tvsl=meta[:,6];usl=meta[:,7]
forms={
 "delta_direct":(np.clip(y,-100,100),lambda p:p),
 "absolute_TVT":(last+y,lambda p:p-last),
 "marker_U_relative":(np.clip(y+dz,-100,100),lambda p:p-dz),
 "marker_U_absolute":(last+y+zps+dz,lambda p:p-last-zps-dz),
 "TVT_slope":(np.clip(y/dmd,-1,1),lambda p:p*dmd),
 "U_slope":(np.clip((y+dz)/dmd,-1,1),lambda p:p*dmd-dz),
 "prefix_TVT_line_residual":(np.clip(y-tvsl*f.d_md.to_numpy(),-80,80),lambda p:p+tvsl*f.d_md.to_numpy()),
 "prefix_U_line_residual":(np.clip(y+dz-usl*f.d_md.to_numpy(),-80,80),lambda p:p+usl*f.d_md.to_numpy()-dz),
 "typewell_normalized_TVT":((last+y-twmin)/twspan,lambda p:p*twspan+twmin-last)}
def model(seed):return lgb.LGBMRegressor(objective="huber",n_estimators=180,learning_rate=.05,num_leaves=24,max_depth=7,
 min_child_samples=110,max_bin=127,colsample_bytree=.7,subsample=.85,subsample_freq=1,reg_alpha=2.,reg_lambda=16.,
 verbosity=-1,n_jobs=12,random_state=seed)
def rm(p,ix=None):
 if ix is None:ix=np.arange(n)
 return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))
folds=list(GroupKFold(5).split(X,groups=g));report={};preds={}
for fi,(name,(lab,recon)) in enumerate(forms.items()):
 raw=np.zeros(n,np.float32);p=np.zeros(n,np.float32)
 for fold,(tr,va) in enumerate(folds):
  fit=tr[tr%16==fold%16];m=model(1000*fi+fold);m.fit(X[fit],lab[fit]);raw[va]=m.predict(X[va]);p[va]=recon(raw[va],va) if False else 0
 # Reconstruction lambdas operate on full row-aligned arrays; apply once.
 p=np.asarray(recon(raw),np.float32)
 fg=[rm(np.zeros(n),va) if False else None for tr,va in folds]
 fold_rm=[rm(p,va) for tr,va in folds]
 r2=1-float(np.sum((raw-lab)**2)/(np.sum((lab-lab.mean())**2)+1e-9))
 report[name]={"rmse":rm(p),"fold_rmse":fold_rm,"label_oof_r2":r2};preds[name]=p
 print(name,report[name],flush=True)
best=min(report,key=lambda k:report[k]["rmse"])
# Diagnose and test the most plausible direction split for the winning representation.
dxpos=meta[:,8]>0;dypos=meta[:,9]>0
for name,mask in {"dx_positive":dxpos,"dy_positive":dypos}.items():
 report[best][name+"_false_rmse"]=rm(preds[best],~mask);report[best][name+"_true_rmse"]=rm(preds[best],mask)
# Strict separate best-representation models by dx sign.
lab,recon=forms[best];raw=np.zeros(n,np.float32)
for fold,(tr,va) in enumerate(folds):
 for val in (False,True):
  fit=tr[dxpos[tr]==val];fit=fit[fit%16==fold%16];vv=va[dxpos[va]==val]
  m=model(90000+10*fold+int(val));m.fit(X[fit],lab[fit]);raw[vv]=m.predict(X[vv])
splitp=np.asarray(recon(raw),np.float32);splitfg=[rm(preds[best],va)-rm(splitp,va) for tr,va in folds]
summary={"rows":n,"wells":int(f.well.nunique()),"features":X.shape[1],"representations":report,"best":best,
 "dx_separate":{"rmse":rm(splitp),"gain_vs_best":rm(preds[best])-rm(splitp),"fold_gains":splitfg,"fold_wins":sum(x>0 for x in splitfg)},
 "protocol":"target-independent 240-well pilot; 5-fold complete-well GKF; stride-16 fit/all suffix score; common-schema anchors/typewell TVT only"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/"oof.npz",y=y,split_dx=splitp,**preds)
print(json.dumps({"best":best,"best_result":report[best],"dx_separate":summary["dx_separate"]},indent=2))
