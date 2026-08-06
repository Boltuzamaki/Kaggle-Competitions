"""Physical S=TVT+Z transition model with visible-prefix calibration."""
from pathlib import Path
import argparse,glob,json,joblib
import numpy as np,pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data/train"
OUT=ROOT/"exp/results/structural_state_transition"

def features(h):
 md=h.MD.to_numpy(float);x=h.X.to_numpy(float);y=h.Y.to_numpy(float);z=h.Z.to_numpy(float)
 dm=np.diff(md);dx=np.diff(x);dy=np.diff(y);dz=np.diff(z)
 dsxy=np.hypot(dx,dy);d3=np.sqrt(dx*dx+dy*dy+dz*dz)
 ux=dx/np.maximum(d3,1e-4);uy=dy/np.maximum(d3,1e-4);uz=dz/np.maximum(d3,1e-4)
 az=np.arctan2(dy,dx);inc=np.arctan2(dsxy,-dz)
 dog=np.r_[0,np.sqrt(np.diff(ux)**2+np.diff(uy)**2+np.diff(uz)**2)]
 cols=[dm,dx,dy,dz,dsxy,d3,ux,uy,uz,np.sin(az),np.cos(az),inc,dog]
 # interactions encode structural dip projected along trajectory.
 cols.extend([dx*ux,dy*uy,dx*uy,dy*ux,dz*inc])
 return np.column_stack(cols).astype(np.float32)

def load(path,global_idx):
 wid=Path(path).name.split("__")[0];h=pd.read_csv(path);ps=int(h.TVT_input.notna().sum())
 if ps<20 or ps>=len(h):return None
 S=h.TVT.to_numpy(float)+h.Z.to_numpy(float)
 return dict(well=wid,h=h,ps=ps,X=features(h),dS=np.diff(S).astype(np.float32),
  S=S,global_idx=np.asarray(global_idx,np.int64))

def calibrate_and_predict(model,w,use_gr=True):
 h=w["h"];ps=w["ps"];pred=model.predict(w["X"])
 # Learned global transition plus a robust, shrunk local dip correction from
 # the last 800 visible increments.
 obs=np.diff(h.TVT_input.iloc[:ps].to_numpy(float)+h.Z.iloc[:ps].to_numpy(float))
 r=obs-pred[:ps-1];tail=r[-min(800,len(r)):]
 med=float(np.median(tail));noise=float(np.median(abs(tail-med))*1.4826)
 shrink=len(tail)/(len(tail)+200*(1+noise/.03))
 bias=float(np.clip(shrink*med,-.06,.06))
 # Curvature prior: a very small trend in transition residual, strongly shrunk.
 if len(tail)>=100:
  q=np.arange(len(tail));slope=np.polyfit(q,tail,1)[0]
  slope=float(np.clip(slope*shrink,-2e-6,2e-6))
 else:slope=0.
 n=len(h)-ps;future=np.arange(1,n+1)
 step=pred[ps-1:]+bias+slope*(future+len(tail)/2)
 S0=float(h.TVT_input.iloc[ps-1]+h.Z.iloc[ps-1])
 Sp=S0+np.cumsum(step);tvt=Sp-h.Z.iloc[ps:].to_numpy(float)
 gr_shift=0.
 if use_gr:
  # Bounded global GR datum likelihood; it may translate the physical curve by
  # at most 3 ft and cannot select a free path.
  t=pd.read_csv(DATA/f"{w['well']}__typewell.csv").sort_values("TVT")
  tt=t.TVT.to_numpy(float);tg=t.GR.interpolate(limit_direction="both").fillna(t.GR.median()).to_numpy(float)
  hg=h.GR.interpolate(limit_direction="both").fillna(h.GR.median()).to_numpy(float)
  ref=np.interp(h.TVT_input.iloc[:ps],tt,tg)
  aa,bb=np.linalg.lstsq(np.c_[ref,np.ones(ps)],hg[:ps],rcond=None)[0]
  shifts=np.arange(-12,12.1,1);idx=np.linspace(0,n-1,min(n,800)).astype(int)
  costs=[]
  scale=max(np.median(abs(hg[:ps]-(aa*ref+bb)))*1.4826,5)
  for sh in shifts:
   z=(hg[ps:][idx]-(aa*np.interp(tvt[idx]+sh,tt,tg)+bb))/scale
   costs.append(np.mean(np.log1p((z/2)**2)))
  p=np.exp(-(np.array(costs)-min(costs))/.15);p/=p.sum()
  gr_shift=float(np.clip(.18*(p@shifts),-3,3));tvt+=gr_shift
 return tvt.astype(np.float32),dict(bias=bias,curvature=slope,gr_shift=gr_shift)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--folds",type=int,default=1);ap.add_argument("--max-wells",type=int)
 a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 feat=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
 v4o=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"])
 bywell={w:q.index.to_numpy() for w,q in feat.groupby("well",sort=False)}
 paths=sorted(glob.glob(str(DATA/"*__horizontal_well.csv")))
 if a.max_wells:paths=paths[:a.max_wells]
 wells=[q for q in (load(p,bywell[Path(p).name.split("__")[0]]) for p in paths) if q]
 splits=list(GroupKFold(min(5,len(wells))).split(wells,groups=[w["well"] for w in wells]))
 oof=np.full(len(feat),np.nan,np.float32);fid=np.full(len(feat),-1,np.int8);fr=[];diag=[]
 for f,(ti,vi) in enumerate(splits[:a.folds]):
  X=np.concatenate([wells[i]["X"][::8] for i in ti]);Y=np.concatenate([wells[i]["dS"][::8] for i in ti])
  model=LGBMRegressor(n_estimators=350,learning_rate=.04,num_leaves=31,min_child_samples=300,
   subsample=.85,colsample_bytree=.8,reg_lambda=5,n_jobs=-1,verbosity=-1,random_state=2026+f)
  model.fit(X,Y);se=bs=nr=0
  for j in vi:
   w=wells[j];p,d=calibrate_and_predict(model,w);truth=w["h"].TVT.iloc[w["ps"]:].to_numpy(float)
   err=p-truth;se+=np.square(err).sum();nr+=len(err)
   # Exact V4 aligned baseline.
   bv=feat.loc[w["global_idx"],"target"].to_numpy(float)
   vv=v4o[w["global_idx"]]
   bs+=np.square(vv-bv).sum();oof[w["global_idx"]]=p-w["h"].TVT_input.iloc[w["ps"]-1];fid[w["global_idx"]]=f
   diag.append(dict(well=w["well"],fold=f,rmse=float(np.sqrt(np.mean(err**2))),**d))
  fr.append(dict(fold=f,rows=int(nr),rmse=float(np.sqrt(se/nr)),v4_rmse=float(np.sqrt(bs/nr))))
  joblib.dump(model,OUT/f"fold{f}.joblib");print(json.dumps(fr[-1]),flush=True)
 total=float(np.sqrt(np.average([x["rmse"]**2 for x in fr],weights=[x["rows"] for x in fr])))
 v4=float(np.sqrt(np.average([x["v4_rmse"]**2 for x in fr],weights=[x["rows"] for x in fr])))
 summary=dict(pooled_rmse=total,v4_rmse=v4,gain=v4-total,folds=fr,wells=len(wells),
  state="S=TVT+Z",outer_fold_transition_training=True,visible_prefix_calibration=True,
  bounded_gr_shift_ft=3,no_neighbors=True)
 np.savez_compressed(OUT/"oof_delta.npz",prediction=oof,fold=fid,target=feat.target.to_numpy(np.float32),ids=feat.id.to_numpy(str))
 pd.DataFrame(diag).to_csv(OUT/"well_metrics.csv",index=False)
 (OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
