"""Fast legal row-feature block ablation around honest V4 OOF candidate paths."""
from pathlib import Path
import json,joblib,sys
import numpy as np,pandas as pd,lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/raw_typewell_feature_blocks";OUT.mkdir(parents=True,exist_ok=True)
f0=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl");oo=joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
base0=np.asarray(oo["lgb7"],float);rng=np.random.RandomState(620);full="--full" in sys.argv
keep=np.array(sorted(f0.well.unique())) if full else np.sort(rng.choice(np.array(sorted(f0.well.unique())),200,False))
f=f0[f0.well.isin(keep)].copy();idx=f.index.to_numpy();f=f.reset_index(drop=True);base=base0[idx];y=f.target.to_numpy(float)
OFF=np.array([-40,-20,-10,-5,0,5,10,20,40],float);blocks={k:[] for k in ("basic","future_gr","lookup","rolling_match")}
def affine(x,y):
 ok=np.isfinite(x)&np.isfinite(y);x,y=x[ok],y[ok];X=np.c_[x,np.ones(len(x))];c=np.linalg.lstsq(X,y,rcond=None)[0]
 for _ in range(4):
  r=y-X@c;sc=1.4826*np.median(np.abs(r-np.median(r)))+1e-3;w=1/np.maximum(1,np.abs(r)/(2.5*sc));c=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]
 return c[0],c[1],sc
for wi,(w,q) in enumerate(f.groupby("well",sort=False)):
 h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv");t=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
 tv=t.TVT.to_numpy(float);tg=pd.to_numeric(t.GR,errors="coerce").interpolate(limit_direction="both").to_numpy(float)
 hg=pd.to_numeric(h.GR,errors="coerce").interpolate(limit_direction="both").to_numpy(float);vis=h.TVT_input.notna().to_numpy();p=np.flatnonzero(vis)[-1]
 a,b,sc=affine(np.interp(h.TVT_input[vis],tv,tg),hg[vis]);rr=np.array([int(z.rsplit("_",1)[1]) for z in q.id],int);n=len(rr)
 path=q.last_known_tvt.to_numpy(float)+base[q.index];obs=hg[rr];md=h.MD.to_numpy(float);xx=h.X.to_numpy(float);yy=h.Y.to_numpy(float);zz=h.Z.to_numpy(float)
 dx=np.gradient(xx)[rr];dy=np.gradient(yy)[rr];dz=np.gradient(zz)[rr]
 basic=np.column_stack([md[rr]-md[p],(md[rr]-md[p])/(md[-1]-md[p]+1),xx[rr]-xx[p],yy[rr]-yy[p],zz[rr]-zz[p],
  xx[-1]-xx[rr],yy[-1]-yy[rr],zz[-1]-zz[rr],dx,dy,dz,np.hypot(dx,dy),np.arctan2(dy,dx),
  np.full(n,h.TVT_input.iloc[p]),np.full(n,a),np.full(n,b),np.full(n,sc),np.full(n,tv.min()),np.full(n,tv.max()),np.full(n,np.ptp(tv))])
 blocks["basic"].append(basic.astype(np.float32))
 # Full suffix including future observations is available for every test well.
 fg=[]
 for win in (51,201,801):
  size=min(win,n);fg += [uniform_filter1d(obs,size,mode="nearest"),
   uniform_filter1d(obs[::-1],size,mode="nearest")[::-1],
   np.sqrt(np.maximum(uniform_filter1d(obs**2,size,mode="nearest")-uniform_filter1d(obs,size,mode="nearest")**2,0))]
 blocks["future_gr"].append(np.column_stack(fg).astype(np.float32))
 ref=np.vstack([a*np.interp(path+o,tv,tg)+b for o in OFF]).T;res=(obs[:,None]-ref)/max(sc,5)
 cost=np.log1p((res/2)**2);P=np.exp(np.clip(-(cost-cost.min(1,keepdims=True))/.4,-35,0));P/=P.sum(1,keepdims=True);pm=P@OFF
 lookup=np.c_[res,cost,pm,np.sqrt(np.sum(P*(OFF-pm[:,None])**2,1)),ref[:,4],
              np.gradient(ref[:,4]),obs-ref[:,4]]
 blocks["lookup"].append(lookup.astype(np.float32))
 roll=[]
 zobs=(obs-uniform_filter1d(obs,min(201,n),mode="nearest"))
 zref=ref[:,4]-uniform_filter1d(ref[:,4],min(201,n),mode="nearest")
 for win in (51,201,801):
  size=min(win,n);C=uniform_filter1d(cost,size,axis=0,mode="nearest")
  PP=np.exp(np.clip(-(C-C.min(1,keepdims=True))/.4,-35,0));PP/=PP.sum(1,keepdims=True)
  den=np.sqrt(np.maximum(uniform_filter1d(zobs*zobs,size,mode="nearest")*
                         uniform_filter1d(zref*zref,size,mode="nearest"),0))+1e-6
  roll += [PP@OFF,np.sqrt(np.sum(PP*(OFF[None]-(PP@OFF)[:,None])**2,1)),
           uniform_filter1d(zobs*zref,size,mode="nearest")/den,
           np.sqrt(uniform_filter1d((obs-ref[:,4])**2,size,mode="nearest"))]
 blocks["rolling_match"].append(np.column_stack(roll).astype(np.float32))
 if (wi+1)%50==0:print("features",wi+1,flush=True)
blocks={k:np.vstack(v) for k,v in blocks.items()};groups=f.well.to_numpy();folds=list(GroupKFold(5).split(y,groups=groups))
def model(seed):return lgb.LGBMRegressor(objective="huber",n_estimators=360,learning_rate=.035,num_leaves=28,max_depth=8,
 min_child_samples=110,max_bin=127,colsample_bytree=.7,subsample=.85,subsample_freq=1,reg_alpha=2.,reg_lambda=16.,
 verbosity=-1,n_jobs=12,random_state=seed)
def rm(p,i=None):
 if i is None:i=np.arange(len(y))
 return float(np.sqrt(np.mean((p[i]-y[i])**2)))
sets={"basic":["basic"],"basic_future":["basic","future_gr"],"basic_lookup":["basic","lookup"],
      "all_blocks":["basic","future_gr","lookup","rolling_match"]}
if full: sets={"all_blocks":sets["all_blocks"]}
report={};preds={}
for si,(name,ks) in enumerate(sets.items()):
 X=np.column_stack([blocks[k] for k in ks]);p=np.zeros(len(y))
 algos={"lgb":lambda seed:model(seed)}
 if full and "--models-all" in sys.argv:
  algos.update({
   "xgb":lambda seed:xgb.XGBRegressor(objective="reg:squarederror",n_estimators=500,learning_rate=.035,
      max_depth=7,min_child_weight=100,subsample=.8,colsample_bytree=.7,reg_alpha=2,reg_lambda=16,
      tree_method="hist",device="cuda",n_jobs=12,random_state=seed),
   "cat":lambda seed:CatBoostRegressor(loss_function="RMSE",iterations=500,depth=7,learning_rate=.04,
      l2_leaf_reg=16,random_seed=seed,verbose=False,thread_count=12)})
 for aname,factory in algos.items():
  pp=np.zeros(len(y))
  for fold,(tr,va) in enumerate(folds):
   fit=tr[tr%8==fold%8];m=factory(1000*si+fold);m.fit(X[fit],(y-base)[fit]);pp[va]=base[va]+m.predict(X[va])
  key=name if len(algos)==1 else f"{name}_{aname}"
  fg=[rm(base,va)-rm(pp,va) for tr,va in folds];report[key]={"features":X.shape[1],"rmse":rm(pp),"gain":rm(base)-rm(pp),
   "fold_gains":fg,"fold_wins":sum(x>0 for x in fg)};preds[key]=pp;print(key,report[key],flush=True)
summary={"wells":len(keep),"rows":len(y),"base":rm(base),"blocks":{k:v.shape[1] for k,v in blocks.items()},"models":report,
 "protocol":("all 773 wells" if full else "target-independent random 200 wells")+
 "; 5-fold complete-well GKF; stride-8 fit/all suffix score; common schemas only"}
tag="full" if full else "pilot"
(OUT/f"{tag}_summary.json").write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/f"{tag}_oof.npz",y=y,base=base,**preds)
print(json.dumps(summary,indent=2))
