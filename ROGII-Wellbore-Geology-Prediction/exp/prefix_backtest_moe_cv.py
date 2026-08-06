"""Same-well visible-prefix backtests as legal complete-well MoE features."""
from pathlib import Path
import contextlib,io,json,runpy
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/prefix_backtest_moe";OUT.mkdir(parents=True,exist_ok=True)
cache=np.load(ROOT/"exp/results/complete_well_moe/well_table.npz",allow_pickle=True)
X0,L,wn,nrow=cache["X"],cache["L"],cache["wn"],cache["nrow"]
names=["accepted_heel_meta","base_five_meta","heel_v4","v4_lgb7",
 "har_physics","har_lgb","har_xgb","pil_blend","v4_poly2","v4_poly3"]

def robust_affine(x,y):
 ok=np.isfinite(x+y);x,y=x[ok],y[ok]
 if len(x)<20:return 1.,0.,20.
 A=np.c_[x,np.ones(len(x))];c=np.linalg.lstsq(A,y,rcond=None)[0]
 for _ in range(4):
  r=y-A@c;s=1.4826*np.median(abs(r-np.median(r)))+2
  w=1/np.maximum(1,abs(r)/(2.5*s));c=np.linalg.lstsq(A*w[:,None],y*w,rcond=None)[0]
 return float(np.clip(c[0],.2,4)),float(c[1]),float(s)

def backtest(w):
 h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
 t=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
 vis=h.TVT_input.notna().to_numpy();end=np.flatnonzero(vis)[-1]+1
 md=h.MD.to_numpy(float)[:end];z=h.Z.to_numpy(float)[:end]
 tv=h.TVT_input.to_numpy(float)[:end]
 hs=pd.to_numeric(h.GR.iloc[:end],errors="coerce").interpolate(limit_direction="both")
 hg=hs.fillna(hs.median() if hs.notna().any() else 0).to_numpy(float)
 ts=pd.to_numeric(t.GR,errors="coerce").interpolate(limit_direction="both")
 tg=ts.fillna(ts.median() if ts.notna().any() else 0).to_numpy(float);tt=t.TVT.to_numpy(float)
 out=[]; winners=[]
 for frac in (.35,.50,.65,.78):
  cut=max(50,min(end-40,int(end*frac)));known=np.arange(max(0,cut-1200),cut);q=np.arange(cut,end)
  x=(md[known]-md[cut-1])/max(np.ptp(md[known]),1);xq=(md[q]-md[cut-1])/max(np.ptp(md[known]),1)
  anchor=tv[cut-1]; preds=[]
  preds.append(np.full(len(q),anchor))
  S=tv[known]+z[known]
  for deg in (0,1,2):
   c=np.polyfit(x,S,deg);sp=np.polyval(c,xq)
   p=sp-z[q];p=np.clip(p,anchor-100,anchor+100);preds.append(p)
  # Very local structural line is a separate expert.
  kl=known[-min(250,len(known)):];xl=md[kl]-md[cut-1]
  c=np.polyfit(xl,tv[kl]+z[kl],1);preds.append(np.clip(np.polyval(c,md[q]-md[cut-1])-z[q],anchor-100,anchor+100))
  # Hidden-GR datum posterior around the global structural line.
  a,b,scale=robust_affine(np.interp(tv[known],tt,tg),hg[known])
  base=preds[2];sh=np.arange(-40,41,4.)
  costs=[]
  for s in sh:
   r=(hg[q]-(a*np.interp(base+s,tt,tg)+b))/max(scale,5)
   costs.append(np.mean(np.log1p((r/2)**2)))
  costs=np.asarray(costs);po=np.exp(-(costs-costs.min())/.15);po/=po.sum()
  preds.append(base+.25*(po@sh))
  er=np.array([np.sqrt(np.mean((p-tv[q])**2)) for p in preds])
  # Early/late errors expose drift and curvature behavior.
  mid=len(q)//2
  early=np.array([np.sqrt(np.mean((p[:mid]-tv[q[:mid]])**2)) for p in preds])
  late=np.array([np.sqrt(np.mean((p[mid:]-tv[q[mid:]])**2)) for p in preds])
  out.extend(np.r_[er,early,late,er.min(),np.partition(er,1)[1]-er.min(),
                    np.std(er),po@sh,-np.sum(po*np.log(po+1e-12))])
  winners.append(int(np.argmin(er)))
 out.extend([np.mean(winners),np.std(winners),len(set(winners))])
 return np.asarray(out,np.float32)

BT=[]
for i,w in enumerate(wn):
 BT.append(backtest(str(w)))
 if i%100==0:print("backtests",i,flush=True)
BT=np.nan_to_num(np.asarray(BT),nan=99,posinf=99,neginf=-99)
np.savez_compressed(OUT/"backtest_features.npz",features=BT,wells=wn)
X=np.c_[X0,BT]

# Recreate exact expert error Gram matrices for hard/soft gate evaluation.
with contextlib.redirect_stdout(io.StringIO()):
 state=runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
common,y,wells,legs=state["common"],state["y"],state["wells"],state["legs"]
ordered_meta_wells=pd.Series(wells).drop_duplicates().to_numpy()
if not np.array_equal(wn.astype(str),ordered_meta_wells.astype(str)):
 raise RuntimeError("cached well-table/meta-state ordered well identity failed")
fm=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/full_meta/oof.npz")
vf=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl").iloc[common].reset_index(drop=True)
heelq=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")
def project(p,deg):
 out=p.copy()
 for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False):
  ix=ii.to_numpy();x=vf.d_md.to_numpy(float)[ix];x=2*(x-x.min())/max(np.ptp(x),1e-6)-1
  s=p[ix]+vf.d_z.to_numpy(float)[ix];out[ix]=np.polyval(np.polyfit(x,s,deg),x)-vf.d_z.to_numpy(float)[ix]
 return out
heel=legs["v4_lgb7"]+heelq["correction"][common]
P=np.column_stack([fm["add"],fm["base"],heel,legs["v4_lgb7"],legs["har_physics"],
 legs["har_lgb"],legs["har_xgb"],legs["pil_blend_oof_postprocessed"],
 project(legs["v4_lgb7"].copy(),2),project(legs["v4_lgb7"].copy(),3)])
wix=[ii.to_numpy() for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False)]
G=np.empty((len(wn),len(names),len(names)))
for i,ix in enumerate(wix):
 e=P[ix]-y[ix,None];G[i]=e.T@e/len(ix)

def fit_predict(Xin,tr,va,seed):
 out=np.zeros((len(va),len(names)))
 for j in range(1,len(names)):
  m=CatBoostRegressor(iterations=180,depth=5,learning_rate=.05,loss_function="Huber:delta=1",
   l2_leaf_reg=20,random_seed=seed+j,verbose=False,allow_writing_files=False,thread_count=8)
  m.fit(Xin[tr],L[tr,j]-L[tr,0],sample_weight=np.sqrt(nrow[tr]))
  out[:,j]=m.predict(Xin[va])
 return out
outer=list(GroupKFold(5).split(X,groups=wn))
predL=np.zeros_like(L);pred0=np.zeros_like(L);fold_id=np.full(len(wn),-1)
for fold,(tr,va) in enumerate(outer):
 predL[va]=fit_predict(X,tr,va,4100+fold*20)
 pred0[va]=fit_predict(X0,tr,va,4100+fold*20)
 fold_id[va]=fold;print("outer",fold,flush=True)

def score(W,ids=None):
 if ids is None:ids=np.arange(len(wn))
 s=np.einsum("ni,nij,nj->n",W[ids],G[ids],W[ids])
 return float(np.sqrt(np.sum(nrow[ids]*s)/np.sum(nrow[ids])))
def score_local(W,ids):
 s=np.einsum("ni,nij,nj->n",W,G[ids],W)
 return float(np.sqrt(np.sum(nrow[ids]*s)/np.sum(nrow[ids])))
baseW=np.zeros_like(predL);baseW[:,0]=1;base=score(baseW)
def make_rows(pred):
 rows=[]
 for kind in ("hard","soft"):
  for temp in (.5,1.,2.):
   W=np.zeros_like(pred)
   if kind=="hard":W[np.arange(len(wn)),np.argmin(pred,axis=1)]=1
   else:
    W=np.exp(np.clip(-(pred-pred.min(1,keepdims=True))/temp,-20,0));W/=W.sum(1,keepdims=True)
   for blend in (.05,.1,.15,.2,.3,.5,1):
    Q=blend*W+(1-blend)*baseW
    fg=[score(baseW,np.where(fold_id==k)[0])-score(Q,np.where(fold_id==k)[0]) for k in range(5)]
    rows.append({"kind":kind,"temperature":temp,"blend":blend,"rmse":score(Q),"fold_gains":fg})
 return rows
rows=make_rows(predL);rows0=make_rows(pred0)
best=min(rows,key=lambda z:z["rmse"]);best0=min(rows0,key=lambda z:z["rmse"])
stable=[r for r in rows if all(x>0 for x in r["fold_gains"])]

# Choose temperature/blend using inner OOF predictions from outer-training wells.
nestedW=np.zeros_like(predL);nested_cfg=[]
for fold,(tr,va) in enumerate(outer):
 ip=np.zeros((len(tr),len(names)))
 for inn,(a,b) in enumerate(GroupKFold(4).split(X[tr],groups=wn[tr])):
  ip[b]=fit_predict(X,tr[a],tr[b],7000+fold*100+inn*20)
 local0=np.zeros_like(ip);local0[:,0]=1;cand=[]
 for kind in ("hard","soft"):
  for temp in (.5,1.,2.):
   W=np.zeros_like(ip)
   if kind=="hard":W[np.arange(len(ip)),np.argmin(ip,axis=1)]=1
   else:
    W=np.exp(np.clip(-(ip-ip.min(1,keepdims=True))/temp,-20,0));W/=W.sum(1,keepdims=True)
   for blend in (.05,.1,.15,.2,.3,.5,1):
    Q=blend*W+(1-blend)*local0
    cand.append((score_local(Q,tr),(kind,temp,blend)))
 cfg=min(cand,key=lambda z:z[0])[1];nested_cfg.append(list(cfg))
 kind,temp,blend=cfg;s=predL[va];W=np.zeros_like(s)
 if kind=="hard":W[np.arange(len(s)),np.argmin(s,axis=1)]=1
 else:
  W=np.exp(np.clip(-(s-s.min(1,keepdims=True))/temp,-20,0));W/=W.sum(1,keepdims=True)
 nestedW[va]=blend*W;nestedW[va,0]+=1-blend
nested_fg=[score(baseW,np.where(fold_id==k)[0])-score(nestedW,np.where(fold_id==k)[0]) for k in range(5)]
summary={"wells":len(wn),"features_base":X0.shape[1],"features_backtest":BT.shape[1],
 "accepted_base":base,"best":best,"fold_wins":sum(x>0 for x in best["fold_gains"]),
 "best_stable":min(stable,key=lambda z:z["rmse"]) if stable else None,
 "without_backtests_best":best0,
 "nested":{"rmse":score(nestedW),"fold_gains":nested_fg,"configs":nested_cfg},
 "oracle_family":float(np.sqrt(np.sum(nrow*np.min(L,axis=1)**2)/nrow.sum())),
 "protocol":"visible-prefix pseudo-start backtests; outer whole-well GKF regret models"}
(OUT/"grid.json").write_text(json.dumps(rows,indent=2))
np.savez_compressed(OUT/"selector_oof.npz",predicted_regret=predL,
 predicted_regret_no_backtest=pred0,nested_weights=nestedW,fold=fold_id)
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
