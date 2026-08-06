"""Nested calibrated MoE v2: regret targets and pairwise winner classification."""
from pathlib import Path
import contextlib,io,json,runpy
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/complete_well_moe_v2";OUT.mkdir(parents=True,exist_ok=True)
tab=np.load(ROOT/"exp/results/complete_well_moe/well_table.npz",allow_pickle=True)
X,L,wn,nrow=tab["X"],tab["L"],tab["wn"],tab["nrow"]
names=["accepted_heel_meta","base_five_meta","heel_v4","v4_lgb7",
 "har_physics","har_lgb","har_xgb","pil_blend","v4_poly2","v4_poly3"]

# Recreate exact expert errors solely to evaluate mixtures; targets never enter X.
with contextlib.redirect_stdout(io.StringIO()):
 state=runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
common,y,wells,legs=state["common"],state["y"],state["wells"],state["legs"]
fm=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/full_meta/oof.npz")
vf=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl").iloc[common].reset_index(drop=True)
heelq=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")
heel=legs["v4_lgb7"]+heelq["correction"][common]
def project(p,deg):
 out=p.copy()
 for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False):
  ix=ii.to_numpy();x=vf.d_md.to_numpy(float)[ix]
  x=2*(x-x.min())/max(np.ptp(x),1e-6)-1
  s=p[ix]+vf.d_z.to_numpy(float)[ix]
  out[ix]=np.polyval(np.polyfit(x,s,deg),x)-vf.d_z.to_numpy(float)[ix]
 return out
E=[fm["add"],fm["base"],heel,legs["v4_lgb7"],legs["har_physics"],
 legs["har_lgb"],legs["har_xgb"],legs["pil_blend_oof_postprocessed"],
 project(legs["v4_lgb7"].copy(),2),project(legs["v4_lgb7"].copy(),3)]
P=np.column_stack(E)
wix=[ii.to_numpy() for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False)]
if len(wix)!=len(wn) or any(len(a)!=b for a,b in zip(wix,nrow)):
 raise RuntimeError("well table alignment failed")
G=np.empty((len(wn),len(names),len(names)))
for i,ix in enumerate(wix):
 err=P[ix]-y[ix,None];G[i]=err.T@err/len(ix)

D=L[:,1:]-L[:,[0]]
Tlog=np.sign(D)*np.log1p(abs(D))
winner=np.argmin(L,axis=1)

def train_predict(tr,va,seed):
 kw=dict(iterations=180,depth=4,learning_rate=.04,l2_leaf_reg=20,
  verbose=False,allow_writing_files=False,thread_count=8,random_seed=seed)
 raw=CatBoostRegressor(loss_function="MultiRMSE",**kw)
 raw.fit(X[tr],D[tr],sample_weight=np.sqrt(nrow[tr]))
 log=CatBoostRegressor(loss_function="MultiRMSE",**{**kw,"random_seed":seed+1})
 log.fit(X[tr],Tlog[tr],sample_weight=np.sqrt(nrow[tr]))
 cl=CatBoostClassifier(loss_function="MultiClass",auto_class_weights="SqrtBalanced",
  **{**kw,"random_seed":seed+2})
 cl.fit(X[tr],winner[tr],sample_weight=np.sqrt(nrow[tr]))
 pr=np.c_[np.zeros(len(va)),np.asarray(raw.predict(X[va]))]
 zz=np.asarray(log.predict(X[va]))
 pl=np.c_[np.zeros(len(va)),np.sign(zz)*np.expm1(abs(zz))]
 pc=np.zeros((len(va),len(names)))
 pc[:,np.asarray(cl.classes_,int)]=cl.predict_proba(X[va])
 return {"raw":pr,"log":pl,"class":pc}

def configs():
 out=[]
 for method in ("raw","log"):
  out.append((method,"hard",0))
  for t in (.35,.7,1.2,2.0):out.append((method,"soft",t))
 out += [("class","hard",0),("class","soft",1)]
 return [(m,k,t,b) for m,k,t in out for b in (.1,.2,.3,.5,.7,1.)]

def get_weights(pred,cfg):
 method,kind,temp,blend=cfg;s=pred[method];W=np.zeros_like(s)
 if method=="class":
  if kind=="hard":W[np.arange(len(s)),np.argmax(s,axis=1)]=1
  else:W=s/np.maximum(s.sum(1,keepdims=True),1e-12)
 elif kind=="hard":
  W[np.arange(len(s)),np.argmin(s,axis=1)]=1
 else:
  W=np.exp(np.clip(-(s-s.min(1,keepdims=True))/temp,-20,0));W/=W.sum(1,keepdims=True)
 W*=blend;W[:,0]+=1-blend
 return W

def score(ids,W):
 s=np.einsum("ni,nij,nj->n",W,G[ids],W)
 return float(np.sqrt(np.sum(nrow[ids]*s)/np.sum(nrow[ids])))

outer=list(GroupKFold(5).split(X,groups=wn));finalW=np.zeros((len(wn),len(names)))
fold_rows=[]
for fold,(tr,va) in enumerate(outer):
 # Calibration/model choice is made on cross-fitted predictions of outer-train.
 inner_pred={k:np.zeros((len(tr),len(names))) for k in ("raw","log","class")}
 for inn,(a,b) in enumerate(GroupKFold(4).split(X[tr],groups=wn[tr])):
  z=train_predict(tr[a],tr[b],2000+fold*100+inn*5)
  for k in z:inner_pred[k][b]=z[k]
 best=None
 for cfg in configs():
  sc=score(tr,get_weights(inner_pred,cfg))
  if best is None or sc<best[0]:best=(sc,cfg)
 pred=train_predict(tr,va,3000+fold*20)
 W=get_weights(pred,best[1]);finalW[va]=W
 base=score(va,np.eye(len(names))[np.zeros(len(va),int)])
 fold_rows.append({"fold":fold,"train_calibrated_rmse":best[0],
  "config":list(best[1]),"base":base,"selected":score(va,W),"gain":base-score(va,W)})
 print(fold_rows[-1],flush=True)

baseW=np.zeros_like(finalW);baseW[:,0]=1
pooled=score(np.arange(len(wn)),finalW);base=score(np.arange(len(wn)),baseW)
# Fold-3 shift: standardized feature means and oracle winner distribution.
f3=outer[3][1];rest=outer[3][0]
sd=X[rest].std(0)+1e-6;smd=(X[f3].mean(0)-X[rest].mean(0))/sd
top=np.argsort(abs(smd))[-10:][::-1]
diag={"top_shift_features":[{"index":int(j),"smd":float(smd[j])} for j in top],
 "winner_fold3":np.bincount(winner[f3],minlength=len(names)).tolist(),
 "winner_rest":np.bincount(winner[rest],minlength=len(names)).tolist()}
summary={"wells":len(wn),"accepted_baseline":base,"selector_v2":pooled,
 "gain":base-pooled,"folds":fold_rows,"fold_wins":sum(r["gain"]>0 for r in fold_rows),
 "oracle_family":float(np.sqrt(np.sum(nrow*np.min(L,axis=1)**2)/nrow.sum())),
 "diagnostic_fold3":diag,"accepted":bool(all(r["gain"]>0 for r in fold_rows)),
 "protocol":"outer GKF; inner GKF calibration; regret/log-regret/winner models"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
np.savez_compressed(OUT/"well_weights.npz",weights=finalW,wells=wn,names=np.array(names))
print(json.dumps(summary,indent=2))
