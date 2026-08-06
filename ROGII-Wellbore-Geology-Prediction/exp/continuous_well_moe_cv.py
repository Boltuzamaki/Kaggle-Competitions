"""Continuous per-well convex expert weights learned from legal prefix features."""
from pathlib import Path
import contextlib,io,runpy,json,sys
import numpy as np,pandas as pd,torch
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold,train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/continuous_well_moe";OUT.mkdir(parents=True,exist_ok=True)
c=np.load(ROOT/"exp/results/complete_well_moe/well_table.npz",allow_pickle=True);b=np.load(ROOT/"exp/results/prefix_backtest_moe/backtest_features.npz",allow_pickle=True)
X=np.c_[c["X"],b["features"]].astype(np.float32)
for j in range(X.shape[1]):
 q=X[:,j];q[~np.isfinite(q)]=np.nanmedian(q[np.isfinite(q)]) if np.isfinite(q).any() else 0
wn=c["wn"].astype(str);nrow=c["nrow"];K=10
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
common,y,wells,L=s["common"],s["y"],s["wells"],s["legs"];fm=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/full_meta/oof.npz")
ordered=pd.Series(wells).drop_duplicates().astype(str).to_numpy()
if not np.array_equal(wn,ordered):
 raise RuntimeError("well-table/meta-state ordered identity mismatch")
if not np.array_equal(wn,b["wells"].astype(str)):
 raise RuntimeError("backtest/well-table ordered identity mismatch")
vf=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl").iloc[common].reset_index(drop=True);hq=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")
def project(p,deg):
 out=p.copy()
 for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False):
  ix=ii.to_numpy();x=vf.d_md.to_numpy(float)[ix];x=2*(x-x.min())/max(np.ptp(x),1e-6)-1;s0=p[ix]+vf.d_z.to_numpy(float)[ix]
  out[ix]=np.polyval(np.polyfit(x,s0,deg),x)-vf.d_z.to_numpy(float)[ix]
 return out
heel=L["v4_lgb7"]+hq["correction"][common]
P=np.column_stack([fm["add"],fm["base"],heel,L["v4_lgb7"],L["har_physics"],L["har_lgb"],L["har_xgb"],
 L["pil_blend_oof_postprocessed"],project(L["v4_lgb7"].copy(),2),project(L["v4_lgb7"].copy(),3)])
wix=[ii.to_numpy() for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False)]
G=np.empty((len(wn),K,K))
for i,ix in enumerate(wix):
 e=P[ix]-y[ix,None];G[i]=e.T@e/len(ix)
np.savez_compressed(OUT/"cpu_search_cache.npz",X=X,well=wn,nrow=nrow,G=G)
if "--cache-only" in sys.argv:
 print(json.dumps({"cache":str(OUT/"cpu_search_cache.npz"),"wells":len(wn),
                   "features":X.shape[1],"experts":K},indent=2))
 raise SystemExit(0)
e0=np.zeros(K);e0[0]=1
def oracle(lam):
 W=np.zeros((len(wn),K))
 for i in range(len(wn)):
  fun=lambda w:w@G[i]@w+lam*np.sum((w-e0)**2)
  W[i]=minimize(fun,e0,method="SLSQP",bounds=[(0,1)]*K,constraints={"type":"eq","fun":lambda w:w.sum()-1},
                options={"ftol":1e-10,"maxiter":300}).x
 return W
oracles={str(z):oracle(z) for z in (0,.5,2.)};target=oracles["0.5"]
def normw(W):
 W=np.clip(W,0,None);den=W.sum(1,keepdims=True);bad=den[:,0]<1e-9;W[bad]=e0;return W/W.sum(1,keepdims=True)
def score(W,ids=None):
 if ids is None:ids=np.arange(len(wn))
 z=np.einsum("ni,nij,nj->n",W[ids],G[ids],W[ids]);return float(np.sqrt(np.sum(nrow[ids]*z)/np.sum(nrow[ids])))
baseW=np.tile(e0,(len(wn),1));folds=list(GroupKFold(5).split(X,groups=wn));pred={k:np.zeros_like(target) for k in ("extra","cat","mlp")};fid=np.zeros(len(wn),int)
for fold,(tr,va) in enumerate(folds):
 et=ExtraTreesRegressor(n_estimators=1200,min_samples_leaf=6,max_features=.7,n_jobs=12,random_state=100+fold).fit(X[tr],target[tr],sample_weight=np.sqrt(nrow[tr]))
 pred["extra"][va]=normw(et.predict(X[va]))
 cb=CatBoostRegressor(loss_function="MultiRMSE",iterations=650,depth=5,learning_rate=.03,l2_leaf_reg=20,
  verbose=False,random_seed=200+fold,thread_count=12).fit(X[tr],target[tr],sample_weight=np.sqrt(nrow[tr]))
 pred["cat"][va]=normw(cb.predict(X[va]))
 # End-to-end neural gate with an inner holdout for early stopping.
 it,st=train_test_split(tr,test_size=.2,random_state=300+fold)
 mu=X[it].mean(0);sd=X[it].std(0)+1e-5
 Xt=torch.tensor((X-mu)/sd,dtype=torch.float32);Gt=torch.tensor(G/100,dtype=torch.float32);nw=torch.tensor(nrow/nrow.mean(),dtype=torch.float32)
 net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],64),torch.nn.GELU(),torch.nn.Dropout(.2),
  torch.nn.Linear(64,32),torch.nn.GELU(),torch.nn.Linear(32,K));opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-2)
 best=(1e9,None)
 for ep in range(500):
  net.train();w=torch.softmax(net(Xt[it]),1);loss=(nw[it]*torch.einsum("ni,nij,nj->n",w,Gt[it],w)).mean()+.01*((w-torch.tensor(e0))**2).mean()
  opt.zero_grad();loss.backward();opt.step()
  if ep%10==0:
   net.eval()
   with torch.no_grad():ws=torch.softmax(net(Xt[st]),1);vl=(nw[st]*torch.einsum("ni,nij,nj->n",ws,Gt[st],ws)).mean().item()
   if vl<best[0]:best=(vl,{k:v.clone() for k,v in net.state_dict().items()})
 net.load_state_dict(best[1]);net.eval()
 with torch.no_grad():pred["mlp"][va]=torch.softmax(net(Xt[va]),1).numpy()
 fid[va]=fold;print("fold",fold,flush=True)
report={}
for name,W in pred.items():
 variants={}
 for a in (.1,.25,.5,1.):
  Q=(1-a)*baseW+a*W;fg=[score(baseW,np.where(fid==k)[0])-score(Q,np.where(fid==k)[0]) for k in range(5)]
  variants[str(a)]={"rmse":score(Q),"gain":score(baseW)-score(Q),"fold_gains":fg,"fold_wins":sum(x>0 for x in fg)}
 report[name]=variants
oracle_report={k:score(v) for k,v in oracles.items()}
summary={"wells":len(wn),"features":X.shape[1],"experts":K,"accepted_base":score(baseW),
 "oracle_simplex":oracle_report,"models":report,
 "protocol":"strict outer 5fold by well; oracle weights computed per training well; fixed ET/Cat hyperparameters; MLP inner early stopping and end-to-end Gram loss"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/"oof_weights.npz",wells=wn,target=target,fold=fid,**pred)
print(json.dumps(summary,indent=2))
