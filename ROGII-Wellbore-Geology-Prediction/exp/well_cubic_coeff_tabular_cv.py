"""One-row-per-well legal tabular prediction of cubic V4 residual curves."""
from pathlib import Path
import json,joblib
import numpy as np,pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/well_cubic_coeff_tabular";OUT.mkdir(parents=True,exist_ok=True)
f=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl");base=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"],float)
y=f.target.to_numpy(float);OFF=np.arange(-40,41,4,dtype=float)
def affine(x,y):
 ok=np.isfinite(x)&np.isfinite(y);x,y=x[ok],y[ok];X=np.c_[x,np.ones(len(x))];c=np.linalg.lstsq(X,y,rcond=None)[0]
 for _ in range(4):
  r=y-X@c;sc=1.4826*np.median(np.abs(r-np.median(r)))+1e-3;w=1/np.maximum(1,np.abs(r)/(2.5*sc));c=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]
 return c[0],c[1],sc,float(np.sqrt(np.mean((y-X@c)**2)))
def sig(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 if not len(v):return [0.]*17
 q=np.quantile(v,[0,.05,.25,.5,.75,.95,1]);out=[*q,np.mean(v),np.std(v)]
 z=v-np.mean(v);den=np.dot(z,z)+1e-9
 for lag in (1,10,50,200):out.append(float(np.dot(z[:-lag],z[lag:])/den) if len(z)>lag else 0)
 pow=np.abs(np.fft.rfft(z))**2;tot=pow.sum()+1e-9;n=len(pow)
 for a,b in ((0,.02),(.02,.08),(.08,.25),(.25,1)):out.append(float(pow[int(a*n):max(int(a*n)+1,int(b*n))].sum()/tot))
 return out
rows=[];targets=[];items=[]
for wi,(w,ix0) in enumerate(f.groupby("well",sort=False).indices.items()):
 ix=np.asarray(ix0);q=f.iloc[ix];h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv");t=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
 tv=t.TVT.to_numpy(float);tg=pd.to_numeric(t.GR,errors="coerce").interpolate(limit_direction="both").to_numpy(float)
 hg=pd.to_numeric(h.GR,errors="coerce").interpolate(limit_direction="both").to_numpy(float);vis=h.TVT_input.notna().to_numpy()
 a,b,sc,calrm=affine(np.interp(h.TVT_input[vis],tv,tg),hg[vis]);rr=np.array([int(z.rsplit("_",1)[1]) for z in q.id],int)
 path=q.last_known_tvt.to_numpy(float)+base[ix];ref=a*np.vstack([np.interp(path+s,tv,tg) for s in OFF]).T+b
 cost=np.log1p(((((hg[rr,None]-ref)/max(sc,5.))/2.))**2)
 feat=[a,b,sc,calrm,len(h),vis.sum(),len(ix),vis.mean()]
 feat += sig(hg)+sig(hg[rr])+sig(tg)
 # Global cost vector and posterior moments.
 gc=cost.mean(0);feat += gc.tolist()
 for temp in (.1,.2,.4):
  P=np.exp(np.clip(-(gc-gc.min())/temp,-35,0));P/=P.sum();m=P@OFF
  feat += [m,np.sqrt(P@(OFF-m)**2),-np.sum(P*np.log(P+1e-9))]
 # Rolling posterior trajectories summarize locally inferred curvature.
 for win in (101,301,801):
  C=uniform_filter1d(cost,size=min(win,len(ix)),axis=0,mode="nearest")
  P=np.exp(np.clip(-(C-C.min(1,keepdims=True))/.4,-35,0));P/=P.sum(1,keepdims=True);m=P@OFF
  feat += [*np.quantile(m,[0,.1,.25,.5,.75,.9,1]),np.mean(m),np.std(m),
           *np.polynomial.legendre.legfit(np.linspace(-1,1,len(m)),m,3)]
 # Existing V4 curve and legal trajectory summaries.
 vd=base[ix];feat += [*np.polynomial.legendre.legfit(np.linspace(-1,1,len(ix)),vd,3),*sig(vd)]
 md=h.MD.to_numpy(float);xx=h.X.to_numpy(float);yy=h.Y.to_numpy(float);zz=h.Z.to_numpy(float);p=np.flatnonzero(vis)[-1]
 feat += [md[0],md[p],md[-1],md[-1]-md[p],xx[p],yy[p],zz[p],xx[-1]-xx[p],yy[-1]-yy[p],zz[-1]-zz[p],
          np.hypot(xx[-1]-xx[p],yy[-1]-yy[p]),np.arctan2(yy[-1]-yy[p],xx[-1]-xx[p]),tv.min(),tv.max(),np.ptp(tv)]
 resid=y[ix]-base[ix];coef=np.polynomial.legendre.legfit(np.linspace(-1,1,len(ix)),resid,3)
 rows.append(feat);targets.append(coef);items.append((w,ix))
 if (wi+1)%100==0:print("features",wi+1,flush=True)
X=np.asarray(rows,np.float32);Y=np.asarray(targets,np.float32);groups=np.array([x[0] for x in items])
folds=list(GroupKFold(5).split(X,groups=groups));pred={k:np.zeros_like(Y) for k in ("ridge","extra","cat")}
for fold,(tr,va) in enumerate(folds):
 ym=Y[tr].mean(0);ys=Y[tr].std(0)+1e-4;T=(Y[tr]-ym)/ys
 models={
  "ridge":make_pipeline(SimpleImputer(),StandardScaler(),Ridge(alpha=100)),
  "extra":make_pipeline(SimpleImputer(),ExtraTreesRegressor(n_estimators=900,min_samples_leaf=5,max_features=.7,n_jobs=12,random_state=fold)),
  "cat":CatBoostRegressor(loss_function="MultiRMSE",iterations=700,depth=5,learning_rate=.035,l2_leaf_reg=12,verbose=False,random_seed=fold,thread_count=12)}
 for name,m in models.items():
  m.fit(X[tr],T);pred[name][va]=m.predict(X[va])*ys+ym
 print("fold",fold,flush=True)
def reconstruct(C):
 p=base.copy()
 for i,(_,ix) in enumerate(items):p[ix]+=np.polynomial.legendre.legval(np.linspace(-1,1,len(ix)),C[i])
 return p
def rm(p,ix=None):
 if ix is None:ix=np.arange(len(y))
 return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))
allpred={k:reconstruct(v) for k,v in pred.items()};pred["ensemble"]=(pred["extra"]+pred["cat"]+pred["ridge"])/3;allpred["ensemble"]=reconstruct(pred["ensemble"])
report={}
for name,p in allpred.items():
 fg=[rm(base,np.concatenate([items[i][1] for i in va]))-rm(p,np.concatenate([items[i][1] for i in va])) for tr,va in folds]
 grid=[{"shrink":float(s),"rmse":rm(base+s*(p-base))} for s in np.linspace(0,1,21)]
 report[name]={"rmse":rm(p),"gain":rm(base)-rm(p),"fold_gains":fg,"fold_wins":sum(z>0 for z in fg),"best_shrink":min(grid,key=lambda z:z["rmse"])}
oracle=reconstruct(Y)
summary={"wells":len(items),"features":X.shape[1],"base":rm(base),"cubic_oracle":rm(oracle),"models":report,
 "protocol":"5-fold complete-well GroupKFold; only common horizontal MD/XYZ/GR/TVT_input and typewell TVT/GR plus honest V4 OOF"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/"oof.npz",y=y,base=base,**allpred)
np.savez_compressed(OUT/"well_table.npz",X=X,Y=Y,wells=groups,
                    ridge_coef=pred["ridge"],extra_coef=pred["extra"],
                    cat_coef=pred["cat"],ensemble_coef=pred["ensemble"])
print(json.dumps(summary,indent=2))
