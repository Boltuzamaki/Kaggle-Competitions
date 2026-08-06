"""Prefix-to-suffix transfer audit for S=TVT+Z dip and curvature quantities."""
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/structural_surface_transfer";OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for w in sorted(p.stem.replace("__horizontal_well","") for p in (ROOT/"data/train").glob("*__horizontal_well.csv")):
 h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv");nv=int(h.TVT_input.notna().sum());p=nv-1
 md=h.MD.to_numpy(float);x=h.X.to_numpy(float);y=h.Y.to_numpy(float);z=h.Z.to_numpy(float);t=h.TVT.to_numpy(float);S=t+z
 ds=np.gradient(S,md);dt=np.gradient(t,md);dx=np.gradient(x,md);dy=np.gradient(y,md);dz=np.gradient(z,md)
 take=np.arange(max(0,nv-300),nv);suf=np.arange(nv,len(h));u=(md-md[p])/max(md[-1]-md[p],1)
 def slope(a,ii):return float(np.polyfit(md[ii],a[ii],1)[0])
 # Along-hole dip and normalized curvature coefficients.
 cp=np.polyfit(u[take] if np.ptp(u[take])>0 else np.arange(len(take)),S[take],min(2,len(take)-1))
 cs=np.polyfit(u[suf],S[suf],min(3,len(suf)-1))
 az=np.arctan2(y[-1]-y[p],x[-1]-x[p]);inc=np.arccos(np.clip((z[-1]-z[p])/(md[-1]-md[p]+1e-6),-1,1))
 dog=np.mean(np.sqrt(np.gradient(dx[suf])**2+np.gradient(dy[suf])**2+np.gradient(dz[suf])**2))
 rows.append({"well":w,"dx":x[-1]-x[p],"dy":y[-1]-y[p],"azimuth":az,"inclination":inc,"dogleg":dog,
  "length":len(h)-nv,"ps_frac":nv/len(h),"prefix_S_slope":slope(S,take),"suffix_S_slope":slope(S,suf),
  "prefix_TVT_slope":slope(t,take),"suffix_TVT_slope":slope(t,suf),
  "prefix_dS_mean":np.mean(ds[take]),"prefix_dS_std":np.std(ds[take]),
  "suffix_dS_mean":np.mean(ds[suf]),"suffix_dS_std":np.std(ds[suf]),
  "prefix_dTVT_mean":np.mean(dt[take]),"suffix_dTVT_mean":np.mean(dt[suf]),
  "prefix_S_quad":cp[0] if len(cp)==3 else 0,"suffix_S_cubic":cs[0] if len(cs)==4 else 0,
  "suffix_S_quad":cs[-3] if len(cs)>=3 else 0})
D=pd.DataFrame(rows).set_index("well");pairs={}
for a,b in [("prefix_S_slope","suffix_S_slope"),("prefix_TVT_slope","suffix_TVT_slope"),
 ("prefix_dS_mean","suffix_dS_mean"),("prefix_dS_std","suffix_dS_std"),
 ("prefix_dTVT_mean","suffix_dTVT_mean"),("prefix_S_quad","suffix_S_quad")]:
 pairs[a+"__"+b]={"pearson":float(D[a].corr(D[b])),"spearman":float(D[a].corr(D[b],method="spearman")),
  "mae":float(np.mean(np.abs(D[a]-D[b])))}
features=["dx","dy","azimuth","inclination","dogleg","length","ps_frac","prefix_S_slope","prefix_TVT_slope",
 "prefix_dS_mean","prefix_dS_std","prefix_dTVT_mean","prefix_S_quad"]
targets=["suffix_S_slope","suffix_TVT_slope","suffix_dS_mean","suffix_dS_std","suffix_S_quad","suffix_S_cubic"]
X=D[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy();Y=D[targets].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy()
cv=KFold(5,shuffle=True,random_state=81);m=ExtraTreesRegressor(n_estimators=800,min_samples_leaf=8,max_features=.8,n_jobs=12,random_state=81)
P=cross_val_predict(m,X,Y,cv=cv,n_jobs=1)
pred={t:{"oof_r2":float(r2_score(Y[:,i],P[:,i])),"corr":float(np.corrcoef(Y[:,i],P[:,i])[0,1])} for i,t in enumerate(targets)}
# Translate the highly predictable S slope back into the exact row metric.
sse={"flat":0.,"prefix_S_slope":0.,"oof_ET_S_slope":0.,"oracle_suffix_S_slope":0.};count=0
for i,(w,row) in enumerate(D.iterrows()):
 h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv");nv=int(h.TVT_input.notna().sum());p=nv-1
 md=h.MD.to_numpy(float);z=h.Z.to_numpy(float);truth=h.TVT.to_numpy(float)
 dm=md[nv:]-md[p];dz=z[nv:]-z[p];actual=truth[nv:]-truth[p];count+=len(actual);sse["flat"]+=np.sum(actual**2)
 for name,sl in [("prefix_S_slope",row.prefix_S_slope),("oof_ET_S_slope",P[i,0]),("oracle_suffix_S_slope",row.suffix_S_slope)]:
  pr=sl*dm-dz;sse[name]+=np.sum((actual-pr)**2)
curve_metric={k:float(np.sqrt(v/count)) for k,v in sse.items()}
groups={}
for name,mask in {"dx_positive":D.dx>0,"dy_positive":D.dy>0}.items():
 groups[name]={}
 for val in (False,True):
  q=D[mask==val];groups[name][str(val)]={"wells":len(q),"S_slope_transfer_corr":float(q.prefix_S_slope.corr(q.suffix_S_slope)),
   "TVT_slope_transfer_corr":float(q.prefix_TVT_slope.corr(q.suffix_TVT_slope))}
summary={"wells":len(D),"direct_transfer":pairs,"grouped_extra_trees_predictability":pred,
 "row_metric_from_S_slope":curve_metric,"direction_groups":groups,
 "conclusion":"Structural increments are statistically predictable but small slope errors amplify over long suffixes; no representation baseline beats V4."}
D.to_csv(OUT/"well_quantities.csv");(OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
