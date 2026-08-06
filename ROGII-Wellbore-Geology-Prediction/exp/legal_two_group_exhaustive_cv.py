"""Inference-legal two-group search and strict group-interaction meta CV."""
from pathlib import Path
import json,runpy,contextlib,io,joblib
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/legal_two_group_exhaustive";OUT.mkdir(parents=True,exist_ok=True)
def one(split,w):
 h=pd.read_csv(ROOT/f"data/{split}/{w}__horizontal_well.csv");t=pd.read_csv(ROOT/f"data/{split}/{w}__typewell.csv").sort_values("TVT")
 nv=int(h.TVT_input.notna().sum());p=nv-1;dx=h.X.iloc[-1]-h.X.iloc[p];dy=h.Y.iloc[-1]-h.Y.iloc[p];dz=h.Z.iloc[-1]-h.Z.iloc[p]
 gr=pd.to_numeric(h.GR,errors="coerce");tv=t.TVT.to_numpy(float);flat=float(h.TVT_input.iloc[p])
 return {"well":w,"dx":dx,"dy":dy,"dz":dz,"azimuth":np.arctan2(dy,dx),"major_ew":abs(dx)>=abs(dy),
  "same_sign_xy":dx*dy>0,"md_total":h.MD.iloc[-1]-h.MD.iloc[0],"suffix_len":len(h)-nv,"ps_frac":nv/len(h),
  "x_ps":h.X.iloc[p],"y_ps":h.Y.iloc[p],"z_ps":h.Z.iloc[p],"x_end":h.X.iloc[-1],"y_end":h.Y.iloc[-1],
  "radial_dot":np.nan,"typewell_min":np.min(tv),"typewell_max":np.max(tv),"typewell_span":np.ptp(tv),
  "anchor_rel":(flat-np.min(tv))/(np.ptp(tv)+1e-6),"gr_mean":gr.mean(),"gr_std":gr.std(),
  "gr_missing":gr.isna().mean(),"gr_suffix_mean":gr.iloc[nv:].mean(),"gr_suffix_std":gr.iloc[nv:].std()}
train_w=sorted(p.stem.replace("__horizontal_well","") for p in (ROOT/"data/train").glob("*__horizontal_well.csv"))
M=pd.DataFrame([one("train",w) for w in train_w]).set_index("well")
cx,cy=M.x_ps.median(),M.y_ps.median();M["radial_dot"]=(M.x_ps-cx)*M.dx+(M.y_ps-cy)*M.dy
test_w=sorted(p.stem.replace("__horizontal_well","") for p in (ROOT/"data/test").glob("*__horizontal_well.csv"))
T=pd.DataFrame([one("test",w) for w in test_w]).set_index("well");T["radial_dot"]=(T.x_ps-cx)*T.dx+(T.y_ps-cy)*T.dy
rules={"dx_positive":(M.dx>0,T.dx>0,None),"dy_positive":(M.dy>0,T.dy>0,None),
 "dz_positive":(M.dz>0,T.dz>0,None),"major_EW":(M.major_ew.astype(bool),T.major_ew.astype(bool),None),
 "same_sign_xy":(M.same_sign_xy.astype(bool),T.same_sign_xy.astype(bool),None),
 "radial_outward":(M.radial_dot>0,T.radial_dot>0,0.)}
bimodal={}
for col in ("md_total","suffix_len","ps_frac","x_ps","y_ps","z_ps","typewell_min","typewell_max",
            "typewell_span","anchor_rel","gr_mean","gr_std","gr_missing","gr_suffix_mean","gr_suffix_std"):
 v=M[col].fillna(M[col].median()).to_numpy(float);km=KMeans(2,n_init=20,random_state=81).fit(v[:,None])
 centers=np.sort(km.cluster_centers_.ravel());thr=float(centers.mean());a=v<=thr
 # Separation is covariate-only; it ranks genuinely bimodal candidates.
 sep=float(abs(v[a].mean()-v[~a].mean())/np.sqrt(.5*(v[a].var()+v[~a].var())+1e-9))
 bal=float(min(a.mean(),1-a.mean()))
 rules[f"{col}_high"]=(M[col]>thr,T[col]>thr,thr);bimodal[col]={"threshold":thr,"separation":sep,"minor_fraction":bal}
# Load exact accepted six legal legs on the 765-well Harshini-compatible universe.
with contextlib.redirect_stdout(io.StringIO()):state=runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
L,y,wells=state["legs"],state["y"],state["wells"];base_names=["har_physics","har_lgb","har_xgb","pil_blend_oof_postprocessed","v4_lgb7"]
heel=L["v4_lgb7"]+np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")["correction"][state["common"]]
Z=np.column_stack([*[L[k] for k in base_names],heel]);splits=list(GroupKFold(5).split(Z,groups=wells))
def rm(p,ix=None):
 if ix is None:ix=np.arange(len(y))
 return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))
base=np.zeros(len(y))
for tr,va in splits:
 m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(Z[tr[::8]],y[tr[::8]]);base[va]=m.predict(Z[va])
results={}
for name,(rw,rt,thr) in rules.items():
 row=pd.Series(wells).map(rw.astype(bool).to_dict()).to_numpy(bool);p=np.zeros(len(y))
 for tr,va in splits:
  # Conservative group interaction: separate coefficients, fallback if tiny.
  for val in (False,True):
   fit=tr[row[tr]==val][::8];vv=va[row[va]==val]
   if len(np.unique(wells[fit]))<30:fit=tr[::8]
   m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(Z[fit],y[fit]);p[vv]=m.predict(Z[vv])
 fg=[rm(base,va)-rm(p,va) for tr,va in splits]
 results[name]={"threshold":thr,"true_wells":int(rw.sum()),"false_wells":int((~rw).sum()),
  "base_false_rmse":rm(base,np.array([not x for x in row])),"base_true_rmse":rm(base,row),
  "split_rmse":rm(p),"gain":rm(base)-rm(p),"fold_gains":fg,"fold_wins":sum(x>0 for x in fg),
  "test_groups":{w:bool(rt.loc[w]) for w in T.index}}
# Attach honest per-well errors for diagnostics, never rule construction.
diag=[]
for w,ix0 in pd.Series(np.arange(len(y))).groupby(wells):
 ix=ix0.to_numpy();diag.append({"well":w,"rows":len(ix),"meta_rmse":rm(base,ix),"mean_residual":float(np.mean(y[ix]-base[ix]))})
D=pd.DataFrame(diag).set_index("well");M.join(D).to_csv(OUT/"well_table.csv")
summary={"train_wells":len(M),"meta_wells":len(np.unique(wells)),"test_wells":len(T),"baseline":rm(base),
 "bimodality":bimodal,"rules":results,"best":min(results,key=lambda k:results[k]["split_rmse"]),
 "acceptance":"positive gain, 5/5 folds, and <7 RMSE; all thresholds covariate-only"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));T.to_csv(OUT/"test_groups_features.csv")
print(json.dumps({"baseline":rm(base),"best":summary["best"],"best_result":results[summary["best"]]},indent=2))
