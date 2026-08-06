"""Covariate-only test-domain OOF audit and strict weighted meta CV."""
from pathlib import Path
import contextlib,io,runpy,json,joblib
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/domain_matched_oof";OUT.mkdir(parents=True,exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
legs=dict(s["legs"]);y=s["y"];wells=s["wells"];common=s["common"]
# Additional fully honest legal legs, aligned through the proven global train_gt indices.
legs["heel_global"]=legs["v4_lgb7"]+np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")["correction"][common]
legs["heel_multiscale"]=np.load(ROOT/"exp/results/heel_gr_multiscale/oof.npz")["prediction"][common]
legs["cubic_cat"]=np.load(ROOT/"exp/results/well_cubic_coeff_tabular/oof.npz")["cat"][common]
rq=np.load(ROOT/"exp/results/raw_typewell_feature_blocks/full_oof.npz")
rk="all_blocks_lgb" if "all_blocks_lgb" in rq.files else "all_blocks"
legs["raw_typewell_lgb"]=rq[rk][common]
M=pd.read_csv(ROOT/"exp/results/legal_two_group_exhaustive/well_table.csv").set_index("well")
T=pd.read_csv(ROOT/"exp/results/legal_two_group_exhaustive/test_groups_features.csv").set_index("well")
cols=["dx","dy","dz","md_total","suffix_len","ps_frac","x_ps","y_ps","z_ps","typewell_min",
      "typewell_max","typewell_span","anchor_rel","gr_mean","gr_std","gr_missing","gr_suffix_mean","gr_suffix_std"]
A=M[cols].replace([np.inf,-np.inf],np.nan);B=T[cols].replace([np.inf,-np.inf],np.nan)
med=A.median();A=A.fillna(med);B=B.fillna(med);scale=(A.quantile(.75)-A.quantile(.25)).replace(0,1)
Az=(A-med)/scale;Bz=(B-med)/scale
dist=np.sqrt(((Az.to_numpy()[:,None,:]-Bz.to_numpy()[None,:,:])**2).mean(2)).min(1)
M["test_distance"]=dist
q25,q50=np.quantile(dist,[.25,.5]);well_domain={
 "quadrant_dxneg_dypos":(M.dx<0)&(M.dy>0),
 "nearest_25pct":M.test_distance<=q25,
 "nearest_50pct":M.test_distance<=q50,
 "all":pd.Series(True,index=M.index)}
row_masks={k:pd.Series(wells).map(v.to_dict()).fillna(False).to_numpy(bool) for k,v in well_domain.items()}
def rm(p,mask):return float(np.sqrt(np.mean((p[mask]-y[mask])**2)))
scores={}
for name,p in legs.items():
 scores[name]={d:{"rmse":rm(p,m),"rows":int(m.sum()),"wells":int(np.unique(wells[m]).size)}
                    for d,m in row_masks.items()}
# Exact accepted five/global-heel blends and domain-weighted variants.
base_names=["har_physics","har_lgb","har_xgb","pil_blend_oof_postprocessed","v4_lgb7"]
Z5=np.column_stack([legs[k] for k in base_names]);Z6=np.c_[Z5,legs["heel_global"]]
splits=list(GroupKFold(5).split(Z5,groups=wells))
rowdist=pd.Series(wells).map(M.test_distance.to_dict()).to_numpy(float);tau=float(np.median(dist))
quad=row_masks["quadrant_dxneg_dypos"]
weight_sets={"unweighted":np.ones(len(y)),
 "distance":np.exp(-.5*(rowdist/tau)**2),
 "distance_plus_direction":np.exp(-.5*(rowdist/tau)**2)*(1+3*quad)}
blendpred={};blend_report={}
for wn,sw in weight_sets.items():
 p=np.zeros(len(y));co=[]
 for tr,va in splits:
  fit=tr[::8];m=Ridge(alpha=100,positive=True,fit_intercept=False)
  m.fit(Z6[fit],y[fit],sample_weight=sw[fit]);p[va]=m.predict(Z6[va]);co.append(m.coef_.tolist())
 blendpred[wn]=p;blend_report[wn]={"domains":{d:rm(p,m) for d,m in row_masks.items()},"weights":co}
scores["accepted_heel_meta"]=blend_report["unweighted"]["domains"]
# Fold gains of weighted models against unweighted inside each requested domain.
for wn in ("distance","distance_plus_direction"):
 blend_report[wn]["fold_domain_gains"]={}
 for d,dm in row_masks.items():
  vals=[]
  for tr,va in splits:
   m=dm.copy();keep=np.zeros(len(y),bool);keep[va]=True;m &= keep
   vals.append(rm(blendpred["unweighted"],m)-rm(blendpred[wn],m) if m.any() else None)
  blend_report[wn]["fold_domain_gains"][d]=vals
summary={"rows":len(y),"wells":int(np.unique(wells).size),"test_wells":list(T.index),
 "distance_features":cols,"distance_thresholds":{"q25":float(q25),"q50":float(q50),"tau":tau},
 "domain_well_counts":{k:int(v.sum()) for k,v in well_domain.items()},
 "leg_scores":scores,"weighted_meta":blend_report,
 "best_by_domain":{d:min(legs,key=lambda k:scores[k][d]["rmse"]) for d in row_masks},
 "acceptance":"No test targets or IDs in fitting; test raw covariates define distance only; every meta fold fits weights on outer-train targets."}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));M.to_csv(OUT/"train_domain_features.csv")
print(json.dumps({"counts":summary["domain_well_counts"],"best":summary["best_by_domain"],
 "best_scores":{d:{k:scores[k][d] for k in [summary["best_by_domain"][d]]} for d in row_masks},
 "weighted":{k:v["domains"] for k,v in blend_report.items()}},indent=2))
