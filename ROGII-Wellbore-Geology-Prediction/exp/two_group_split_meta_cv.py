"""Audit physically predeclared two-group rules with strict grouped meta CV."""
from pathlib import Path
import json, joblib, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/two_group_split_meta"
OUT.mkdir(parents=True,exist_ok=True)
GT=pd.read_parquet(ROOT/"exp/public_artifacts/pilkwang/oof/train_gt.parquet")
yfull=GT.target_delta_from_last_known.to_numpy(float); n=len(GT)
hr=pd.read_pickle(ROOT/"exp/results/harshini_cached_xgb/rows.pkl")
bywell={w:q.index.to_numpy() for w,q in GT.groupby("well_id",sort=False)}
hidx=np.empty(len(hr),int)
for w,q in hr.groupby("well",sort=False):hidx[q.index.to_numpy()]=bywell[w]
y=yfull[hidx]; wells=GT.well_id.to_numpy()[hidx]
warm=1-np.exp(-np.maximum(hr.md_since.to_numpy(float),0)/85.)
legs={"har_physics":warm*hr.blend_d.to_numpy(float),
 "har_lgb":warm*np.load(ROOT/"exp/results/harshini_cached_xgb/lgb_fast_oof.npy"),
 "har_xgb":warm*np.load(ROOT/"exp/results/harshini_cached_xgb/xgb_oof.npy")}
sv=joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
for k in ("lgb123","lgb7","xgb","cat"):legs["v4_"+k]=np.asarray(sv[k])[hidx]
pp=ROOT/"exp/public_artifacts/pilkwang/oof"
for k in ("blend_oof_postprocessed","catboost_oof","sequence_tcn_oof","lgb_oof"):
    legs["pil_"+k]=np.asarray(np.load(pp/f"{k}.npy",mmap_mode="r")[hidx])
Z=np.column_stack(list(legs.values()))

# One row of test-available metadata per well.
M=[]
for w in sorted(np.unique(wells)):
    h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
    tp=next(iter((ROOT/"data/train").glob(f"{w}__typewell*.csv")))
    t=pd.read_csv(tp); nv=int(h.TVT_input.notna().sum()); c=nv-1
    dx=h.X.iloc[-1]-h.X.iloc[c]; dy=h.Y.iloc[-1]-h.Y.iloc[c]
    u=(h.TVT_input+h.Z).iloc[:nv].to_numpy(float)
    psl=np.polyfit(h.MD.iloc[max(0,nv-300):nv],u[max(0,nv-300):],1)[0]
    tvsl=np.polyfit(h.MD.iloc[max(0,nv-300):nv],
                    h.TVT_input.iloc[max(0,nv-300):nv],1)[0]
    gr=h.GR.to_numpy(float); forms=set(t.Geology.dropna().astype(str))
    M.append(dict(well=w,dx=dx,dy=dy,major_ew=abs(dx)>=abs(dy),
      prefix_u_slope=psl,prefix_tvt_slope=tvsl,ps_frac=nv/len(h),
      gr_missing=np.mean(~np.isfinite(gr)),gr_std=np.nanstd(gr),
      all6=all(x in forms for x in ("ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA")),
      geology_coverage=sum(x in forms for x in ("ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"))))
M=pd.DataFrame(M).set_index("well")
# Median thresholds are covariate-only and fixed before looking at errors.
rules={
 "azimuth_dx_positive":M.dx>0,
 "azimuth_dy_positive":M.dy>0,
 "azimuth_major_EW":M.major_ew.astype(bool),
 "prefix_U_slope_positive":M.prefix_u_slope>0,
 "prefix_TVT_slope_positive":M.prefix_tvt_slope>0,
 "typewell_all6":M.all6.astype(bool),
 "PS_fraction_high":M.ps_frac>M.ps_frac.median(),
 "GR_variance_high":M.gr_std>M.gr_std.median(),
}
row_rule={k:pd.Series(wells).map(v.astype(bool).to_dict()).to_numpy(bool)
          for k,v in rules.items()}
split=list(GroupKFold(5).split(Z,groups=wells))
def rmse(p,m=None):
    if m is None:m=np.ones(len(y),bool)
    return float(np.sqrt(np.mean((p[m]-y[m])**2)))
def fit_global():
    p=np.zeros(len(y))
    for tr,va in split:
        m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(Z[tr[::8]],y[tr[::8]])
        p[va]=m.predict(Z[va])
    return p
base=fit_global(); results={}
for name,gp in row_rule.items():
    p=np.zeros(len(y))
    for f,(tr,va) in enumerate(split):
        for val in (False,True):
            tr2=tr[gp[tr]==val][::8];va2=va[gp[va]==val]
            if len(np.unique(wells[tr2]))<30:
                tr2=tr[::8]
            m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(Z[tr2],y[tr2])
            p[va2]=m.predict(Z[va2])
    # Error by group for the common global meta.
    results[name]={"false_wells":int((~rules[name]).sum()),
      "true_wells":int(rules[name].sum()),"false_rows":int((~gp).sum()),
      "true_rows":int(gp.sum()),"global_false_rmse":rmse(base,~gp),
      "global_true_rmse":rmse(base,gp),"split_rmse":rmse(p),
      "gain":rmse(base)-rmse(p),
      "fold_gains":[rmse(base,va)-rmse(p,va) for tr,va in split],
      "fold_wins":sum(rmse(p,va)<rmse(base,va) for tr,va in split)}
out={"rows":len(y),"wells":len(M),"global_meta_rmse":rmse(base),
 "rules":results,"best_by_rmse":min(results,key=lambda k:results[k]["split_rmse"]),
 "acceptance":"Require positive pooled gain and 5/5 fold wins; rules are physical/covariate-only, no test IDs."}
(OUT/"summary.json").write_text(json.dumps(out,indent=2))
M.to_csv(OUT/"well_groups.csv")
print(json.dumps(out,indent=2))
