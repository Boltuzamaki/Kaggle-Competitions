"""Identity-audited nested meta-stack across all compatible saved OOF legs."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/meta_all_honest_oof"
OUT.mkdir(parents=True,exist_ok=True)
GT=pd.read_parquet(ROOT/"exp/public_artifacts/pilkwang/oof/train_gt.parquet")
n=len(GT); yfull=GT.target_delta_from_last_known.to_numpy(float)

# Harshini contains 765/773 wells in PF-cache insertion order. Recover exact
# global identity by (well, within-hidden-suffix row), then prove target equality.
hr=pd.read_pickle(ROOT/"exp/results/harshini_cached_xgb/rows.pkl")
bywell={w:q.index.to_numpy() for w,q in GT.groupby("well_id",sort=False)}
hidx=np.empty(len(hr),int)
for w,q in hr.groupby("well",sort=False):
    gi=bywell[w]
    if len(gi)!=len(q): raise ValueError(f"row count mismatch {w}")
    hidx[q.index.to_numpy()]=gi
err=np.max(np.abs(hr.target.to_numpy(float)-yfull[hidx]))
if err>.002: raise ValueError(f"Harshini target identity failed: {err}")
common=hidx
y=yfull[common]; wells=GT.well_id.to_numpy()[common]
warm=1-np.exp(-np.maximum(hr.md_since.to_numpy(float),0)/85.)
legs={
 "har_physics":warm*hr.blend_d.to_numpy(float),
 "har_lgb":warm*np.load(ROOT/"exp/results/harshini_cached_xgb/lgb_fast_oof.npy"),
 "har_xgb":warm*np.load(ROOT/"exp/results/harshini_cached_xgb/xgb_oof.npy"),
}

# Stack V4 and Pilkwang arrays are in exact train_gt order; verify their length.
sv=joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
for k,v in sv.items():
    if len(v)!=n: raise ValueError("Stack V4 length mismatch")
    legs["v4_"+k]=np.asarray(v)[common]
pp=ROOT/"exp/public_artifacts/pilkwang/oof"
for k in ("blend_oof_postprocessed","catboost_oof","sequence_tcn_oof","lgb_oof"):
    a=np.load(pp/f"{k}.npy",mmap_mode="r")
    if len(a)!=n: raise ValueError("Pilkwang length mismatch")
    legs["pil_"+k]=np.asarray(a[common])

# v10 diagnostics are in their own validation ordering: align only by exact ID.
vm=pd.read_csv(ROOT/"exp/public_artifacts/v10/diagnostics/oof_val_meta.csv",
               usecols=["id","target"])
vz=np.load(ROOT/"exp/public_artifacts/v10/diagnostics/oof_val_predictions.npz")
vp=vz["predictions"]; keys=json.loads(
    (ROOT/"exp/public_artifacts/v10/diagnostics/prediction_keys.json").read_text())
id_to_v=pd.Series(np.arange(len(vm)),index=vm.id).to_dict()
vi=np.fromiter((id_to_v[x] for x in GT.id.to_numpy()[common]),int,len(common))
verr=np.max(np.abs(vm.target.to_numpy()[vi]-y))
if verr>.002: raise ValueError(f"v10 target identity failed: {verr}")
for k in ("lgb123","cb123","tabicl_A","tabicl_B"):
    legs["v10_"+k]=vp[vi,keys.index(k)]

Z=np.column_stack(list(legs.values()))
def rmse(p,m=None):
    if m is None: return float(np.sqrt(np.mean((p-y)**2)))
    return float(np.sqrt(np.mean((p[m]-y[m])**2)))

# Nested outer-well meta fit. Fit on stride-8 rows exactly as base tabular models,
# apply to every row of the held-out wells.
fold_id=np.full(len(y),-1,np.int8); pred=np.zeros(len(y))
weights=[]; split=list(GroupKFold(5).split(Z,groups=wells))
for f,(tr,va) in enumerate(split):
    fit=tr[::8]
    model=Ridge(alpha=100,positive=True,fit_intercept=False).fit(Z[fit],y[fit])
    pred[va]=model.predict(Z[va]); fold_id[va]=f
    weights.append(dict(zip(legs,model.coef_.tolist())))

# Ablation: greedily add source families to identify genuine complementarity.
families={
 "har":["har_physics","har_lgb","har_xgb"],
 "pil":["pil_blend_oof_postprocessed","pil_catboost_oof","pil_sequence_tcn_oof","pil_lgb_oof"],
 "v4":["v4_lgb123","v4_lgb7","v4_xgb","v4_cat"],
 "v10":["v10_lgb123","v10_cb123","v10_tabicl_A","v10_tabicl_B"],
}
abl={}
abl_preds={}
for name,cols in {
    "har":families["har"],
    "har_pil":families["har"]+families["pil"],
    "har_pil_v4":families["har"]+families["pil"]+families["v4"],
    "all":list(legs),
}.items():
    ix=[list(legs).index(c) for c in cols]; po=np.zeros(len(y))
    for tr,va in split:
        m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(Z[tr[::8]][:,ix],y[tr[::8]])
        po[va]=m.predict(Z[va][:,ix])
    abl[name]=rmse(po)
    abl_preds[name]=po

# The source-family ablation is itself nested. Promote only the best complete
# source set; this avoids allowing weak v10 legs to degrade the saved artifact.
best_name=min(abl,key=abl.get)
best_pred=abl_preds[best_name]

wr=[]
for w,ii in pd.Series(np.arange(len(y))).groupby(wells):
    jj=ii.to_numpy()
    wr.append({"well":w,"rows":len(jj),"base":rmse(legs["har_physics"],jj),
               "meta":rmse(pred,jj)})
wr=pd.DataFrame(wr)
worst=wr.nlargest(max(1,int(np.ceil(.1*len(wr)))),"base")
summary={
 "rows":len(y),"wells":int(pd.Series(wells).nunique()),
 "identity":{"har_target_max_abs":float(err),"v10_target_max_abs":float(verr)},
 "individual":{k:rmse(v) for k,v in legs.items()},
 "nested_rmse":rmse(pred),"ablation":abl,
 "best_family_set":best_name,"best_nested_rmse":rmse(best_pred),
 "folds":[{"fold":f,"rows":int((fold_id==f).sum()),"rmse":rmse(pred,fold_id==f)}
          for f in range(5)],
 "weights":weights,"win_rate":float((wr.meta<wr.base).mean()),
 "p90_base":float(wr.base.quantile(.9)),"p90_meta":float(wr.meta.quantile(.9)),
 "worst_decile_base":float(np.sqrt(np.average(worst.base**2,weights=worst.rows))),
 "worst_decile_meta":float(np.sqrt(np.average(worst.meta**2,weights=worst.rows))),
}
print(json.dumps(summary,indent=2),flush=True)
np.savez_compressed(OUT/"meta_oof.npz",prediction=pred,target=y,global_indices=common,
                    fold=fold_id)
np.savez_compressed(OUT/"best_meta_oof.npz",prediction=best_pred,target=y,
                    global_indices=common,fold=fold_id)
wr.to_csv(OUT/"well_metrics.csv",index=False)
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
