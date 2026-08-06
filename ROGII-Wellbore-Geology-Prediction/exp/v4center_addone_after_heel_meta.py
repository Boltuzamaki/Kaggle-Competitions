"""Strict grouped add-one test of V4-centered matcher after accepted heel leg."""
from pathlib import Path
import contextlib, io, json, os, runpy
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
variant=os.environ.get("ALIGNMENT_VARIANT","v4center")
OUT=ROOT/f"exp/results/{variant}_addone_after_heel_meta"; OUT.mkdir(parents=True,exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    s=runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
legs,y,wells,common=s["legs"],s["y"],s["wells"],s["common"]
names=["har_physics","har_lgb","har_xgb","pil_blend_oof_postprocessed","v4_lgb7"]
heel=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")
unet=np.load(ROOT/f"exp/results/alignment_unet_{variant}/oof_delta.npz")
heel_leg=legs["v4_lgb7"]+heel["correction"][common].astype(float)
unet_leg=unet["prediction"][common].astype(float)
Z=np.column_stack([*[legs[k] for k in names],heel_leg,unet_leg])
splits=list(GroupKFold(5).split(Z,groups=wells))

def fit(ix):
    p=np.zeros(len(y)); weights=[]
    for tr,va in splits:
        m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(Z[tr[::8]][:,ix],y[tr[::8]])
        p[va]=m.predict(Z[va][:,ix]); weights.append(m.coef_.tolist())
    return p,weights
def rmse(p,ix=None):
    if ix is None: ix=np.arange(len(y))
    return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))

base,bw=fit(list(range(6))); add,aw=fit(list(range(7)))
fold_gains=[rmse(base,va)-rmse(add,va) for _,va in splits]
wr=[]
for w,ii in pd.Series(np.arange(len(y))).groupby(wells):
    jj=ii.to_numpy(); wr.append({"well":w,"rows":len(jj),"base":rmse(base,jj),"add":rmse(add,jj)})
wr=pd.DataFrame(wr); worst=wr.nlargest(max(1,int(np.ceil(.1*len(wr)))),"base")
summary={"rows":len(y),"wells":int(pd.Series(wells).nunique()),
 "base_names":names+["heel"],"add_name":f"{variant}_unet",
 "base_rmse":rmse(base),"add_rmse":rmse(add),"gain":rmse(base)-rmse(add),
 "fold_gains":fold_gains,"fold_wins":int(sum(x>0 for x in fold_gains)),
 "base_weights":bw,"add_weights":aw,
 "unet_weights":[x[-1] for x in aw],
 "well_win_rate":float((wr["add"]<wr.base).mean()),
 "p90_base":float(wr.base.quantile(.9)),"p90_add":float(wr["add"].quantile(.9)),
 "worst_decile_base":float(np.sqrt(np.average(worst.base**2,weights=worst.rows))),
 "worst_decile_add":float(np.sqrt(np.average(worst["add"]**2,weights=worst.rows)))}
np.savez_compressed(OUT/"oof.npz",base=base,add=add,y=y,global_indices=common)
wr.to_csv(OUT/"well_metrics.csv",index=False)
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
