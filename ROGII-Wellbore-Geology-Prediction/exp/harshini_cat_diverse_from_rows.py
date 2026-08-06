"""Diverse CatBoost OOF legs and tail-risk audit on cached Harshini rows."""
from pathlib import Path
import json, os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/harshini_cached_xgb"
d=pd.read_pickle(OUT/"rows.pkl")
features=[c for c in d if c not in ("well","flat","target")]
X=d[features].to_numpy(np.float32); y=d.target.to_numpy(np.float32); g=d.well.to_numpy()
sub=np.arange(len(d))%8==0
fold_id=np.full(len(d),-1,np.int8)

configs=[
    dict(name="cat_d8",iterations=1400,depth=8,learning_rate=.035,
         l2_leaf_reg=5.,random_strength=.5,bagging_temperature=.5,random_seed=707),
    dict(name="cat_d10",iterations=1000,depth=10,learning_rate=.035,
         l2_leaf_reg=12.,random_strength=1.25,bagging_temperature=1.,random_seed=1707),
]
oofs={q["name"]:np.zeros(len(d),np.float32) for q in configs}
splits=list(GroupKFold(5).split(X[sub],y[sub],g[sub]))
for fold,(tr,va) in enumerate(splits):
    vw=set(g[sub][va]); vm=np.fromiter((w in vw for w in g),bool,len(g))
    fold_id[vm]=fold
    for q in configs:
        kw={k:v for k,v in q.items() if k!="name"}
        m=CatBoostRegressor(
            **kw,loss_function="RMSE",task_type="GPU",devices="0",
            border_count=128,verbose=False,allow_writing_files=False)
        m.fit(X[sub][tr],y[sub][tr])
        oofs[q["name"]][vm]=m.predict(X[vm])
        print("fold",fold,q["name"],"done",flush=True)

warm=1-np.exp(-np.maximum(d.md_since.to_numpy(),0)/85.)
legs={"physics":d.blend_d.to_numpy(float),
      "lgb_fast":np.load(OUT/"lgb_fast_oof.npy"),
      "xgb":np.load(OUT/"xgb_oof.npy"),**oofs}
def rmse(p,mask=None):
    if mask is None: mask=np.ones(len(y),bool)
    return float(np.sqrt(np.mean((warm[mask]*p[mask]-y[mask])**2)))
Z=np.column_stack(list(legs.values()))
ridge=Ridge(alpha=1,positive=True,fit_intercept=False).fit(Z,y)
final=ridge.predict(Z)

folds=[]
for f in range(5):
    m=fold_id==f
    folds.append({"fold":f,"rows":int(m.sum()),"wells":int(pd.Series(g[m]).nunique()),
                  "physics":rmse(legs["physics"],m),"final":rmse(final,m)})
well_rows=[]
for w,ix in pd.Series(np.arange(len(d))).groupby(g):
    ii=ix.to_numpy(); e0=rmse(legs["physics"],ii); e1=rmse(final,ii)
    well_rows.append({"well":w,"rows":len(ii),"physics":e0,"final":e1})
wr=pd.DataFrame(well_rows)
worst=wr.nlargest(max(1,int(np.ceil(.1*len(wr)))),"physics")
summary={
    "individual":{k:rmse(v) for k,v in legs.items()},
    "ridge_weights":dict(zip(legs,ridge.coef_.tolist())),
    "ridge_rmse":rmse(final),
    "folds":folds,
    "well_win_rate":float((wr.final<wr.physics).mean()),
    "p90_physics":float(wr.physics.quantile(.9)),
    "p90_final":float(wr.final.quantile(.9)),
    "worst_decile_physics":float(np.sqrt(np.average(worst.physics**2,weights=worst.rows))),
    "worst_decile_final":float(np.sqrt(np.average(worst.final**2,weights=worst.rows))),
}
print(json.dumps(summary,indent=2),flush=True)
for k,v in oofs.items(): np.save(OUT/f"{k}_oof.npy",v)
np.save(OUT/"stack_cat_final_oof.npy",final)
wr.to_csv(OUT/"stack_cat_wells.csv",index=False)
joblib.dump({"features":features,"ridge":ridge,"columns":list(legs)},
            OUT/"stack_cat_meta.joblib")
(OUT/"stack_cat_summary.json").write_text(json.dumps(summary,indent=2))
