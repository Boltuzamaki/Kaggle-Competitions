"""Package the small Harshini per-well trust model used at test inference."""
from pathlib import Path
import hashlib,json,pickle
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/"exp/results/harshini_fast_model_package"
CACHE=pickle.load(open(ROOT/"exp/results/harshini_kernel_output/imp_cache.pkl","rb"))
SH0,CL0=.6,80
def agg(d,pw=2):
    cs=[c for c in d["P"] if c in d["PR"]]
    M=np.stack([d["P"][c] for c in cs])
    wt=np.array([1/(d["PR"][c]**pw+1e-6) for c in cs]); wt/=wt.sum()
    return (M*wt[:,None]).sum(0)
rows=[]
for w,d in CACHE.items():
    dl=np.clip(agg(d)-d["flat"],-CL0,CL0); tr=d["true"]-d["flat"]
    den=float(np.sum(dl*dl)); wo=float(np.clip(np.sum(dl*tr)/den,0,1.5)) if den>1e-9 else 0
    prs=np.array(list(d["PR"].values())); M=np.stack(list(d["P"].values()))
    rows.append(dict(well=w,w_opt=wo,pr_min=prs.min(),pr_med=np.median(prs),
      pr_max=prs.max(),pr_std=prs.std(),nn=d["nn"],dl_absmean=np.abs(dl).mean(),
      dl_absmax=np.abs(dl).max(),dl_std=dl.std(),dl_end=abs(dl[-1]),
      dl_slope=np.polyfit(np.linspace(0,1,len(dl)),dl,1)[0],
      disagree=M.std(0).mean(),disagree_max=M.std(0).max(),n_form=len(d["P"])))
r=pd.DataFrame(rows)
feats=["pr_min","pr_med","pr_max","pr_std","nn","dl_absmean","dl_absmax",
       "dl_std","dl_end","dl_slope","disagree","disagree_max","n_form"]
m=lgb.LGBMRegressor(n_estimators=300,learning_rate=.05,num_leaves=15,
 min_child_samples=20,subsample=.8,colsample_bytree=.8,reg_lambda=5,verbose=-1)
m.fit(r[feats],r.w_opt); m.booster_.save_model(str(PKG/"harshini_trust_lgb.txt"))
manifest=json.loads((PKG/"manifest.json").read_text())
manifest["trust_model"]={"features":feats,"SH0":SH0,"CL0":CL0,"A_TRUST":.70,
 "sha256":hashlib.sha256((PKG/"harshini_trust_lgb.txt").read_bytes()).hexdigest()}
(PKG/"manifest.json").write_text(json.dumps(manifest,indent=2))
print("saved trust model",len(r),feats)
