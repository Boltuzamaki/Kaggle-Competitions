"""Exact meta audit for the full legal raw/typewell row-feature LGB leg."""
from pathlib import Path
import contextlib,io,runpy,json,numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1];O=R/"exp/results/raw_typewell_feature_blocks/meta";O.mkdir(parents=True,exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(R/"exp/meta_all_honest_oof.py"))
L,y,g,c=s["legs"],s["y"],s["wells"],s["common"];names=["har_physics","har_lgb","har_xgb","pil_blend_oof_postprocessed","v4_lgb7"]
Z=np.column_stack([L[k] for k in names]);heel=L["v4_lgb7"]+np.load(R/"exp/results/heel_calibrated_gr_datum/oof.npz")["correction"][c]
new=np.load(R/"exp/results/raw_typewell_feature_blocks/full_oof.npz")["all_blocks"][c]
folds=list(GroupKFold(5).split(Z,groups=g))
def cv(A):
 p=np.zeros(len(y));co=[]
 for tr,va in folds:
  m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(A[tr[::8]],y[tr[::8]])
  p[va]=m.predict(A[va]);co.append(m.coef_.tolist())
 return p,co
def rm(p,i=None):
 if i is None:i=np.arange(len(y))
 return float(np.sqrt(np.mean((p[i]-y[i])**2)))
variants={"five":Z,"five_plus_raw":np.c_[Z,new],"heel":np.c_[Z,heel],"heel_plus_raw":np.c_[Z,heel,new]}
pred={k:cv(A) for k,A in variants.items()};b=pred["five"][0];out={}
for k,(p,co) in pred.items():
 fg=[rm(b,va)-rm(p,va) for tr,va in folds];out[k]={"rmse":rm(p),"gain_vs_five":rm(b)-rm(p),
 "fold_gains":fg,"fold_wins":sum(x>0 for x in fg),"weights":co}
S={"variants":out};(O/"summary.json").write_text(json.dumps(S,indent=2));print(json.dumps(S,indent=2))
