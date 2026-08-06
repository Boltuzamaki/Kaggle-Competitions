"""Exact deployable-meta comparison: global heel, multiscale heel, and both."""
from pathlib import Path
import contextlib,io,runpy,json,numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1];O=R/"exp/results/heel_gr_multiscale/meta_compare";O.mkdir(parents=True,exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(R/"exp/meta_all_honest_oof.py"))
L,y,g,common=s["legs"],s["y"],s["wells"],s["common"]
names=["har_physics","har_lgb","har_xgb","pil_blend_oof_postprocessed","v4_lgb7"]
Z=np.column_stack([L[k] for k in names]);folds=list(GroupKFold(5).split(Z,groups=g))
qg=np.load(R/"exp/results/heel_calibrated_gr_datum/oof.npz")
qm=np.load(R/"exp/results/heel_gr_multiscale/oof.npz")
glob=L["v4_lgb7"]+qg["correction"][common];multi=qm["prediction"][common]
def cv(A):
 p=np.zeros(len(y));co=[]
 for tr,va in folds:
  m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(A[tr[::8]],y[tr[::8]])
  p[va]=m.predict(A[va]);co.append(m.coef_.tolist())
 return p,co
def rm(p,i=None):
 if i is None:i=np.arange(len(y))
 return float(np.sqrt(np.mean((p[i]-y[i])**2)))
b,_=cv(Z);out={}
for name,A in {"global":np.c_[Z,glob],"multiscale":np.c_[Z,multi],
               "both":np.c_[Z,glob,multi]}.items():
 p,co=cv(A);fg=[rm(b,va)-rm(p,va) for tr,va in folds]
 out[name]={"rmse":rm(p),"gain":rm(b)-rm(p),"fold_gains":fg,
            "fold_wins":sum(x>0 for x in fg),"weights":co}
S={"base":rm(b),"variants":out};(O/"summary.json").write_text(json.dumps(S,indent=2));print(json.dumps(S,indent=2))
