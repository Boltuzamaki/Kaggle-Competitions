"""Deployable-meta add-one audit for complete-well cubic CatBoost leg."""
from pathlib import Path
import contextlib,io,runpy,json,numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1];O=R/"exp/results/well_cubic_coeff_tabular/meta";O.mkdir(parents=True,exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(R/"exp/meta_all_honest_oof.py"))
L,y,g,common=s["legs"],s["y"],s["wells"],s["common"];names=["har_physics","har_lgb","har_xgb","pil_blend_oof_postprocessed","v4_lgb7"]
Z=np.column_stack([L[k] for k in names]);q=np.load(R/"exp/results/well_cubic_coeff_tabular/oof.npz")
cat=q["cat"][common];glob=L["v4_lgb7"]+np.load(R/"exp/results/heel_calibrated_gr_datum/oof.npz")["correction"][common]
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
b,_=cv(Z);out={}
for name,A in {"cubic_cat":np.c_[Z,cat],"global_heel":np.c_[Z,glob],"both":np.c_[Z,glob,cat]}.items():
 p,co=cv(A);fg=[rm(b,va)-rm(p,va) for tr,va in folds]
 out[name]={"rmse":rm(p),"gain":rm(b)-rm(p),"fold_gains":fg,"fold_wins":sum(x>0 for x in fg),"weights":co}
S={"base":rm(b),"variants":out};(O/"summary.json").write_text(json.dumps(S,indent=2));print(json.dumps(S,indent=2))
