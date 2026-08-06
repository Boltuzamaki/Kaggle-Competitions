"""Cubic-leg add-ons to the accepted six-leg global-heel meta."""
from pathlib import Path
import contextlib,io,runpy,json,numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1];O=R/"exp/results/well_cubic_coeff_tabular/meta_with_heel";O.mkdir(parents=True,exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(R/"exp/meta_all_honest_oof.py"))
L,y,g,c=s["legs"],s["y"],s["wells"],s["common"];n=["har_physics","har_lgb","har_xgb","pil_blend_oof_postprocessed","v4_lgb7"]
v4=L["v4_lgb7"];heel=v4+np.load(R/"exp/results/heel_calibrated_gr_datum/oof.npz")["correction"][c]
Z=np.column_stack([*[L[k] for k in n],heel]);q=np.load(R/"exp/results/well_cubic_coeff_tabular/oof.npz")
cands={"cat_raw":q["cat"][c],"cat_shrink80":v4+.8*(q["cat"][c]-v4),
 "extra_shrink75":v4+.75*(q["extra"][c]-v4),"ensemble_shrink75":v4+.75*(q["ensemble"][c]-v4)}
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
for name,x in cands.items():
 p,co=cv(np.c_[Z,x]);fg=[rm(b,va)-rm(p,va) for tr,va in folds]
 out[name]={"rmse":rm(p),"gain_vs_heel":rm(b)-rm(p),"fold_gains":fg,"fold_wins":sum(z>0 for z in fg),"weights":co}
S={"heel_base":rm(b),"variants":out};(O/"summary.json").write_text(json.dumps(S,indent=2));print(json.dumps(S,indent=2))
