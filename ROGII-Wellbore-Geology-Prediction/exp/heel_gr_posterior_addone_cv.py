"""Strict meta comparison of accepted heel correction and tuned posterior."""
from pathlib import Path
import contextlib,io,runpy,json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1];O=R/'exp/results/heel_gr_posterior_search';O.mkdir(parents=True,exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(R/'exp/meta_all_honest_oof.py'))
L,y,g,common=s['legs'],s['y'],s['wells'],s['common']
names=['har_physics','har_lgb','har_xgb','pil_blend_oof_postprocessed','v4_lgb7']
Z=np.column_stack([L[k] for k in names]); folds=list(GroupKFold(5).split(Z,groups=g))
oldq=np.load(R/'exp/results/heel_calibrated_gr_datum/oof.npz');newq=np.load(R/'exp/results/heel_gr_posterior_search/best_oof.npz')
old=L['v4_lgb7']+oldq['correction'][common];new=L['v4_lgb7']+newq['correction'][common]
def cv(A):
 p=np.zeros(len(y));co=[]
 for tr,va in folds:
  m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(A[tr[::8]],y[tr[::8]])
  p[va]=m.predict(A[va]);co.append(m.coef_.tolist())
 return p,co
def rm(p,ix=None):
 if ix is None:ix=np.arange(len(y))
 return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))
bp,_=cv(np.c_[Z,old]);out={}
for name,A in {'accepted_old':np.c_[Z,old],'replace_tuned':np.c_[Z,new],'both':np.c_[Z,old,new]}.items():
 p,co=cv(A);fg=[rm(bp,va)-rm(p,va) for _,va in folds]
 out[name]={'rmse':rm(p),'gain_vs_accepted':rm(bp)-rm(p),'fold_gains':fg,'fold_wins':sum(x>0 for x in fg),'weights':co}
S={'warning':'tuned parameters selected on these OOF predictions; improvement requires locked confirmation','variants':out}
(O/'meta_summary.json').write_text(json.dumps(S,indent=2));print(json.dumps(S,indent=2))
