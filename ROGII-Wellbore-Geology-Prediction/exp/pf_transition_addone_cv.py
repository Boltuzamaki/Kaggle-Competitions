"""Strict add-one meta audit for locked high-persistence PF curve."""
from pathlib import Path
import contextlib,io,json,runpy
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1];O=R/'exp/results/pf_transition_law'
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(R/'exp/meta_all_honest_oof.py'))
L,y,g,common=s['legs'],s['y'],s['wells'],s['common'];names=['har_physics','har_lgb','har_xgb','pil_blend_oof_postprocessed','v4_lgb7']
vf=pd.read_pickle(R/'r_v4b/train_feats.pkl')[['id','last_known_tvt','target']];q=np.load(O/'full_oof.npz',allow_pickle=True)
mp=pd.Series(np.arange(len(q['id'])),index=q['id'].astype(str));ix=mp.loc[vf.id.astype(str)].to_numpy();new=q['prediction'][ix]-vf.last_known_tvt.to_numpy(float)
terr=float(np.max(np.abs((q['y'][ix]-vf.last_known_tvt.to_numpy(float))-vf.target.to_numpy(float))))
if terr>.01:raise RuntimeError(f'target identity {terr}')
oldq=np.load(R/'exp/results/heel_calibrated_gr_datum/oof.npz');old=L['v4_lgb7']+oldq['correction'][common]
Z=np.column_stack([L[k] for k in names]);new=new[common];folds=list(GroupKFold(5).split(Z,groups=g))
def cv(A):
 p=np.zeros(len(y));co=[]
 for tr,va in folds:
  m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(A[tr[::8]],y[tr[::8]]);p[va]=m.predict(A[va]);co.append(m.coef_.tolist())
 return p,co
def rm(p,ix=None):
 if ix is None:ix=np.arange(len(y))
 return float(np.sqrt(np.mean(np.square(p[ix]-y[ix]))))
bp,_=cv(np.c_[Z,old]);out={};preds={}
for name,A in {'accepted':np.c_[Z,old],'add_pf':np.c_[Z,old,new],'replace_v4_with_pf':np.c_[Z[:,:4],new,old]}.items():
 p,co=cv(A);preds[name]=p;fg=[rm(bp,va)-rm(p,va) for _,va in folds];out[name]={'rmse':rm(p),'gain':rm(bp)-rm(p),'fold_gains':fg,'fold_wins':sum(x>0 for x in fg),'weights':co}
grid=[]
for a in np.linspace(0,1,21):
 p=(1-a)*bp+a*preds['replace_v4_with_pf'];fg=[rm(bp,va)-rm(p,va) for _,va in folds]
 grid.append({'replacement_blend':float(a),'rmse':rm(p),'fold_gains':fg,'fold_wins':sum(x>0 for x in fg)})
S={'target_identity_max_abs':terr,'new_leg_rmse':rm(new),'variants':out,'best_blend':min(grid,key=lambda x:x['rmse']),
   'best_5of5_blend':min((x for x in grid if x['fold_wins']==5),key=lambda x:x['rmse'],default=None)}
np.savez_compressed(O/'meta_oof.npz',accepted=bp,replacement=preds['replace_v4_with_pf'],y=y,groups=g)
(O/'meta_summary.json').write_text(json.dumps(S,indent=2));print(json.dumps(S,indent=2))
