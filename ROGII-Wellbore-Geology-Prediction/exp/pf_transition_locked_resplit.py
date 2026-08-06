"""Independent whole-well resplit confirmation of locked persistent-PF replacement."""
from pathlib import Path
import contextlib,io,json,runpy
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
R=Path(__file__).resolve().parents[1];O=R/'exp/results/pf_transition_law'
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(R/'exp/meta_all_honest_oof.py'))
L,y,g,common=s['legs'],s['y'],s['wells'],s['common'];names=['har_physics','har_lgb','har_xgb','pil_blend_oof_postprocessed','v4_lgb7']
vf=pd.read_pickle(R/'r_v4b/train_feats.pkl')[['id','last_known_tvt']];q=np.load(O/'full_oof.npz',allow_pickle=True);mp=pd.Series(np.arange(len(q['id'])),index=q['id'].astype(str));ix=mp.loc[vf.id.astype(str)].to_numpy()
new=(q['prediction'][ix]-vf.last_known_tvt.to_numpy(float))[common];oldq=np.load(R/'exp/results/heel_calibrated_gr_datum/oof.npz');old=L['v4_lgb7']+oldq['correction'][common]
Z=np.column_stack([L[k] for k in names]);uw=pd.Series(g).drop_duplicates().to_numpy();spl=[]
for trw,vaw in KFold(5,shuffle=True,random_state=20260801).split(uw):spl.append((np.flatnonzero(np.isin(g,uw[trw])),np.flatnonzero(np.isin(g,uw[vaw]))))
def cv(A):
 p=np.zeros(len(y))
 for tr,va in spl:p[va]=Ridge(alpha=100,positive=True,fit_intercept=False).fit(A[tr[::8]],y[tr[::8]]).predict(A[va])
 return p
def rm(p,ix=None):
 if ix is None:ix=np.arange(len(y))
 return float(np.sqrt(np.mean(np.square(p[ix]-y[ix]))))
b=cv(np.c_[Z,old]);p=cv(np.c_[Z[:,:4],new,old]);fg=[rm(b,va)-rm(p,va) for _,va in spl]
out={'split':'KFold unique wells shuffle=True random_state=20260801','locked_change':'replace raw v4 leg with MOM=.9995 PF; no blend tuning','accepted':rm(b),'replacement':rm(p),'gain':rm(b)-rm(p),'fold_gains':fg,'fold_wins':sum(x>0 for x in fg)}
(O/'locked_resplit_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
