"""Strict add/replace audit for locked dynamic-BMA PF OOF curve."""
from pathlib import Path
import contextlib,io,json,joblib,runpy
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/pf_dynamic_bma_full_oof'
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(ROOT/'exp/meta_all_honest_oof.py'))
legs=s['legs'];y=s['y'];groups=s['wells'];common=s['common']
names=['har_physics','har_lgb','har_xgb','pil_blend_oof_postprocessed','v4_lgb7']
f=pd.read_pickle(ROOT/'r_v4b/train_feats.pkl');a=joblib.load(OUT/'predictions.joblib')
pred=np.empty(len(f));filled=np.zeros(len(f),bool);pos=f.groupby(f.well.astype(str),sort=False).indices
for w,o in a.items():
 ix=pos[w];rows=f.id.iloc[ix].str.rsplit('_',n=1).str[-1].astype(int).to_numpy()
 if not np.array_equal(rows,o['row_index']):raise RuntimeError(f'identity {w}')
 pred[ix]=o['prediction'];filled[ix]=1
if not filled.all():raise RuntimeError('missing rows')
pred=(pred-f.last_known_tvt.to_numpy(float))[common]
# Independently locked Student-t10 curve for the promoted-stack comparison.
sa=joblib.load(ROOT/'exp/results/pf_student10_full_oof/predictions.joblib');student=np.empty(len(f));sf=np.zeros(len(f),bool)
for w,o in sa.items():
 ix=pos[w];student[ix]=o['prediction'];sf[ix]=1
if not sf.all():raise RuntimeError('missing Student-t rows')
student=(student-f.last_known_tvt.to_numpy(float))[common]
heel=legs['v4_lgb7']+np.load(ROOT/'exp/results/heel_calibrated_gr_datum/oof.npz')['correction'][common]
base=np.column_stack([legs[n] for n in names]);folds=list(GroupKFold(5).split(base,groups=groups))
def fit(x):
 p=np.zeros(len(y));co=[]
 for tr,va in folds:
  m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(x[tr[::8]],y[tr[::8]])
  p[va]=m.predict(x[va]);co.append(m.coef_.tolist())
 return p,co
def score(p,ix=None):return float(np.sqrt(np.mean((p-y if ix is None else p[ix]-y[ix])**2)))
mats={'accepted':np.c_[base,heel],'add_bma':np.c_[base,heel,pred],
 'replace_v4':np.c_[base[:,:4],pred,heel],
 'student_promoted':np.c_[base,heel,student],
 'student_plus_bma':np.c_[base,heel,student,pred]}
res={};pp={}
for n,x in mats.items():pp[n],co=fit(x);res[n]={'rmse':score(pp[n]),'weights':co}
b=pp['accepted'];br=score(b)
for n in res:
 gains=[score(b,va)-score(pp[n],va) for _,va in folds]
 res[n].update(gain=br-res[n]['rmse'],fold_gains=gains,fold_wins=sum(g>0 for g in gains))
grid=[]
for a in np.linspace(0,1,21):
 p=(1-a)*b+a*pp['replace_v4'];g=[score(b,va)-score(p,va) for _,va in folds]
 grid.append({'blend':float(a),'rmse':score(p),'gain':br-score(p),'fold_gains':g,'fold_wins':sum(x>0 for x in g)})
summary={'bma_standalone':score(pred),'variants':res,'best_blend':min(grid,key=lambda z:z['rmse']),
 'best_5of5':min((z for z in grid if z['fold_wins']==5),key=lambda z:z['rmse'],default=None),
 'bma_gain_over_student_promoted':res['student_promoted']['rmse']-res['student_plus_bma']['rmse'],
 'below_7':min(v['rmse'] for v in res.values())<7}
(OUT/'meta_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
