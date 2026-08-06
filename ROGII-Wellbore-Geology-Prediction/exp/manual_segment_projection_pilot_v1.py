"""Target-free projection of OOF paths onto manual-style dip segments.

ROGII describes manual interpretation as segmenting and stretch/squeeze/fault
editing.  Approximate that label geometry by median-filtering the predicted
increment (piecewise-stable dip), then integrate from the exact PS anchor.
Truth is read only after all candidate paths are frozen for scoring.
"""
from pathlib import Path
import json, joblib
import numpy as np, pandas as pd
from scipy.ndimage import median_filter, gaussian_filter1d

R=Path(__file__).resolve().parents[1]
O=R/'exp/results/manual_segment_projection_pilot_v1';O.mkdir(parents=True,exist_ok=True)
f=pd.read_pickle(R/'r_v4b/train_feats.pkl')
z=joblib.load(R/'r_v4b/stack_v4_oofs.joblib'); full=np.asarray(z['oofs']['lgb7'],float)
# Bounded, deterministic pilot to coexist with the other active compute jobs.
keep=set(sorted(f.well.astype(str).unique())[:240]); mask=f.well.astype(str).isin(keep).to_numpy()
f=f.loc[mask].reset_index(drop=True);base=full[mask]
y=f.target.to_numpy(float); wells=f.well.astype(str).to_numpy()

def rm(a,b,m=None):
 if m is None:m=np.ones(len(a),bool)
 return float(np.sqrt(np.mean((a[m]-b[m])**2)))

def project(p,win,sigma):
 # p is delta from known PS TVT, so prepend the exact zero anchor.
 q=np.r_[0.,p]; d=np.diff(q)
 ds=median_filter(d,size=win,mode='nearest')
 if sigma: ds=gaussian_filter1d(ds,sigma,mode='nearest')
 return np.cumsum(ds)

groups=[g.index.to_numpy() for _,g in f.groupby('well',sort=False)]

fold=np.asarray(z.get('fold_ids', z.get('fold', np.full(len(y),-1))))
if len(fold)!=len(y) or (fold<0).all():
 # Stable whole-well hash folds are diagnostic only; candidate generation is
 # target-free and no model is fit.
 fold=np.array([int(w,16)%5 for w in wells])
rows=[]
for win in (11,31,61,121,241,481):
 for sig in (0,2,5,10):
  p=np.empty_like(base)
  for m in groups:p[m]=project(base[m],win,sig)
  for blend in (.1,.25,.5,.75,1.):
   q=(1-blend)*base+blend*p
   rec={'window':win,'sigma':sig,'blend':blend,'rmse':rm(y,q),'gain':rm(y,base)-rm(y,q)}
   rec['fold_gains']=[rm(y,base,fold==k)-rm(y,q,fold==k) for k in sorted(set(fold))]
   rec['fold_wins']=sum(x>0 for x in rec['fold_gains']);rows.append(rec)
rows=sorted(rows,key=lambda x:x['rmse'])
summary={'baseline':rm(y,base),'best':rows[0],
 'best_5of5':next((x for x in rows if x['fold_wins']==5),None),
 'protocol':'shape-only transform of fixed lgb7 OOF; truth used only to score frozen grid; exploratory, requires locked confirmation',
 'source_clue':'organizer deck manual segment/stretch/squeeze/fault workflow'}
(O/'summary.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(rows).to_json(O/'grid.json',orient='records',indent=2)
print(json.dumps(summary,indent=2))
