"""Legal cached heel-GR posterior parameter search; diagnostic OOF tuning only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/results/heel_gr_posterior_search';OUT.mkdir(parents=True,exist_ok=True)
records=json.loads((ROOT/'exp/results/heel_calibrated_gr_datum/costs.json').read_text())
z=np.load(ROOT/'exp/results/heel_calibrated_gr_datum/oof.npz',allow_pickle=True)
base,y,groups=z['base'].astype(float),z['y'].astype(float),z['groups'].astype(str)
shifts=np.arange(-60,61,2,dtype=float)
C={k:np.asarray([r[f'{k}_costs'] for r in records],float) for k in ('cauchy','huber','l1')}
wells=pd.Series(groups).drop_duplicates().to_numpy(); wix=[np.flatnonzero(groups==w) for w in wells]
splits=list(GroupKFold(5).split(wells,groups=wells))
err=base-y
nw=np.asarray([len(ix) for ix in wix],float)
sew=np.asarray([err[ix].sum() for ix in wix],float)
ssew=np.asarray([np.square(err[ix]).sum() for ix in wix],float)
row=[]; bestcorr=None; bestscore=1e9
mixtures={'cauchy':C['cauchy'],'huber':C['huber'],'l1':C['l1'],
          'c75h25':.75*C['cauchy']+.25*C['huber'],
          'c50h50':.5*C['cauchy']+.5*C['huber'],
          'c75l25':.75*C['cauchy']+.25*C['l1']}
def score_cw(cw,ids=None):
 if ids is None: ids=np.arange(len(wells))
 return float(np.sqrt(np.sum(ssew[ids]+2*cw[ids]*sew[ids]+nw[ids]*cw[ids]**2)/nw[ids].sum()))
for loss,cost in mixtures.items():
 for temp in (.03,.05,.08,.12,.2,.3,.5,.8,1.2,2.0):
  for sigma in (5,8,12,18,25,40,80,1e6):
   cp=cost+(shifts[None,:]/sigma)**2*temp/2
   P=np.exp(np.clip(-(cp-cp.min(1,keepdims=True))/temp,-50,0));P/=P.sum(1,keepdims=True)
   mean=P@shifts; sd=np.sqrt(np.maximum(P@(shifts**2)-mean**2,0))
   for hedge in (.1,.2,.3,.4,.5,.7,1.):
    corrw=hedge*mean
    for clip in (4,8,12,20,60):
     cw=np.clip(corrw,-clip,clip);s=score_cw(cw)
     gains=[]
     for _,va in splits:
      gains.append(score_cw(np.zeros(len(wells)),va)-score_cw(cw,va))
     q={'loss':loss,'temperature':temp,'prior_sigma':sigma,'hedge':hedge,'clip':clip,
        'rmse':s,'fold_gains':gains,'fold_wins':sum(g>0 for g in gains),'mean_sd':float(sd.mean())}
     row.append(q)
     if s<bestscore:bestscore=s;bestcorr=cw.copy()
df=pd.DataFrame(row).sort_values('rmse');df.to_csv(OUT/'grid.csv',index=False)
summary={'warning':'post-hoc parameter search on existing OOF; requires locked confirmation/add-one audit',
         'base':score_cw(np.zeros(len(wells))),'best':row[int(np.argmin([q['rmse'] for q in row]))],
         'best_5of5':min((q for q in row if q['fold_wins']==5),key=lambda q:q['rmse'],default=None),
         'settings':len(row)}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
corr=np.zeros(len(y))
for j,ix in enumerate(wix):corr[ix]=bestcorr[j]
np.savez_compressed(OUT/'best_oof.npz',correction=corr,base=base,y=y,groups=groups)
print(json.dumps(summary,indent=2))
