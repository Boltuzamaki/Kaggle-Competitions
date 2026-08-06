"""Cross-fitted global horizontal-log GR reference and V4 datum scan."""
from pathlib import Path
import json,joblib
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/crossfit_horizontal_gr_reference';OUT.mkdir(parents=True,exist_ok=True)
f=pd.read_pickle(ROOT/'r_v4b/train_feats.pkl');base=np.asarray(joblib.load(ROOT/'r_v4b/stack_v4_oofs.joblib')['oofs']['lgb7'],float)
wells=pd.Series(f.well).drop_duplicates().to_numpy();folds=list(GroupKFold(5).split(wells,groups=wells));shifts=np.arange(-40,40.1,2)
cache={}
for w in wells:
 h=pd.read_csv(ROOT/f'data/train/{w}__horizontal_well.csv');gr=h.GR.interpolate(limit_direction='both').to_numpy(float)
 med=np.nanmedian(gr);sc=max(1.4826*np.nanmedian(np.abs(gr-med)),8);cache[w]=(h,(gr-med)/sc)
pred_shift=np.zeros(len(wells));oracle_shift=np.zeros(len(wells));corr_true=[];corr_base=[];fold_id=np.zeros(len(wells),int)
cost_matrix=np.zeros((len(wells),len(shifts)),np.float32)
for fold,(tr,va) in enumerate(folds):
 vals=[]
 for j in tr:
  h,z=cache[wells[j]];take=np.arange(0,len(h),5);vals.append(pd.DataFrame({'bin':np.round(h.TVT.to_numpy(float)[take]*2)/2,'z':np.clip(z[take],-4,4)}))
 a=pd.concat(vals,ignore_index=True).groupby('bin').z.median();tv=a.index.to_numpy(float);rg=a.to_numpy(float)
 for j in va:
  w=wells[j];fold_id[j]=fold;h,z=cache[w];ix=np.flatnonzero(f.well.to_numpy()==w);rows=f.id.iloc[ix].str.rsplit('_',n=1).str[1].astype(int).to_numpy()
  take=np.linspace(0,len(rows)-1,min(800,len(rows))).astype(int);rr=rows[take];obs=z[rr];path=f.last_known_tvt.to_numpy(float)[ix][take]+base[ix][take];truth=h.TVT.to_numpy(float)[rr]
  costs=np.asarray([np.mean(np.log1p((obs-np.interp(path+s,tv,rg))**2)) for s in shifts]);P=np.exp(-(costs-costs.min())/.2);P/=P.sum();pred_shift[j]=P@shifts
  cost_matrix[j]=costs
  oracle_shift[j]=np.mean(truth-path);corr_true.append(np.corrcoef(obs,np.interp(truth,tv,rg))[0,1]);corr_base.append(np.corrcoef(obs,np.interp(path,tv,rg))[0,1])
 print('fold',fold,'done',flush=True)
# expand constant well corrections and exact scores
corr=np.zeros(len(f));oracle=np.zeros(len(f))
for j,w in enumerate(wells):
 ix=np.flatnonzero(f.well.to_numpy()==w);corr[ix]=pred_shift[j];oracle[ix]=oracle_shift[j]
y=f.target.to_numpy(float);defp=base
rm=lambda p:float(np.sqrt(np.mean(np.square(p-y))))
grid=[{'hedge':float(a),'rmse':rm(defp+a*corr)} for a in np.linspace(0,1,21)]
out={'protocol':'outer whole-well GKF global horizontal GR(TVT) reference; validation well excluded; full horizontal GR normalization is inference-legal',
     'wells':len(wells),'base':rm(defp),'best':min(grid,key=lambda x:x['rmse']),'raw':rm(defp+corr),'constant_oracle':rm(defp+oracle),
     'median_corr_true':float(np.nanmedian(corr_true)),'median_corr_base':float(np.nanmedian(corr_base))}
np.savez_compressed(OUT/'oof.npz',correction=corr,oracle=oracle,well_shift=pred_shift,oracle_well_shift=oracle_shift,
                    costs=cost_matrix,shifts=shifts,wells=wells,fold=fold_id);(OUT/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
