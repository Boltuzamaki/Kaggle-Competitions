"""Full-field frozen Setchell correction and Student-t stack add-one audit."""
from pathlib import Path
import json, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'exp/results/setchell_fullfield_addone';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'exp'))
from whole_well_gr_warp_selector_cv import candidate_paths,rmse
from binned_stratigraphic_correlation_cv import path_scores

frame=pd.read_pickle(ROOT/'r_v4b/train_feats.pkl');oof=joblib.load(ROOT/'r_v4b/stack_v4_oofs.joblib')['oofs']
ids=[]; corr=[]; yy=[]; wells=[]; standalone=[]; baseall=[]
for i,(well,g) in enumerate(frame.groupby('well',sort=True)):
 hw=pd.read_csv(ROOT/f'data/train/{well}__horizontal_well.csv');tw=pd.read_csv(ROOT/f'data/train/{well}__typewell.csv').sort_values('TVT')
 base,paths,labels=candidate_paths(g,oof); scores,_=path_scores(hw,tw,g,paths);e=scores['pearson_0.5']
 complexity=np.asarray([(s/20)**2+(q/12)**2+(c/6)**2 for _,s,q,c in labels]);cost=-e+.3*complexity
 scale=np.std(cost)*.25+1e-6;w=np.exp(np.clip(-(cost-cost.min())/scale,-30,0));w/=w.sum(); post=w@paths
 pred=.7*base+.3*post
 ids.extend(g.id.astype(str));corr.extend(pred-base);yy.extend(g.target);wells.extend([well]*len(g));standalone.extend(pred);baseall.extend(base)
 if (i+1)%50==0:print(f'wells={i+1}',flush=True)
ids=np.asarray(ids);corr=np.asarray(corr);yy=np.asarray(yy);wells=np.asarray(wells);standalone=np.asarray(standalone);baseall=np.asarray(baseall)

# Align to the exact common-row Student-t meta artifact.
aq=np.load(ROOT/'exp/results/heel_calibrated_gr_datum/full_meta/oof.npz'); common=aq['global_indices']; y=aq['y']
gt=pd.read_parquet(ROOT/'exp/public_artifacts/pilkwang/oof/train_gt.parquet',columns=['id','well_id','target_delta_from_last_known'])
mp=pd.Series(np.arange(len(ids)),index=ids);ix=mp.loc[gt.id.astype(str).to_numpy()[common]].to_numpy(); correction=corr[ix]
if np.max(np.abs(yy[ix]-y))>.01:raise RuntimeError('target alignment')
meta=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True); student=.1*meta['accepted']+.9*meta['replacement']
groups=gt.well_id.astype(str).to_numpy()[common];folds=list(GroupKFold(5).split(y,groups=groups))
def score(p,ind=None):
 r=p-y if ind is None else p[ind]-y[ind];return float(np.sqrt(np.mean(r*r)))
grid=[]
for a in np.linspace(-.5,1.5,41):
 p=student+a*correction; fg=[score(student,v)-score(p,v) for _,v in folds]
 grid.append({'scale':float(a),'rmse':score(p),'gain':score(student)-score(p),'fold_gains':fg,'fold_wins':sum(x>0 for x in fg)})
summary={'rows':len(y),'wells':len(set(groups)),'full_standalone_base':rmse(yy,baseall),'full_standalone_setchell':rmse(yy,standalone),
 'student_baseline':score(student),'frozen_scale_1':next(x for x in grid if x['scale']==1.0),'best':min(grid,key=lambda x:x['rmse']),
 'best_5of5':min((x for x in grid if x['fold_wins']==5),key=lambda x:x['rmse'],default=None),'truth_used_only_after_frozen_correction':True}
np.savez_compressed(OUT/'correction.npz',id=ids,correction=corr);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
