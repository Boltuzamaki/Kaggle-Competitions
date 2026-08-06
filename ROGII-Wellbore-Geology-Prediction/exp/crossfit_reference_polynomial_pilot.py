"""Pilot complete-well polynomial correction search against cross-fit horizontal GR reference."""
from pathlib import Path
import json,joblib
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/crossfit_horizontal_gr_reference';f=pd.read_pickle(ROOT/'r_v4b/train_feats.pkl');base=np.asarray(joblib.load(ROOT/'r_v4b/stack_v4_oofs.joblib')['oofs']['lgb7'],float)
wells=pd.Series(f.well).drop_duplicates().to_numpy();folds=list(GroupKFold(5).split(wells,groups=wells));foldof=np.zeros(len(wells),int)
for k,(_,va) in enumerate(folds):foldof[va]=k
cache={}
for w in wells:
 h=pd.read_csv(ROOT/f'data/train/{w}__horizontal_well.csv');gr=h.GR.interpolate(limit_direction='both').to_numpy(float);med=np.nanmedian(gr);sc=max(1.4826*np.nanmedian(np.abs(gr-med)),8);cache[w]=(h,(gr-med)/sc)
refs=[]
for fold,(tr,_) in enumerate(folds):
 vals=[]
 for j in tr:
  h,z=cache[wells[j]];take=np.arange(0,len(h),8);vals.append(pd.DataFrame({'bin':np.round(h.TVT.to_numpy(float)[take]*2)/2,'z':np.clip(z[take],-4,4)}))
 a=pd.concat(vals,ignore_index=True).groupby('bin').z.median();refs.append((a.index.to_numpy(float),a.to_numpy(float)))
rng=np.random.RandomState(912);sample=rng.choice(len(wells),120,replace=False);records=[]
for jj,j in enumerate(sample):
 w=wells[j];h,z=cache[w];ix=np.flatnonzero(f.well.to_numpy()==w);rows=f.id.iloc[ix].str.rsplit('_',n=1).str[1].astype(int).to_numpy();truth=h.TVT.to_numpy(float)[rows];path=f.last_known_tvt.to_numpy(float)[ix]+base[ix]
 take=np.linspace(0,len(ix)-1,min(600,len(ix))).astype(int);x=np.linspace(-1,1,len(ix));xs=x[take];obs=z[rows[take]];tv,rg=refs[foldof[j]]
 coef=np.zeros((601,4));coef[1:,0]=rng.uniform(-35,35,600);coef[1:,1]=rng.uniform(-18,18,600);coef[1:,2]=rng.uniform(-12,12,600);coef[1:,3]=rng.uniform(-8,8,600)
 phi=np.c_[np.ones(len(xs)),xs,xs*xs-1/3,xs**3];cost=np.empty(len(coef));oracle=np.empty(len(coef))
 for q in range(0,len(coef),50):
  cc=coef[q:q+50];delta=phi@cc.T;ref=np.interp(path[take,None]+delta,tv,rg);cost[q:q+len(cc)]=np.mean(np.log1p((obs[:,None]-ref)**2),axis=0)+2e-4*np.sum(cc[:,1:]**2,axis=1)
  phif=np.c_[np.ones(len(x)),x,x*x-1/3,x**3];pp=path[:,None]+phif@cc.T;oracle[q:q+len(cc)]=np.mean((pp-truth[:,None])**2,axis=0)
 sel=np.argmin(cost);oi=np.argmin(oracle);delta=np.c_[np.ones(len(x)),x,x*x-1/3,x**3]@coef[sel]
 records.append({'well':str(w),'n':len(ix),'sse_base':float(np.square(path-truth).sum()),'sse_selected':float(np.square(path+delta-truth).sum()),'sse_oracle':float(oracle[oi]*len(ix)),'selected_coef':coef[sel].tolist(),'oracle_coef':coef[oi].tolist(),'cost_rank_oracle':int(np.argsort(cost).tolist().index(oi))})
 if (jj+1)%20==0:print('wells',jj+1,flush=True)
n=sum(q['n'] for q in records);score=lambda k:float(np.sqrt(sum(q[k] for q in records)/n));out={'protocol':'fixed 120-well seed-912 pilot; 601 cubic corrections; cross-fit global horizontal GR reference','wells':len(records),'base':score('sse_base'),'selected':score('sse_selected'),'candidate_oracle':score('sse_oracle'),'selected_well_wins':sum(q['sse_selected']<q['sse_base'] for q in records),'median_oracle_cost_rank':float(np.median([q['cost_rank_oracle'] for q in records]))}
(OUT/'polynomial_pilot_summary.json').write_text(json.dumps(out,indent=2));(OUT/'polynomial_pilot_rows.json').write_text(json.dumps(records));print(json.dumps(out,indent=2))
