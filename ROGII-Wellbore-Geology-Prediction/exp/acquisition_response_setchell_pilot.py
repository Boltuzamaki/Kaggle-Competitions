"""Pilot a measured-tool-response proxy: low-pass wireline GR before Setchell scoring."""
from pathlib import Path
import json,sys
import joblib,numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/acquisition_response_setchell_pilot';OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(ROOT/'exp'))
from whole_well_gr_warp_selector_cv import candidate_paths,rmse
from binned_stratigraphic_correlation_cv import percentile_normalize,corr

def evidence(hw,tw,rows,paths,sigma):
 station=rows.id.astype(str).str.rsplit('_',n=1).str[1].astype(int).to_numpy();obs=percentile_normalize(hw.GR.to_numpy(float))[station]
 tt=tw.TVT.to_numpy(float); raw=tw.GR.interpolate(limit_direction='both').fillna(tw.GR.median()).to_numpy(float)
 # sigma is in feet, converted using median wireline TVT sampling.
 step=max(np.nanmedian(np.abs(np.diff(tt))),.05)
 filtered = raw if sigma == 0 else gaussian_filter1d(raw,sigma/step,mode='nearest')
 ref=percentile_normalize(filtered)
 last=float(rows.last_known_tvt.iloc[0]);out=[]
 for path in paths:
  tvt=last+path;use=np.isfinite(obs)&np.isfinite(tvt);origin=np.floor(tvt[use].min()/.5)*.5;b=np.floor((tvt[use]-origin)/.5).astype(int)
  count=np.bincount(b);total=np.bincount(b,weights=obs[use]);occ=np.flatnonzero(count>=2)
  if len(occ)<8:occ=np.flatnonzero(count)
  lat=total[occ]/count[occ];cent=origin+(occ+.5)*.5;out.append(np.arctanh(np.clip(corr(lat,np.interp(cent,tt,ref)),-.999,.999)))
 return np.asarray(out)

frame=pd.read_pickle(ROOT/'r_v4b/train_feats.pkl');oof=joblib.load(ROOT/'r_v4b/stack_v4_oofs.joblib')['oofs'];stats=[]
for well,g in frame.groupby('well'):
 b,_,_=candidate_paths(g,oof);stats.append((well,rmse(g.target,b)))
stats=pd.DataFrame(stats,columns=['well','err']);stats['bin']=pd.qcut(stats.err,4,labels=False)
def pick(seed,excluded=()):
 rng=np.random.RandomState(seed);a=[]
 for _,g in stats[~stats.well.isin(excluded)].groupby('bin'):a.extend(rng.choice(g.well,30,False))
 return a
dev=pick(440);conf=pick(1440,dev);store={}
for phase,ws in [('dev',dev),('confirm',conf)]:
 for i,(well,g) in enumerate(frame[frame.well.isin(ws)].groupby('well')):
  hw=pd.read_csv(ROOT/f'data/train/{well}__horizontal_well.csv');tw=pd.read_csv(ROOT/f'data/train/{well}__typewell.csv').sort_values('TVT');base,paths,labels=candidate_paths(g,oof)
  ev={s:evidence(hw,tw,g,paths,s) for s in (0,1,2,4,8,16)};complexity=np.asarray([(x/20)**2+(q/12)**2+(c/6)**2 for _,x,q,c in labels]);store[(phase,well)]=(g.target.to_numpy(),base,paths,complexity,ev)
  if (i+1)%30==0:print(phase,i+1,flush=True)
def eval_phase(phase,sigma):
 ya=[];ba=[];pa=[]
 for (ph,_),(y,b,p,c,e) in store.items():
  if ph!=phase:continue
  cost=-e[sigma]+.3*c;scale=np.std(cost)*.25+1e-6;w=np.exp(np.clip(-(cost-cost.min())/scale,-30,0));w/=w.sum();ya.append(y);ba.append(b);pa.append(.7*b+.3*(w@p))
 y,b,p=map(np.concatenate,(ya,ba,pa));return rmse(y,b),rmse(y,p)
dev_scores={s:eval_phase('dev',s) for s in (0,1,2,4,8,16)};best=min(dev_scores,key=lambda s:dev_scores[s][1]);cb,cp=eval_phase('confirm',best);zcb,zcp=eval_phase('confirm',0)
summary={'dev_wells':120,'confirm_wells':120,'overlap':0,'dev':{str(s):{'base':v[0],'score':v[1],'gain':v[0]-v[1]} for s,v in dev_scores.items()},'selected_sigma_ft':best,'confirm_base':cb,'confirm_selected':cp,'confirm_gain':cb-cp,'confirm_unsmoothed':zcp,'gain_over_unsmoothed':zcp-cp}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
