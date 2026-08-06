"""Pilot: give the existing PF a visible-prefix affine-calibrated GR reference."""
from pathlib import Path
import json,time
import numpy as np
from joblib import Parallel,delayed
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'exp'))
from wellbore_lib import load_well,list_wells,ps_index
from pf_tracker import pf_predict
OUT=ROOT/'exp/results/pf_affine_reference';OUT.mkdir(parents=True,exist_ok=True)

def robust_affine(x,y):
 ok=np.isfinite(x)&np.isfinite(y);x,y=x[ok],y[ok]
 X=np.c_[x,np.ones(len(x))];coef=np.linalg.lstsq(X,y,rcond=None)[0]
 for _ in range(5):
  r=y-X@coef;s=1.4826*np.median(np.abs(r-np.median(r)))+1e-3
  w=1/np.maximum(1,np.abs(r)/(2.5*s));coef=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]
 return float(coef[0]),float(coef[1])
def one(w,N,seeds,scale):
 h,t=load_well('data/train',w);ps=ps_index(h)
 if ps<20 or ps>=len(h)-5:return None
 ev=h.TVT_input.isna().to_numpy();true=h.TVT.to_numpy(float)[ev]
 raw=pf_predict(h,t,N,seeds,scale)[ev]
 ts=t.sort_values('TVT').copy();kn=h[~ev]
 ref=np.interp(kn.TVT_input.to_numpy(float),ts.TVT.to_numpy(float),ts.GR.interpolate(limit_direction='both').to_numpy(float))
 a,b=robust_affine(ref,kn.GR.interpolate(limit_direction='both').to_numpy(float))
 tc=ts.copy();tc['GR']=a*tc.GR.to_numpy(float)+b
 cal=pf_predict(h,tc,N,seeds,scale)[ev]
 return {'well':w,'n':len(true),'sse_raw':float(np.square(raw-true).sum()),'sse_cal':float(np.square(cal-true).sum()),
         'rmse_raw':float(np.sqrt(np.mean(np.square(raw-true)))),'rmse_cal':float(np.sqrt(np.mean(np.square(cal-true)))), 'a':a,'b':b}
if __name__=='__main__':
 wells=list_wells('data/train');rng=np.random.RandomState(7);sample=[wells[i] for i in rng.permutation(len(wells))[:120]]
 t=time.time();rows=[x for x in Parallel(n_jobs=10)(delayed(one)(w,200,12,12.) for w in sample) if x]
 n=sum(x['n'] for x in rows);raw=np.sqrt(sum(x['sse_raw'] for x in rows)/n);cal=np.sqrt(sum(x['sse_cal'] for x in rows)/n)
 out={'protocol':'fixed 120-well seed-7 pilot; identical PF parameters; affine fit uses visible prefix only','wells':len(rows),'rows':n,'seconds':time.time()-t,
      'raw':raw,'calibrated':cal,'gain':raw-cal,'well_wins':sum(x['rmse_cal']<x['rmse_raw'] for x in rows)}
 (OUT/'pilot_summary.json').write_text(json.dumps(out,indent=2));(OUT/'pilot_rows.json').write_text(json.dumps(rows));print(json.dumps(out,indent=2))
