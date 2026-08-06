"""Fixed pilot for physically motivated PF structural-rate mean reversion."""
from pathlib import Path
import json,sys,time
import numpy as np
from joblib import Parallel,delayed
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'exp')]
from wellbore_lib import load_well,list_wells,ps_index

MOMS=(.95,.98,.99,.995,.998,.9995)
def run(md,z,gr,tv,tg,last_tvt,last_z,ir,gs,N,seed,mom):
 rng=np.random.default_rng(seed);pos=last_tvt+last_z+2*rng.standard_normal(N);rate=ir+.01*rng.standard_normal(N);w=np.ones(N)/N
 out=np.empty(len(md));lo,hi=tv[0]-100,tv[-1]+100
 for i in range(len(md)):
  rate=mom*rate+.002*rng.standard_normal(N);pos+=rate+.005*rng.standard_normal(N)
  tp=np.clip(pos-z[i],lo,hi);pos=tp+z[i]
  d=(gr[i]-np.interp(tp,tv,tg))/gs;lk=np.maximum(np.exp(-.5*np.minimum(d*d,600)),1e-300)
  w*=lk;s=w.sum();w=w/s if s>0 else np.ones(N)/N
  if 1/(w*w).sum()<.5*N:
   idx=np.clip(np.searchsorted(np.cumsum(w),rng.uniform(0,1/N)+np.arange(N)/N),0,N-1)
   pos=pos[idx]+.1*rng.standard_normal(N);rate=rate[idx]+.001*rng.standard_normal(N);w[:]=1/N
  out[i]=w@(pos-z[i])
 return out
def one(w,N=100,seeds=4,moms=MOMS):
 h,t=load_well('data/train',w);ps=ps_index(h)
 if ps<20 or ps>=len(h)-5:return None
 kn=h.iloc[:ps];ev=h.iloc[ps:];ts=t.sort_values('TVT');tv=ts.TVT.to_numpy(float);tg=ts.GR.interpolate(limit_direction='both').to_numpy(float)
 last=kn.iloc[-1];at=np.interp(kn.TVT_input,tv,tg);gs=float(np.clip(np.nanstd(kn.GR.fillna(0)-at),10,60))
 tail=kn.tail(30);dm=np.diff(tail.MD);ir=float(np.median((np.diff(tail.TVT_input)+np.diff(tail.Z))[dm>0]/dm[dm>0]))
 gr=h.GR.interpolate(limit_direction='both').fillna(np.nanmean(tg)).to_numpy(float)[ps:];y=h.TVT.to_numpy(float)[ps:]
 ans={'well':w,'n':len(y)}
 for mom in moms:
  pp=np.mean([run(ev.MD.to_numpy(float),ev.Z.to_numpy(float),gr,tv,tg,float(last.TVT_input),float(last.Z),ir,gs,N,s,mom) for s in range(seeds)],0)
  ans[f'sse_{mom}']=float(np.square(pp-y).sum());ans[f'rmse_{mom}']=float(np.sqrt(np.mean(np.square(pp-y))))
 return ans
if __name__=='__main__':
 wells=list_wells('data/train');rng=np.random.RandomState(19);sample=[wells[i] for i in rng.permutation(len(wells))[:60]];st=time.time()
 rows=[x for x in Parallel(n_jobs=10)(delayed(one)(w) for w in sample) if x];n=sum(x['n'] for x in rows)
 scores={str(m):float(np.sqrt(sum(x[f'sse_{m}'] for x in rows)/n)) for m in MOMS};base=scores['0.998']
 wins={str(m):sum(x[f'rmse_{m}']<x['rmse_0.998'] for x in rows) for m in MOMS}
 out={'protocol':'fixed seed-19 60-well pilot; 100 particles x4 seeds; only MOM changed','wells':len(rows),'rows':n,'seconds':time.time()-st,'baseline_mom_0998':base,'scores':scores,'gains_vs_baseline':{k:base-v for k,v in scores.items()},'well_wins_vs_baseline':wins}
 O=ROOT/'exp/results/pf_transition_law';O.mkdir(parents=True,exist_ok=True);(O/'pilot_summary.json').write_text(json.dumps(out,indent=2));(O/'pilot_rows.json').write_text(json.dumps(rows));print(json.dumps(out,indent=2))
