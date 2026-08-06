"""Dynamic Bayesian averaging of robust GR likelihoods inside the PF.

Development selects one forgetting factor on 60 fixed wells.  The selected
factor is then frozen and evaluated on 120 disjoint confirmation wells.  All PF
inputs are legal at inference; TVT is read only for terminal scoring.
"""
from pathlib import Path
import json, sys, time
import numpy as np
from joblib import Parallel, delayed

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exp'),str(ROOT)]
from wellbore_lib import list_wells,load_well,ps_index

OUT=ROOT/'exp/results/pf_dynamic_bma';OUT.mkdir(parents=True,exist_ok=True)
ALPHAS=(.80,.90,.95,.98)
N_PARTICLES=200;N_SEEDS=12;SEED0=53000


def run(md,z,gr,tt,tg,ls,ir,gs0,n,seed,alpha=None):
 rng=np.random.default_rng(seed);pos=ls+rng.uniform(1.6,2.4)*rng.standard_normal(n)
 rate=ir+.01*rng.standard_normal(n);w=np.ones(n)/n;q=np.ones(3)/3
 lo,hi=tt[0]-60,tt[-1]+60;out=np.empty(len(md));ll=0.;pm=md[0]-1
 # One tight dynamics draw is shared by all likelihood models.
 mom=.998;vn=.002*rng.uniform(.93,1.08);pn=.005*rng.uniform(.88,1.15)
 rp=.1*rng.uniform(.85,1.2);resamp=rng.uniform(.47,.55);init_q=np.ones(3)/3
 for i in range(len(md)):
  dm=max(md[i]-pm,1.);rate=mom*rate+vn*rng.standard_normal(n)
  pos=pos+rate*dm+pn*rng.standard_normal(n);tvt=np.clip(pos-z[i],lo,hi);pos=tvt+z[i]
  if np.isfinite(gr[i]):
   residual=gr[i]-np.interp(tvt,tt,tg)
   scales=(max(gs0,30.),max(gs0,45.),max(gs0,45.));nus=(0.,3.,10.)
   lk=[]
   for scale,nu in zip(scales,nus):
    d=residual/scale
    v=np.exp(-.5*np.minimum(d*d,600)) if nu==0 else (1+d*d/nu)**(-.5*(nu+1))
    lk.append(np.maximum(v,1e-300))
   lk=np.asarray(lk)
   if alpha is None: mix=lk[2] # locked Student-t10 comparator
   else:
    evidence=np.maximum(lk@w,1e-300)
    logq=alpha*np.log(np.maximum(q,1e-300))+np.log(evidence)
    logq-=logq.max();q=np.exp(logq);q/=q.sum()
    mix=q@lk
   avg=max(float(w@mix),1e-300);ll+=np.log(avg);w*=mix;w/=max(w.sum(),1e-300)
  if 1/(w@w)<resamp*n:
   idx=np.clip(np.searchsorted(np.cumsum(w),rng.uniform(0,1/n)+np.arange(n)/n),0,n-1)
   pos=pos[idx]+rp*rng.standard_normal(n);rate=rate[idx]+.001*rng.standard_normal(n);w.fill(1/n)
  out[i]=w@(pos-z[i]);pm=md[i]
 return out,ll,q if alpha is not None else init_q


def predict(h,tw,alpha):
 tw=tw.sort_values('TVT');tt=tw.TVT.to_numpy(float);tg=tw.GR.fillna(tw.GR.mean()).to_numpy(float)
 kn=h[h.TVT_input.notna()];ev=h[h.TVT_input.isna()];idx=ev.index.to_numpy();last=kn.iloc[-1]
 at=np.interp(kn.TVT_input,tt,tg);gs0=float(np.clip(np.nanstd(kn.GR.fillna(0)-at),10,60))
 tail=kn.tail(30);dm=np.diff(tail.MD);valid=dm>0
 ir=float(np.median((np.diff(tail.TVT_input)+np.diff(tail.Z))[valid]/dm[valid])) if valid.sum()>=3 else 0.
 ga=h.GR.interpolate(limit_direction='both').fillna(np.nanmean(tg)).to_numpy(float)
 md=np.r_[float(last.MD),ev.MD.to_numpy(float)];z=np.r_[float(last.Z),ev.Z.to_numpy(float)];gr=np.r_[np.nan,ga[idx]]
 curves=[];ll=[];qs=[]
 for s in range(N_SEEDS):
  p,l,q=run(md,z,gr,tt,tg,float(last.TVT_input+last.Z),ir,gs0,N_PARTICLES,SEED0+s,alpha)
  curves.append(p);ll.append(l);qs.append(q)
 ll=np.asarray(ll);ww=np.exp((ll-ll.max())/10);ww/=ww.sum()
 return (ww@np.asarray(curves)) [1:],np.mean(qs,axis=0)


def one(w,alphas):
 h,tw=load_well('data/train',w);ps=ps_index(h)
 if ps<10 or ps>=len(h)-5:return None
 ev=h.TVT_input.isna().to_numpy();row={'well':w,'y':h.TVT.to_numpy(float)[ev]}
 row['student10'],_=predict(h,tw,None)
 for a in alphas:row[f'a{a:.2f}'],row[f'q{a:.2f}']=predict(h,tw,a)
 return row


def score(rows,key):return float(np.sqrt(np.mean(np.concatenate([(r['y']-r[key])**2 for r in rows]))))
def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


if __name__=='__main__':
 wells=list_wells('data/train');order=[wells[i] for i in np.random.RandomState(37).permutation(len(wells))]
 dev=order[:60];remaining=[w for w in wells if w not in set(dev)]
 conf=[remaining[i] for i in np.random.RandomState(73).permutation(len(remaining))[:120]]
 t=time.time();dr=[r for r in Parallel(n_jobs=12)(delayed(one)(w,ALPHAS) for w in dev) if r]
 ds={'student10':score(dr,'student10')}
 for a in ALPHAS:ds[f'a{a:.2f}']=score(dr,f'a{a:.2f}')
 best=min(ALPHAS,key=lambda a:ds[f'a{a:.2f}'])
 # Only the locked alpha and comparator are computed on untouched confirmation.
 cr=[r for r in Parallel(n_jobs=12)(delayed(one)(w,(best,)) for w in conf) if r]
 c0=score(cr,'student10');cb=score(cr,f'a{best:.2f}')
 q=np.mean([r[f'q{best:.2f}'] for r in cr],axis=0)
 result={'development_wells':len(dr),'confirmation_wells':len(cr),'alphas':ALPHAS,
  'development_scores':ds,'locked_alpha':best,'confirmation_student10':c0,
  'confirmation_dynamic_bma':cb,'confirmation_gain':c0-cb,
  'confirmation_well_wins':sum(rmse(r['y'],r[f'a{best:.2f}'])<rmse(r['y'],r['student10']) for r in cr),
  'mean_final_model_weights':{'gaussian30':q[0],'student3_45':q[1],'student10_45':q[2]},
  'particles':N_PARTICLES,'seeds':N_SEEDS,'seed0':SEED0,'runtime_seconds':time.time()-t,
  'legal':True,'submission_created':False}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2,default=float));print(json.dumps(result,indent=2,default=float))
