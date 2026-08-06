"""Private research kernel: locked alpha=.98 dynamic-BMA full training OOF."""
import glob,os,json,time
from pathlib import Path
import numpy as np,pandas as pd
from joblib import Parallel,delayed

horizontal=glob.glob('/kaggle/input/**/*__horizontal_well.csv',recursive=True)
if not horizontal: raise RuntimeError('competition horizontal files not found under /kaggle/input')
# Prefer the populated training directory over the three-row placeholder test source.
parents={Path(p).parent for p in horizontal}
TRAIN=max(parents,key=lambda p:len(list(p.glob('*__horizontal_well.csv'))))
OUT=Path('/kaggle/working'); A=.98;NP=250;NS=40;SEED0=63000

def run(md,z,gr,tt,tg,ls,ir,gs0,n,seed):
 r=np.random.default_rng(seed);pos=ls+r.uniform(1.6,2.4)*r.standard_normal(n);rate=ir+.01*r.standard_normal(n)
 w=np.ones(n)/n;q=np.ones(3)/3;lo,hi=tt[0]-60,tt[-1]+60;o=np.empty(len(md));ll=0.;pm=md[0]-1
 vn=.002*r.uniform(.93,1.08);pn=.005*r.uniform(.88,1.15);rp=.1*r.uniform(.85,1.2);rs=r.uniform(.47,.55)
 for i in range(len(md)):
  dm=max(md[i]-pm,1.);rate=.998*rate+vn*r.standard_normal(n);pos+=rate*dm+pn*r.standard_normal(n)
  tv=np.clip(pos-z[i],lo,hi);pos=tv+z[i]
  if np.isfinite(gr[i]):
   e=gr[i]-np.interp(tv,tt,tg);lk=[]
   for sc,nu in ((max(gs0,30),0),(max(gs0,45),3),(max(gs0,45),10)):
    d=e/sc;v=np.exp(-.5*np.minimum(d*d,600)) if nu==0 else (1+d*d/nu)**(-.5*(nu+1));lk.append(np.maximum(v,1e-300))
   lk=np.asarray(lk);evid=np.maximum(lk@w,1e-300);lq=A*np.log(np.maximum(q,1e-300))+np.log(evid);lq-=lq.max();q=np.exp(lq);q/=q.sum();mix=q@lk
   av=max(float(w@mix),1e-300);ll+=np.log(av);w*=mix;w/=max(w.sum(),1e-300)
  if 1/(w@w)<rs*n:
   ix=np.clip(np.searchsorted(np.cumsum(w),r.uniform(0,1/n)+np.arange(n)/n),0,n-1);pos=pos[ix]+rp*r.standard_normal(n);rate=rate[ix]+.001*r.standard_normal(n);w.fill(1/n)
  o[i]=w@(pos-z[i]);pm=md[i]
 return o,ll,q

def one(well):
 h=pd.read_csv(TRAIN/f'{well}__horizontal_well.csv');tw=pd.read_csv(TRAIN/f'{well}__typewell.csv').sort_values('TVT')
 kn=h[h.TVT_input.notna()];ev=h[h.TVT_input.isna()];ix=ev.index.to_numpy();last=kn.iloc[-1]
 tt=tw.TVT.to_numpy(float);tg=tw.GR.fillna(tw.GR.mean()).to_numpy(float);at=np.interp(kn.TVT_input,tt,tg)
 gs=float(np.clip(np.nanstd(kn.GR.fillna(0)-at),10,60));ta=kn.tail(30);dm=np.diff(ta.MD);m=dm>0
 ir=float(np.median((np.diff(ta.TVT_input)+np.diff(ta.Z))[m]/dm[m])) if m.sum()>=3 else 0
 ga=h.GR.interpolate(limit_direction='both').fillna(np.nanmean(tg)).to_numpy(float)
 md=np.r_[float(last.MD),ev.MD];z=np.r_[float(last.Z),ev.Z];gr=np.r_[np.nan,ga[ix]];ps=[];ll=[];qs=[]
 for s in range(NS):
  p,l,q=run(md,z,gr,tt,tg,float(last.TVT_input+last.Z),ir,gs,NP,SEED0+s);ps.append(p);ll.append(l);qs.append(q)
 ll=np.asarray(ll);wt=np.exp((ll-ll.max())/10);wt/=wt.sum();p=(wt@np.asarray(ps))[1:]
 return {'well':well,'row_index':ix.tolist(),'prediction':p.astype(float).tolist(),
  'target':h.TVT.to_numpy(float)[ix].tolist(),'final_weights':np.mean(qs,0).tolist()}

t=time.time();wells=sorted({Path(f).name.split('__')[0] for f in glob.glob(str(TRAIN/'*__horizontal_well.csv'))})
rows=Parallel(n_jobs=4,verbose=10)(delayed(one)(w) for w in wells)
y=np.concatenate([r['target'] for r in rows]);p=np.concatenate([r['prediction'] for r in rows])
(OUT/'dynamic_bma_oof.json').write_text(json.dumps(rows))
summary={'wells':len(rows),'rows':len(y),'rmse':float(np.sqrt(np.mean((y-p)**2))),'alpha':A,'particles':NP,'seeds':NS,'seconds':time.time()-t,'submission_created':False}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
