"""Exact locked Student-t PF replay at historical TVT_input cutpoints."""
from pathlib import Path
import sys,time,json
import numpy as np,pandas as pd
from joblib import Parallel,delayed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'exp'))
from pf_decorr import pf_predict_decorr
from wellbore_lib import load_well,list_wells,ps_index
OUT=ROOT/'exp/results/generative_curve_prior_exact_pf_v2';OUT.mkdir(parents=True,exist_ok=True)
def one(w):
 h,tw=load_well(ROOT/'data/train',w); ps=ps_index(h); row={'well':w}
 for frac in [.60,.78,.90]:
  cut=max(30,int(ps*frac)); q=h.copy();q.loc[cut:,'TVT_input']=np.nan
  p=pf_predict_decorr(q,tw,n_particles=250,n_seeds=40,scale=10.,decorrelate=True,
      seed0=41000,gs_mult=1.,gs_floor=45.,student_nu=10.)
  r=h.TVT.to_numpy()[cut:ps]-p[cut:ps]; tag=str(int(frac*100))
  x=np.linspace(-1,0,len(r));co=np.polyfit(x,r,min(2,len(r)-1))
  row.update({f'pf{tag}_bias':float(r.mean()),f'pf{tag}_end':float(r[-1]),
   f'pf{tag}_rmse':float(np.sqrt(np.mean(r*r))),f'pf{tag}_slope':float(co[-2]),
   f'pf{tag}_curve':float(co[-3]) if len(co)>2 else 0.})
 return row
n=int(sys.argv[1]) if len(sys.argv)>1 else 80; wells=list_wells(ROOT/'data/train')
wells=list(np.random.RandomState(2201).choice(wells,min(n,len(wells)),False));t=time.time()
rows=Parallel(n_jobs=12,verbose=10)(delayed(one)(w) for w in wells)
pd.DataFrame(rows).sort_values('well').to_csv(OUT/f'pilot_{n}_features.csv',index=False)
meta={'wells':n,'seconds':time.time()-t,'particles':250,'seeds':40,'seed0':41000,'cuts':[.60,.78,.90]}
(OUT/f'pilot_{n}_meta.json').write_text(json.dumps(meta,indent=2));print(meta)
