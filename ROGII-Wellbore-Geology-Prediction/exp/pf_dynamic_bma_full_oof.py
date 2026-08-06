"""Resumable full-field locked dynamic-BMA PF OOF generation."""
from pathlib import Path
import json,sys,time,joblib
import numpy as np
from joblib import Parallel,delayed

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'exp'),str(ROOT)]
import pf_dynamic_bma_cv as bma
from wellbore_lib import list_wells,load_well,ps_index

OUT=ROOT/'exp/results/pf_dynamic_bma_full_oof';PARTS=OUT/'parts';PARTS.mkdir(parents=True,exist_ok=True)
bma.N_PARTICLES=250;bma.N_SEEDS=40;bma.SEED0=63000
ALPHA=.98

def work(w):
 path=PARTS/f'{w}.joblib'
 if path.exists():return w,'cached'
 h,tw=load_well('data/train',w);ps=ps_index(h)
 if ps<10 or ps>=len(h)-5:return w,'skip'
 ev=h.TVT_input.isna().to_numpy();pred,q=bma.predict(h,tw,ALPHA)
 obj={'well':w,'row_index':np.flatnonzero(ev),'prediction':pred.astype(np.float32),
      'target':h.TVT.to_numpy(float)[ev].astype(np.float32),'final_weights':np.asarray(q,np.float32)}
 tmp=path.with_suffix('.tmp');joblib.dump(obj,tmp,compress=3);tmp.replace(path)
 return w,'done'

if __name__=='__main__':
 wells=list_wells('data/train');t=time.time()
 rows=Parallel(n_jobs=12,verbose=10)(delayed(work)(w) for w in wells)
 artifacts={}
 for w in wells:
  p=PARTS/f'{w}.joblib'
  if p.exists():artifacts[w]=joblib.load(p)
 if len(artifacts)!=len(wells):raise RuntimeError(f'only {len(artifacts)}/{len(wells)} complete')
 y=np.concatenate([artifacts[w]['target'] for w in wells]);p=np.concatenate([artifacts[w]['prediction'] for w in wells])
 summary={'wells':len(wells),'rows':len(y),'rmse':float(np.sqrt(np.mean((y-p)**2))),
  'alpha':ALPHA,'particles':bma.N_PARTICLES,'seeds':bma.N_SEEDS,'seed0':bma.SEED0,
  'mean_final_weights':np.mean([artifacts[w]['final_weights'] for w in wells],axis=0).tolist(),
  'seconds':time.time()-t,'resumable_parts':True}
 joblib.dump(artifacts,OUT/'predictions.joblib',compress=3)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
