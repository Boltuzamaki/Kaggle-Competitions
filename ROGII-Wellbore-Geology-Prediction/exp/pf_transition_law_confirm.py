"""Higher-fidelity confirmation of the PF momentum pilot."""
from pathlib import Path
import json,time,sys
import numpy as np
from joblib import Parallel,delayed
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'exp')]
from wellbore_lib import list_wells
from pf_transition_law_cv import one
MOMS=(.998,.999,.9995,.9998,.9999,1.0)
if __name__=='__main__':
 wells=list_wells('data/train');rng=np.random.RandomState(7);sample=[wells[i] for i in rng.permutation(len(wells))[:120]];st=time.time()
 rows=[x for x in Parallel(n_jobs=10)(delayed(one)(w,200,8,MOMS) for w in sample) if x];n=sum(x['n'] for x in rows)
 scores={str(m):float(np.sqrt(sum(x[f'sse_{m}'] for x in rows)/n)) for m in MOMS};base=scores['0.998']
 out={'protocol':'fixed seed-7 120-well confirmation; 200 particles x8 seeds; only MOM changed','wells':len(rows),'rows':n,'seconds':time.time()-st,
      'baseline_mom_0998':base,'scores':scores,'gains_vs_baseline':{k:base-v for k,v in scores.items()},
      'well_wins_vs_baseline':{str(m):sum(x[f'rmse_{m}']<x['rmse_0.998'] for x in rows) for m in MOMS}}
 O=ROOT/'exp/results/pf_transition_law';(O/'confirm_summary.json').write_text(json.dumps(out,indent=2));(O/'confirm_rows.json').write_text(json.dumps(rows));print(json.dumps(out,indent=2))
