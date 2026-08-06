"""Independent 120-well confirmation of the paper-derived PF likelihoods."""
import os,sys,time
import numpy as np
from joblib import Parallel,delayed
HERE=os.path.dirname(os.path.abspath(__file__));sys.path[:0]=[HERE,os.path.dirname(HERE)]
from pf_decorr import pf_predict_decorr
from wellbore_lib import list_wells,load_well,ps_index

CONFIGS=(("gauss_adaptive",0.,0.),("gauss_floor30",30.,0.),
         ("student3_floor45",45.,3.),("student10_floor45",45.,10.))
def one(w):
 h,tw=load_well("data/train",w);ps=ps_index(h)
 if ps<10 or ps>=len(h)-5:return None
 ev=h.TVT_input.isna().to_numpy();o={"y":h.TVT.to_numpy(float)[ev]}
 for name,floor,nu in CONFIGS:
  o[name]=pf_predict_decorr(h,tw,250,40,scale=10.,decorrelate=True,
   seed0=41000,gs_mult=1.,gs_floor=floor,student_nu=nu)[ev]
 return o
def score(rows,key):return float(np.sqrt(np.mean(np.concatenate([(r["y"]-r[key])**2 for r in rows]))))
if __name__=="__main__":
 n=int(sys.argv[1]) if len(sys.argv)>1 else 120;ws=list_wells("data/train")
 sample=[ws[i] for i in np.random.RandomState(73).permutation(len(ws))[:n]];t=time.time()
 rows=[r for r in Parallel(n_jobs=12)(delayed(one)(w) for w in sample) if r]
 print(f"wells={len(rows)} seconds={time.time()-t:.0f}")
 for name,_,_ in CONFIGS:print(f"{name:22s} {score(rows,name):.6f}")
