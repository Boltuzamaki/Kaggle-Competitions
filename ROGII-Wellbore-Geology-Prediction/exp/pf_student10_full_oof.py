"""Generate full legal Student-t PF OOF predictions for ensemble evaluation."""
import os,sys,time,json
from pathlib import Path
import joblib
import numpy as np
from joblib import Parallel,delayed

HERE=os.path.dirname(os.path.abspath(__file__));ROOT=Path(HERE).parent
sys.path[:0]=[HERE,str(ROOT)]
from pf_decorr import pf_predict_decorr
from wellbore_lib import list_wells,load_well,ps_index

def one(w):
    h,tw=load_well("data/train",w);ps=ps_index(h)
    if ps<10 or ps>=len(h)-5:return None
    ev=h.TVT_input.isna().to_numpy()
    pred=pf_predict_decorr(h,tw,250,40,scale=10.,decorrelate=True,
                          seed0=41000,gs_mult=1.,gs_floor=45.,student_nu=10.)
    return w,{"row_index":np.flatnonzero(ev),"target":h.TVT.to_numpy(float)[ev],"prediction":pred[ev]}

if __name__=="__main__":
    wells=list_wells("data/train");t=time.time()
    pairs=[x for x in Parallel(n_jobs=12,verbose=5)(delayed(one)(w) for w in wells) if x]
    out=ROOT/"exp/results/pf_student10_full_oof";out.mkdir(parents=True,exist_ok=True)
    artifact={w:v for w,v in pairs};joblib.dump(artifact,out/"predictions.joblib",compress=3)
    y=np.concatenate([v["target"] for v in artifact.values()]);p=np.concatenate([v["prediction"] for v in artifact.values()])
    summary={"wells":len(artifact),"rows":len(y),"rmse":float(np.sqrt(np.mean((y-p)**2))),
             "seconds":time.time()-t,"parameters":{"particles":250,"reps":40,"seed0":41000,
             "gs_floor":45,"student_nu":10,"gs_mult":1}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
