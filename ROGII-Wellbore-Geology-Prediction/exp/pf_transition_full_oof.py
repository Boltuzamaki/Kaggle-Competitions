"""Full legal OOF-like tracker leg with locked MOM=0.9995 (no learned target use)."""
from pathlib import Path
import json,time,sys
import numpy as np
from joblib import Parallel,delayed
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'exp')]
from wellbore_lib import load_well,list_wells,ps_index
from pf_transition_law_cv import run
MOM=.9995
def one(w):
 h,t=load_well('data/train',w);ps=ps_index(h)
 if ps<20 or ps>=len(h)-5:return None
 kn=h.iloc[:ps];ev=h.iloc[ps:];ts=t.sort_values('TVT');tv=ts.TVT.to_numpy(float);tg=ts.GR.interpolate(limit_direction='both').to_numpy(float)
 last=kn.iloc[-1];at=np.interp(kn.TVT_input,tv,tg);gs=float(np.clip(np.nanstd(kn.GR.fillna(0)-at),10,60))
 tail=kn.tail(30);dm=np.diff(tail.MD);ir=float(np.median((np.diff(tail.TVT_input)+np.diff(tail.Z))[dm>0]/dm[dm>0]))
 gr=h.GR.interpolate(limit_direction='both').fillna(np.nanmean(tg)).to_numpy(float)[ps:]
 pred=np.mean([run(ev.MD.to_numpy(float),ev.Z.to_numpy(float),gr,tv,tg,float(last.TVT_input),float(last.Z),ir,gs,250,s,MOM) for s in range(8)],0)
 ids=np.asarray([f'{w}_{i}' for i in ev.index],object);true=h.TVT.to_numpy(float)[ps:]
 return ids,pred,true
if __name__=='__main__':
 wells=list_wells('data/train');st=time.time();res=[x for x in Parallel(n_jobs=10)(delayed(one)(w) for w in wells) if x]
 ids=np.concatenate([x[0] for x in res]);pred=np.concatenate([x[1] for x in res]);y=np.concatenate([x[2] for x in res])
 out={'protocol':'all legal train wells; locked MOM=.9995 from independent 120-well pilot; 250 particles x8 seeds','wells':len(res),'rows':len(y),'seconds':time.time()-st,'rmse':float(np.sqrt(np.mean(np.square(pred-y))))}
 O=ROOT/'exp/results/pf_transition_law';np.savez_compressed(O/'full_oof.npz',id=ids,prediction=pred,y=y);(O/'full_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
