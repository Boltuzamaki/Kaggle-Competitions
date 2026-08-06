from pathlib import Path
import hashlib,json,numpy as np
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_nested_v7';z=np.load(OUT/'oof.npz');W=z['wells'].astype(str);P=z['pred'];L=P.shape[1];s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);base=.1*s['accepted']+.9*s['replacement'];y=s['y'];g=s['groups'].astype(str);fm={W[i]:k for k,(_,va) in enumerate(GroupKFold(5).split(W,groups=W)) for i in va}
se=np.zeros(6);nn=np.zeros(6)
for i,w in enumerate(W):
 m=g==w;n=m.sum();p=base[m]+np.interp(np.linspace(0,1,n),np.linspace(0,1,L),P[i]);e=(y[m]-p)**2;se[5]+=e.sum();nn[5]+=n;se[fm[w]]+=e.sum();nn[fm[w]]+=n
out={'exact':float(np.sqrt(se[5]/nn[5])),'folds':[float(np.sqrt(se[k]/nn[k])) for k in range(5)],'sha256':hashlib.sha256((OUT/'oof.npz').read_bytes()).hexdigest()};(OUT/'independent_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
