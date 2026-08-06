from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; run=ROOT/'exp/results'/sys.argv[1]
z=np.load(run/'well_curves.npz'); W=z['wells'].astype(str); P=z['et'];L=len(P[0])
s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True)
base=.1*s['accepted']+.9*s['replacement']; y=s['y']; groups=s['groups'].astype(str)
pp=[];yy=[];bb=[]; fold=[]
from sklearn.model_selection import GroupKFold
fm={W[i]:k for k,(_,va) in enumerate(GroupKFold(5).split(W,groups=W)) for i in va}
for i,w in enumerate(W):
 m=groups==w;n=m.sum();corr=np.interp(np.linspace(0,1,n),np.linspace(0,1,L),P[i])
 pp.append(base[m]+corr);yy.append(y[m]);bb.append(base[m]);fold.append(np.full(n,fm[w]))
pp=np.concatenate(pp);yy=np.concatenate(yy);bb=np.concatenate(bb);fold=np.concatenate(fold)
rm=lambda p,m=np.ones(len(yy),bool):float(np.sqrt(np.mean((yy[m]-p[m])**2)))
out={'baseline':rm(bb),'et_full':rm(pp),'folds':[{'fold':k,'baseline':rm(bb,fold==k),'et_full':rm(pp,fold==k),'gain':rm(bb,fold==k)-rm(pp,fold==k)} for k in range(5)],
 'npz_sha256':hashlib.sha256((run/'well_curves.npz').read_bytes()).hexdigest(),
 'summary_sha256':hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()}
print(json.dumps(out,indent=2));(run/'independent_verification.json').write_text(json.dumps(out,indent=2))
