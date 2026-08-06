from pathlib import Path
import hashlib,json,numpy as np
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_expert_errors_v6';z=np.load(OUT/'oof.npz');W=z['wells'].astype(str);P=z['pred'];L=P.shape[1]
s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);base=.1*s['accepted']+.9*s['replacement'];y=s['y'];g=s['groups'].astype(str);fm={W[i]:k for k,(_,va) in enumerate(GroupKFold(5).split(W,groups=W)) for i in va}
pp=[];bb=[];yy=[];ff=[]
for i,w in enumerate(W):
 m=g==w;n=m.sum();r=np.interp(np.linspace(0,1,n),np.linspace(0,1,L),P[i]);pp.append(base[m]+r);bb.append(base[m]);yy.append(y[m]);ff.append(np.full(n,fm[w]))
pp,bb,yy,ff=map(np.concatenate,(pp,bb,yy,ff));rm=lambda p,m:float(np.sqrt(np.mean((yy[m]-p[m])**2)));allm=np.ones(len(yy),bool)
out={'baseline':rm(bb,allm),'v6':rm(pp,allm),'folds':[{'fold':k,'baseline':rm(bb,ff==k),'v6':rm(pp,ff==k),'gain':rm(bb,ff==k)-rm(pp,ff==k)} for k in range(5)],'oof_sha256':hashlib.sha256((OUT/'oof.npz').read_bytes()).hexdigest()};(OUT/'independent_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
