from pathlib import Path
import json,numpy as np
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_expert_errors_v6';a=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');b=np.load(OUT/'oof.npz');W=b['wells'].astype(str);assert np.array_equal(W,a['wells'].astype(str));L=b['pred'].shape[1]
s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);base=.1*s['accepted']+.9*s['replacement'];y=s['y'];g=s['groups'].astype(str);fm={W[i]:k for k,(_,va) in enumerate(GroupKFold(5).split(W,groups=W)) for i in va}
rows=[]
for wt in [0,.25,.5,.75,1]:
 pp=[];yy=[];ff=[]
 C=(1-wt)*a['et']+wt*b['pred']
 for i,w in enumerate(W):
  m=g==w;n=m.sum();pp.append(base[m]+np.interp(np.linspace(0,1,n),np.linspace(0,1,L),C[i]));yy.append(y[m]);ff.append(np.full(n,fm[w]))
 pp,yy,ff=map(np.concatenate,(pp,yy,ff));rm=lambda m:float(np.sqrt(np.mean((yy[m]-pp[m])**2)));rows.append({'v6_weight':wt,'rmse':rm(np.ones(len(yy),bool)),'folds':[rm(ff==k) for k in range(5)]})
out={'grid':rows,'best':min(rows,key=lambda x:x['rmse']),'protocol':'fixed exact-row v1/v6 correction blends'};(OUT/'v1_v6_blend_exact.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
