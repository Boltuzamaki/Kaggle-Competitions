"""Post-hoc diagnostic only: complementarity of two existing TabICL OOF gates."""
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
c=np.load(ROOT/'exp/results/continuous_well_moe/cpu_search_cache.npz',allow_pickle=True)
r=np.load(ROOT/'exp/results/tabicl_complete_well_gate/full_oof.npz')
w=np.load(ROOT/'exp/results/tabicl_oracle_weight_gate/oof.npz')
G,nrow,X,wn=c['G'],c['nrow'],c['X'],c['well']
base=np.zeros((len(wn),10));base[:,0]=1
q=r['predicted_regret']; R=np.exp(np.clip(-(q-q.min(1,keepdims=True))/.7,-20,0));R/=R.sum(1,keepdims=True)
W=np.maximum(w['predicted_weights'],0);W/=np.maximum(W.sum(1,keepdims=True),1e-8)
W[W.sum(1)==0,0]=1
def score(A,ids):
 z=np.einsum('ni,nij,nj->n',A,G[ids],A)
 return float(np.sqrt(np.sum(nrow[ids]*z)/np.sum(nrow[ids])))
splits=list(GroupKFold(5).split(X,groups=wn)); ids=np.arange(len(wn)); rows=[]
for total in (.1,.2,.3,.5,.7):
 for share in (0,.25,.5,.75,1):
  A=(1-total)*base+total*(share*R+(1-share)*W)
  gains=[score(base[va],va)-score(A[va],va) for _,va in splits]
  rows.append({'total_blend':total,'regret_share':share,'rmse':score(A,ids),'fold_gains':gains,'fold_wins':sum(x>0 for x in gains)})
out={'warning':'post-hoc diagnostic on same folds; not an honest promoted score','base':score(base,ids),'best':min(rows,key=lambda x:x['rmse']),'best_5of5':min((x for x in rows if x['fold_wins']==5),key=lambda x:x['rmse'],default=None),'grid':rows}
p=ROOT/'exp/results/tabicl_complete_well_gate/complementarity.json';p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
