"""Query-specific spatial/trajectory/typewell residual-manifold experts."""
from pathlib import Path
import json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/local_analogue_curve_prior';OUT.mkdir(parents=True,exist_ok=True)
z=np.load(ROOT/'exp/results/generative_curve_prior_retrieval/well_curves.npz')
W=z['wells'];C=z['true'];Q=z['query'];lens=z['length'];globalp=z['et']
folds=list(GroupKFold(5).split(Q,groups=W)); pilot=len(sys.argv)>1 and sys.argv[1]=='pilot'
pred=np.zeros_like(C); used=np.zeros(len(W),bool); rows=[]
# Context excludes model predictions and uses visible prefix + absolute pose + typewell.
CTX=slice(294,None)
for fold,(tr,va0) in enumerate(folds):
    if pilot and fold: continue
    va=va0[:60] if pilot else va0; used[va]=True
    cs=StandardScaler().fit(Q[tr,CTX]); A=cs.transform(Q[tr,CTX]);B=cs.transform(Q[va,CTX])
    fs=StandardScaler().fit(Q[tr]);FA=fs.transform(Q[tr]);FB=fs.transform(Q[va])
    cp=PCA(24,random_state=71).fit(C[tr]); latent=cp.transform(C[tr])
    for j,v in enumerate(va):
        dist=np.mean((A-B[j])**2,axis=1); near=np.argsort(dist)[:120]
        # Query-specific expert. Local coordinates are implicit differences to
        # the query vector, preventing absolute field coordinates dominating.
        X=FA[near]-FB[j]; x=np.zeros((1,Q.shape[1]))
        m=ExtraTreesRegressor(n_estimators=180,min_samples_leaf=8,max_features=.65,n_jobs=-1,random_state=900+v)
        m.fit(X,latent[near]); pred[v]=cp.inverse_transform(m.predict(x))[0]
    def score(P): return float(np.sqrt(np.average(np.mean((C[va]-P[va])**2,1),weights=lens[va])))
    rows.append({'fold':fold,'wells':len(va),'zero':score(np.zeros_like(C)),'global':score(globalp),'local':score(pred)})
    print(rows[-1],flush=True)
summary={'mode':'pilot' if pilot else 'full','folds':rows,
 'weighted_zero':float(np.sqrt(np.average(np.mean(C[used]**2,1),weights=lens[used]))),
 'weighted_global':float(np.sqrt(np.average(np.mean((C[used]-globalp[used])**2,1),weights=lens[used]))),
 'weighted_local':float(np.sqrt(np.average(np.mean((C[used]-pred[used])**2,1),weights=lens[used])))}
summary['gain_vs_global']=summary['weighted_global']-summary['weighted_local']
print(json.dumps(summary,indent=2));(OUT/f"{summary['mode']}.json").write_text(json.dumps(summary,indent=2))
np.savez_compressed(OUT/f"{summary['mode']}.npz",pred=pred,used=used)
