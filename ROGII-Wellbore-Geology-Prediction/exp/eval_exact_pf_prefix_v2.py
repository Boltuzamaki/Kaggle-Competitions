from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/generative_curve_prior_exact_pf_v2'
z=np.load(ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=z['wells'].astype(str);C=z['true'];Q=z['query'];lens=z['length']
d=pd.read_csv(OUT/'pilot_80_features.csv',dtype={'well':str}).set_index('well');d=d.loc[d.index.intersection(W)]
ii=np.array([np.where(W==w)[0][0] for w in d.index]);Q=Q[ii];C=C[ii];lens=lens[ii]
E=d.to_numpy(); pred0=np.zeros_like(C);pred1=np.zeros_like(C)
for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=2202).split(Q)):
 cp=PCA(16,random_state=1).fit(C[tr]);y=cp.transform(C[tr])
 for X,P in [(Q,pred0),(np.c_[Q,E],pred1)]:
  sc=StandardScaler().fit(X[tr]);m=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=6,max_features=.65,n_jobs=-1,random_state=fold).fit(sc.transform(X[tr]),y)
  P[va]=cp.inverse_transform(m.predict(sc.transform(X[va])))
rm=lambda p:float(np.sqrt(np.average(np.mean((C-p)**2,1),weights=lens)))
out={'wells':len(d),'zero':rm(0*C),'base_features':rm(pred0),'plus_exact_pf':rm(pred1),'gain':rm(pred0)-rm(pred1),
 'features_sha256':hashlib.sha256((OUT/'pilot_80_features.csv').read_bytes()).hexdigest()}
print(json.dumps(out,indent=2));(OUT/'pilot_80_eval.json').write_text(json.dumps(out,indent=2))
