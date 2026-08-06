"""Immutable causal-prefix-v1 multimodal residual curve mixture, strict outer GKF."""
from pathlib import Path
import hashlib,json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz'
OUT=ROOT/'exp/results/multimodal_residual_curve_mixture_v1';OUT.mkdir(parents=True,exist_ok=True)
Z=np.load(SRC,allow_pickle=True); W=Z['wells'].astype(str); C=Z['true'];Q=Z['query']; lens=Z['length']
folds=list(GroupKFold(5).split(Q,groups=W))

def predict(K,leaf):
    mean=np.zeros_like(C); mode=np.zeros_like(C); conf=np.zeros(len(W)); ent=np.zeros(len(W))
    for f,(tr,va) in enumerate(folds):
        qs=StandardScaler().fit(Q[tr]); a=qs.transform(Q[tr]);b=qs.transform(Q[va])
        qp=PCA(n_components=32,whiten=True,svd_solver='randomized',random_state=71).fit(a);a=qp.transform(a);b=qp.transform(b)
        cp=PCA(n_components=20,random_state=72).fit(C[tr]); lat=cp.transform(C[tr])
        km=KMeans(K,n_init=30,random_state=73).fit(lat); lab=km.labels_
        centers=np.array([C[tr][lab==k].mean(0) for k in range(K)])
        clf=ExtraTreesClassifier(n_estimators=50,min_samples_leaf=leaf,max_features=.7,
             class_weight='balanced',n_jobs=4,random_state=74).fit(a,lab)
        pp=clf.predict_proba(b); full=np.zeros((len(va),K));full[:,clf.classes_.astype(int)]=pp
        mean[va]=full@centers;mode[va]=centers[np.argmax(full,1)]
        conf[va]=full.max(1);ent[va]=-(full*np.log(full+1e-12)).sum(1)/np.log(K)
    return mean,mode,conf,ent

# Exact row reconstruction in the immutable Student baseline order.
s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True)
y=s['y']; groups=s['groups'].astype(str);base=.1*s['accepted']+.9*s['replacement']
cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)]
ROWIX={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}
assert len(ROWIX)==len(W)
def rows(P):
    out=np.empty_like(y)
    for w,c in zip(W,P):
        ix=ROWIX[w];out[ix]=np.interp(np.linspace(0,1,len(ix)),np.linspace(0,1,len(c)),c)
    return out
def rm(p):return float(np.sqrt(np.mean((y-p)**2)))
basefold=[]
wf={w:f for f,(_,va) in enumerate(folds) for w in W[va]}
rf=np.array([wf[w] for w in groups])
for f in range(5):basefold.append(rm(base[rf==f]) if False else float(np.sqrt(np.mean((y[rf==f]-base[rf==f])**2))))

grid=[]; saved={}
for K in [8]:
 for leaf in [16]:
    print('fit',K,leaf,flush=True)
    pm,ph,cf,en=predict(K,leaf); rmean=rows(pm);rmode=rows(ph)
    # Confidence gate is centered: uncertain wells shrink toward zero correction.
    gates={'plain':np.ones(len(W)),'conf':np.clip((cf-1/K)/(1-1/K),0,1),
           'conf2':np.clip((cf-1/K)/(1-1/K),0,1)**2}
    for dec,rp in [('mean',rmean),('mode',rmode)]:
      for gate,g in gates.items():
        rg=np.empty_like(y)
        for w,gv in zip(W,g):rg[ROWIX[w]]=gv
        for alpha in [.1,.2,.35,.5,.75,1.0]:
          predv=base+alpha*rp*rg; fs=[float(np.sqrt(np.mean((y[rf==f]-predv[rf==f])**2))) for f in range(5)]
          grid.append(dict(K=K,leaf=leaf,decoder=dec,gate=gate,alpha=alpha,rmse=rm(predv),
                           wins=sum(fs[f]<basefold[f] for f in range(5)),**{f'f{f}':fs[f] for f in range(5)}))
    saved[(K,leaf)]=(pm,ph,cf,en)
d=pd.DataFrame(grid).sort_values('rmse');d.to_csv(OUT/'grid.csv',index=False)
best=d.iloc[0].to_dict(); key=(int(best['K']),int(best['leaf']));pm,ph,cf,en=saved[key]
np.savez_compressed(OUT/'predictions.npz',wells=W,posterior_mean=pm,map_mode=ph,confidence=cf,entropy=en)
summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'baseline':rm(base),'best':best,
 'best_5of5':d[d.wins==5].iloc[0].to_dict() if (d.wins==5).any() else None,
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
