"""Preregistered nested distributionally-robust stack over existing legal OOF legs."""
from pathlib import Path
import contextlib, hashlib, io, json, runpy
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

R=Path(__file__).resolve().parents[1]
OUT=R/'exp/results/legal_level2_robust_stack_v1'; OUT.mkdir(parents=True,exist_ok=True)
# Reconstruct the exact immutable leg matrix using the audited level-2 loader.
with contextlib.redirect_stdout(io.StringIO()):
    q=runpy.run_path(str(R/'exp/legal_level2_all_oof_v1.py'))
X=q['X']; names=q['names']; y=q['y']; base=q['base']; groups=q['groups']; W=q['W']; folds=q['folds']; ixs=q['ixs']
vi=names.index('v6'); cols=[j for j in range(len(names)) if j!=vi]
A=X[:,cols]-X[:,[vi]]; target=y-base-X[:,vi]
wf={w:f for f,(_,va) in enumerate(folds) for w in W[va]}
rf=np.array([wf[w] for w in groups])

def stats(wells):
    """Equal-well quadratic risks; prevents long wells dominating uncertainty."""
    ans=[]
    for w in wells:
        ix=ixs[w]; a=A[ix]; t=target[ix]
        ans.append((a.T@a/len(ix), a.T@t/len(ix), float(t@t/len(ix))))
    return ans

def fit(wells, rho, lam):
    ss=stats(wells); p=A.shape[1]; scale=np.mean([np.trace(G)/p for G,_,_ in ss])
    # Fixed deterministic four-block adversary, defined from training wells only.
    def agg(s): return (np.mean([x[0] for x in s],axis=0),np.mean([x[1] for x in s],axis=0),np.mean([x[2] for x in s]))
    mean=agg(ss); blocks=[agg(stats(b)) for b in np.array_split(np.asarray(wells),4)]
    def risk(z,s): G,h,c=s; return c-2*z@h+z@G@z
    # Convex epigraph formulation of mean + worst-block risk.
    def fun(v):
        z=v[:-1]; return (1-rho)*risk(z,mean)+rho*v[-1]+lam*scale*(z@z)
    def jac(v):
        z=v[:-1]; G,h,_=mean
        return np.r_[2*(1-rho)*(G@z-h)+2*lam*scale*z,rho]
    cons=[]
    for s in blocks:
        cons.append({'type':'ineq','fun':lambda v,s=s:v[-1]-risk(v[:-1],s),
                     'jac':lambda v,s=s:np.r_[-2*(s[0]@v[:-1]-s[1]),1.]})
    t0=max(risk(np.zeros(p),s) for s in blocks)
    res=minimize(fun,np.r_[np.zeros(p),t0],jac=jac,constraints=cons,method='SLSQP',
                 options={'maxiter':300,'ftol':1e-10})
    if not res.success: raise RuntimeError(res.message)
    return res.x[:-1]

# Three objective families and two conservative ridge levels were preregistered.
candidates=[(rho,lam) for rho in (0.0,0.25,0.5) for lam in (0.1,1.0)]
pred=np.zeros(len(y)); choices=[]
for fo,(tr,va) in enumerate(folds):
    train=W[tr]; inner=list(GroupKFold(4).split(train,groups=train)); scored=[]
    for rho,lam in candidates:
        fs=[]
        for it,iv in inner:
            z=fit(train[it],rho,lam); rows=np.concatenate([ixs[w] for w in train[iv]])
            fs.append(float(np.sqrt(np.mean((target[rows]-A[rows]@z)**2))))
        # Robust selection criterion fixed in advance: worst inner fold, then mean.
        scored.append((max(fs),np.mean(fs),rho,lam,fs))
    best=min(scored); z=fit(train,best[2],best[3]); rows=np.concatenate([ixs[w] for w in W[va]])
    pred[rows]=X[rows,vi]+A[rows]@z
    choices.append({'fold':fo,'rho':best[2],'lambda':best[3],
                    'inner_worst':best[0],'inner_mean':best[1],'inner_folds':best[4],
                    'weight_norm':float(np.linalg.norm(z))})

center=base+pred; stable=base+np.load(R/'exp/results/legal_level2_all_oof_v1/oof.npz')['anchored_difference']
def rm(a,m=None):
    if m is None:m=np.ones(len(y),bool)
    return float(np.sqrt(np.mean((y[m]-a[m])**2)))
fold_scores=[]
for f in range(5):
    m=rf==f; fold_scores.append({'fold':f,'baseline':rm(stable,m),'robust':rm(center,m),'gain':rm(stable,m)-rm(center,m)})
summary={'stable':rm(stable),'robust':rm(center),'folds':fold_scores,
         'fold_wins':sum(x['gain']>0 for x in fold_scores),'choices':choices,
         'legs':names,'candidates':[{'rho':r,'lambda':l} for r,l in candidates],
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
np.savez_compressed(OUT/'oof.npz',prediction=pred,groups=groups)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
