"""Within-well cubic GR path optimization around honest Stack-V4 OOF.

No cross-well learned mapping: each well uses its visible TVT_input prefix for
GR calibration and its inference-available hidden horizontal GR for likelihood.
Truth is read only after optimization for scoring/oracle diagnostics.
"""
from pathlib import Path
import json, joblib, sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/cubic_gr_path_optimizer"; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/"exp/kaggle_heel_gr_dataset"))
from heel_gr_datum import _robust_affine

N = int(sys.argv[1]) if len(sys.argv)>1 else 200
f = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
sv = np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"],float)
wlist = f.well.drop_duplicates().iloc[:N].tolist()
rng = np.random.default_rng(781)
scales = np.array([28.,20.,12.,8.])

def basis(x):
    return np.c_[np.ones(len(x)),x,.5*(3*x*x-1),.5*(5*x**3-3*x)]

def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

rows=[]; all_y=[]; all_b=[]; all_p=[]; all_or=[]
for wi,w in enumerate(wlist):
    ix=np.asarray(f.index[f.well==w]); qf=f.loc[ix]
    h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
    t=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
    tv=t.TVT.to_numpy(float)
    ts=pd.to_numeric(t.GR,errors="coerce").interpolate(limit_direction="both")
    tg=ts.fillna(ts.median() if ts.notna().any() else 0).to_numpy(float)
    hs=pd.to_numeric(h.GR,errors="coerce").interpolate(limit_direction="both")
    hgall=hs.fillna(hs.median() if hs.notna().any() else 0).to_numpy(float)
    vis=h.TVT_input.notna().to_numpy()
    alpha,beta,noise=_robust_affine(np.interp(h.loc[vis,"TVT_input"],tv,tg),hgall[vis])
    rowmap={f"{w}_{j}":j for j in h.index}
    hr=np.array([rowmap[x] for x in qf.id],int)
    base_abs=qf.last_known_tvt.to_numpy(float)+sv[ix]
    truth=qf.last_known_tvt.to_numpy(float)+qf.target.to_numpy(float)
    x=np.linspace(-1,1,len(ix)); B=basis(x)
    take=np.linspace(0,len(ix)-1,min(700,len(ix))).astype(int)
    Bt=B[take]; obs0=hgall[hr]
    # Multiscale full horizontal curves, then select hidden scoring rows.
    obs_sc=[obs0[take],gaussian_filter1d(obs0,5)[take],
            gaussian_filter1d(obs0,15)[take]]
    tg_sc=[tg,gaussian_filter1d(tg,2),gaussian_filter1d(tg,6)]
    scale=max(noise,5.)

    def objective(c):
        path=base_abs[take]+Bt@c
        val=0.
        for obs,g in zip(obs_sc,tg_sc):
            z=(obs-(alpha*np.interp(path,tv,g)+beta))/scale
            val += np.mean(np.log1p((z/2)**2))
        val/=len(obs_sc)
        val += .018*np.sum((np.asarray(c)/scales)**2)
        return float(val)

    # Broad deterministic prior draws plus local refinements of distinct minima.
    cand=rng.normal(size=(320,4))*scales
    cand=np.vstack([np.zeros(4),cand,np.c_[np.linspace(-50,50,41),np.zeros((41,3))]])
    obj=np.array([objective(c) for c in cand])
    starts=cand[np.argsort(obj)[:6]]
    sols=[]
    for st in starts:
        z=minimize(objective,st,method="Powell",
                   bounds=[(-60,60),(-45,45),(-25,25),(-15,15)],
                   options={"maxiter":70,"xtol":.15,"ftol":2e-4})
        sols.append((z.fun,z.x))
    sols=sorted(sols,key=lambda a:a[0])
    vals=np.array([a[0] for a in sols]); co=np.array([a[1] for a in sols])
    # Posterior over minima, then ambiguity hedge based on coefficient dispersion.
    post=np.exp(-(vals-vals.min())/.025);post/=post.sum()
    cmean=post@co
    disp=np.sqrt(np.sum(post[:,None]*(co-cmean)**2))
    hedge=1/(1+(disp/18)**2)
    chosen=hedge*cmean
    pred=base_abs+B@chosen
    # Legal family oracle and objective/rank diagnostic over candidate bank.
    oracle=np.linalg.lstsq(B,truth-base_abs,rcond=None)[0]
    op=base_abs+B@oracle
    cr=np.array([rmse(truth,base_abs+B@c) for c in cand])
    rho=float(spearmanr(obj,cr).statistic)
    rec={"well":w,"rows":len(ix),"base_rmse":rmse(truth,base_abs),
         "chosen_rmse":rmse(truth,pred),"oracle_rmse":rmse(truth,op),
         "objective_oracle_rank_corr":rho,"best_objective":float(vals[0]),
         "posterior_dispersion":float(disp),"hedge":float(hedge),
         **{f"coef_{j}":float(chosen[j]) for j in range(4)}}
    rows.append(rec);all_y.append(truth);all_b.append(base_abs)
    all_p.append(pred);all_or.append(op)
    if wi%20==0: print(wi,rec,flush=True)

y=np.concatenate(all_y);b=np.concatenate(all_b);p=np.concatenate(all_p);op=np.concatenate(all_or)
wd=pd.DataFrame(rows)
# Deterministic five buckets approximate whole-well fold stability for the pilot.
folds=[]
for k in range(5):
    ww=set(wlist[k::5]); m=np.concatenate([np.full(len(a),w in ww)
        for w,a in zip(wlist,all_y)])
    folds.append({"fold":k,"base":rmse(y[m],b[m]),"chosen":rmse(y[m],p[m]),
                  "gain":rmse(y[m],b[m])-rmse(y[m],p[m])})
summary={"wells":len(wlist),"rows":len(y),"base":rmse(y,b),"chosen":rmse(y,p),
 "gain":rmse(y,b)-rmse(y,p),"cubic_oracle":rmse(y,op),
 "median_rank_corr":float(wd.objective_oracle_rank_corr.median()),
 "mean_rank_corr":float(wd.objective_oracle_rank_corr.mean()),
 "well_win_rate":float((wd.chosen_rmse<wd.base_rmse).mean()),"folds":folds,
 "protocol":"per-well only; visible-prefix affine calibration + hidden GR; no cross-well mapping"}
wd.to_csv(OUT/"well_metrics.csv",index=False)
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
