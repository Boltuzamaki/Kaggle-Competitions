"""Strict complete-well residual-shape prior and retrieval diagnostic.

The hidden suffix is represented on a normalized arclength grid.  The library
contains residual shapes from outer-training wells only.  Query embeddings use
only legal complete-suffix features/predictions (never target/formation).
"""
from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results" / os.environ.get("CURVE_PRIOR_RUN", "generative_curve_prior_retrieval")
OUT.mkdir(parents=True, exist_ok=True)
L = 128
SEQ = ["d_z", "dz_dmd", "gr", "gr_m21", "gr_s21", "pf_d", "pf3_d",
       "beam_d", "ncc15_d", "ncc15_s", "sp_mean_d", "sp_std", "sp_dmin"]

def rs(v):
    v = np.asarray(v, float)
    return np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(v)), v)

def rmse(y, p, w=None):
    e = (np.asarray(y)-np.asarray(p))**2
    if w is not None and e.ndim == 2:
        e = e.mean(1)
    return float(np.sqrt(np.average(e, weights=w)))

f = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
z = np.load(ROOT/"exp/results/heel_calibrated_gr_datum/full_meta/oof.npz")
ix = z["global_indices"].astype(int)
f = f.iloc[ix].reset_index(drop=True)
s = np.load(ROOT/"exp/results/pf_student10_full_oof/meta_oof.npz", allow_pickle=True)
assert np.allclose(s["y"], f.target)
base = .1*s["accepted"] + .9*s["replacement"]
f["_base"] = base

curves=[]; query=[]; lens=[]; wells=[]
for w,g in f.groupby("well", sort=True):
    r = rs(g.target.to_numpy()-g._base.to_numpy())
    # Shape about its legal geometric anchor.  Keeping the datum separately
    # diagnoses whether failures are datum or curvature.
    curves.append(r)
    q=[]
    for c in SEQ + ["_base"]:
        x=rs(g[c])
        # low-frequency complete-well signature, insensitive to row count
        q.extend(x.reshape(16,8).mean(1))
        q.extend([x.mean(),x.std(),x[-1]-x[0],np.quantile(x,.1),np.quantile(x,.9)])
    # Legal structural-regime fingerprint from the organizer-visible prefix.
    # The first pilot omitted the actual TVT_input history and only saw suffix
    # acquisition/trajectory proxies.
    hh=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv',usecols=['MD','X','Y','Z','TVT_input'])
    hh=hh[hh.TVT_input.notna()]
    for c in ['TVT_input','Z']:
        x=rs(hh[c]); x=x-x[-1]
        q.extend(x.reshape(16,8).mean(1));q.extend([x.mean(),x.std(),x[-1]-x[0]])
    slope=np.gradient(hh.TVT_input.to_numpy())/(np.gradient(hh.MD.to_numpy())+1e-6)
    x=rs(slope);q.extend(x.reshape(16,8).mean(1));q.extend([x.mean(),x.std(),x[-1],np.mean(x[-16:])])
    # Directly observed causal prefix model error.  The geometry expert at a
    # past origin predicts dTVT=dZ; its realized error is observable entirely
    # inside TVT_input. Differences make every point a one-step causal audit.
    tv=hh.TVT_input.to_numpy(float); zz=hh.Z.to_numpy(float); md=hh.MD.to_numpy(float)
    er=np.r_[0,np.diff(tv)-np.diff(zz)]
    cum=np.cumsum(er); cum-=cum[-1]
    for raw in [er,cum]:
        x=rs(raw);q.extend(x.reshape(16,8).mean(1));q.extend([x.mean(),x.std(),x[-1]-x[0],np.mean(x[-16:]),np.std(x[-16:])])
    # Multihorizon strictly-past trend experts. At each historical cut, fit on
    # the preceding window and score a subsequent known block.
    for win in [50,150,400]:
        vals=[]
        for frac0 in [.55,.7,.82,.92]:
            cut=max(win+2,int(len(hh)*frac0)); end=min(len(hh),cut+max(10,len(hh)//20))
            if end<=cut: vals.extend([0,0,0]);continue
            xx=md[max(0,cut-win):cut]-md[cut-1]; yy=tv[max(0,cut-win):cut]-tv[cut-1]
            co=np.polyfit(xx,yy,1); pp=np.polyval(co,md[cut:end]-md[cut-1]); rr=(tv[cut:end]-tv[cut-1])-pp
            vals.extend([rr.mean(),rr[-1],np.sqrt(np.mean(rr*rr))])
        q.extend(vals)
    # Absolute field position/orientation and the complete typewell signature
    # permit spatially local analogue experts.
    q.extend([hh.X.iloc[-1],hh.Y.iloc[-1],hh.Z.iloc[-1],hh.X.iloc[0],hh.Y.iloc[0],
              hh.X.iloc[-1]-hh.X.iloc[0],hh.Y.iloc[-1]-hh.Y.iloc[0]])
    tw=pd.read_csv(ROOT/'data/train'/f'{w}__typewell.csv',usecols=['TVT','GR']).dropna()
    tg=rs(tw.GR);tt=rs(tw.TVT);q.extend(tg.reshape(16,8).mean(1));q.extend([tg.mean(),tg.std(),tt[0],tt[-1]])
    query.append(q); lens.append(len(g)); wells.append(w)
C=np.asarray(curves); Q=np.nan_to_num(np.asarray(query)); lens=np.asarray(lens)
wells=np.asarray(wells); folds=list(GroupKFold(5).split(Q,groups=wells))

# Equal-grid oracle first, plus row-count weighted score approximation.
pred_knn=np.zeros_like(C); pred_ridge=np.zeros_like(C); pred_et=np.zeros_like(C)
oracle=np.zeros_like(C); ranks=[]; fold_rows=[]
for k,(tr,va) in enumerate(folds):
    sc=StandardScaler().fit(Q[tr]); a=sc.transform(Q[tr]); b=sc.transform(Q[va])
    # PCA suppresses noisy GR aliases; dimensions selected without targets.
    pc=PCA(n_components=min(48,len(tr)-1), whiten=True, random_state=43).fit(a)
    a=pc.transform(a); b=pc.transform(b)
    dist=((b[:,None,:]-a[None,:,:])**2).mean(2)
    nn=np.argsort(dist,axis=1)
    for j,v in enumerate(va):
        # distance-softened 8-neighbour generative posterior mean
        jj=nn[j,:8]; dd=np.sqrt(dist[j,jj]); ww=np.exp(-dd/(np.median(dd)+1e-6)); ww/=ww.sum()
        pred_knn[v]=ww@C[tr[jj]]
        # Oracle retrieval has no fitted affine transform: it measures whether
        # a real training-well suffix shape can represent this held-out well.
        er=((C[tr]-C[v])**2).mean(1); oi=np.argmin(er); oracle[v]=C[tr[oi]]
        ranks.append(int(np.where(nn[j]==oi)[0][0])+1)
    # Conditional latent generator: PCA residual manifold + two deliberately
    # different low-variance mappings. This is a VAE-equivalent deterministic
    # pilot appropriate for only ~600 outer-training examples.
    cp=PCA(n_components=24, random_state=44).fit(C[tr])
    latent=cp.transform(C[tr])
    rr=Ridge(alpha=100).fit(a,latent)
    pred_ridge[va]=cp.inverse_transform(rr.predict(b))
    et=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=12,max_features=.6,
                           n_jobs=-1,random_state=45).fit(a,latent)
    pred_et[va]=cp.inverse_transform(et.predict(b))
    fold_rows.append({"fold":k,"wells":len(va),"base":rmse(C[va]*0,C[va],lens[va]),
      "oracle_library":rmse(C[va],oracle[va],lens[va]),
      "knn":rmse(C[va],pred_knn[va],lens[va]),"ridge":rmse(C[va],pred_ridge[va],lens[va]),
      "et":rmse(C[va],pred_et[va],lens[va])})
    print(fold_rows[-1],flush=True)

# Convert generated grid corrections back to exact rows and audit blends.
row_y=[]; row_b=[]; row_preds={n:[] for n in ["oracle","knn","ridge","et"]}
P={"oracle":oracle,"knn":pred_knn,"ridge":pred_ridge,"et":pred_et}
for i,(w,g) in enumerate(f.groupby("well",sort=True)):
    row_y.append(g.target.to_numpy()); row_b.append(g._base.to_numpy())
    for n in P: row_preds[n].append(np.interp(np.linspace(0,1,len(g)),np.linspace(0,1,L),P[n][i]))
row_y=np.concatenate(row_y); row_b=np.concatenate(row_b)
grid=[]
for n in P:
    p=np.concatenate(row_preds[n])
    for blend in [0,.05,.1,.2,.35,.5,.75,1]:
        grid.append({"model":n,"blend":blend,"rmse":rmse(row_y,row_b+blend*p)})
grid=pd.DataFrame(grid).sort_values("rmse")
summary={"wells":len(wells),"rows":len(row_y),"baseline":rmse(row_y,row_b),
         "best":grid.iloc[0].to_dict(),"median_oracle_retrieval_rank":float(np.median(ranks)),
         "folds":fold_rows}
print(json.dumps(summary,indent=2),flush=True)
grid.to_csv(OUT/"blend_grid.csv",index=False)
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
np.savez_compressed(OUT/"well_curves.npz",wells=wells,true=C,oracle=oracle,knn=pred_knn,
                    ridge=pred_ridge,et=pred_et,length=lens,query=Q)
