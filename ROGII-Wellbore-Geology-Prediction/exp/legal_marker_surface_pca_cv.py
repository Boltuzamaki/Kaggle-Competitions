"""Predict EGFDU marker-surface shape from strictly test-available inputs.

Target marker columns are used only to construct the training label.  Inputs:
full horizontal MD/XYZ/GR, visible TVT prefix, and paired typewell TVT/GR/
Geology.  A fold-specific PCA compresses the complete marker curve; Ridge and
ExtraTrees predict its coefficients. TVT follows from U_pred = U_PS + dMarker.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT/"data/train").glob("*__horizontal_well.csv"))
OUT = ROOT/"exp/results/legal_marker_surface_pca"
OUT.mkdir(parents=True, exist_ok=True)
NQ, NY, SEED = 64, 128, 73126

def rs(v, n=NQ):
    v=np.asarray(v,float); x=np.linspace(0,1,len(v)); ok=np.isfinite(v)
    if ok.sum()<2: return np.full(n,np.nanmedian(v) if ok.any() else 0.)
    return np.interp(np.linspace(0,1,n),x[ok],v[ok])

def encode(hp):
    d=pd.read_csv(hp).sort_values("MD").reset_index(drop=True)
    nvis=int(d.TVT_input.notna().sum())
    if nvis<20 or nvis>=len(d)-4 or "EGFDU" not in d:return None
    c=nvis-1; twp=next(hp.parent.glob(hp.name.replace(
        "__horizontal_well.csv","__typewell*.csv")))
    tw=pd.read_csv(twp).sort_values("TVT")
    md=d.MD.to_numpy(float); x=d.X.to_numpy(float); yy=d.Y.to_numpy(float)
    z=d.Z.to_numpy(float); gr=d.GR.interpolate(limit_direction="both").fillna(
        d.GR.median()).to_numpy(float)
    # Complete horizontal sequences, centered/scaled without train statistics.
    seq=[(md-md[c])/4000,(x-x[c])/4000,(yy-yy[c])/4000,
         (z-z[c])/200, np.gradient(z)/2,
         (gr-np.median(gr))/(np.std(gr)+1e-6)]
    seq += [gaussian_filter1d(seq[-1],s) for s in (4,12,32)]
    # Visible-prefix U trajectory padded by resampling prefix itself.
    u=(d.TVT_input+d.Z).to_numpy(float)
    up=(u[:nvis]-u[c])/50
    prefix=[rs(up),rs(np.gradient(up))]
    # Paired typewell is fully legal at inference.
    tg=tw.GR.interpolate(limit_direction="both").fillna(tw.GR.median()).to_numpy(float)
    tg=(tg-np.median(tg))/(np.std(tg)+1e-6)
    ttv=tw.TVT.to_numpy(float)
    twseq=[rs(tg),rs(gaussian_filter1d(tg,4)),rs(gaussian_filter1d(tg,16))]
    geo=tw.Geology.notna().to_numpy(float) if "Geology" in tw else np.zeros(len(tw))
    twseq += [rs(geo)]
    stats=np.array([len(d)/6000,nvis/len(d),(md[-1]-md[c])/4000,
        np.ptp(x)/5000,np.ptp(yy)/5000,np.ptp(z)/300,
        (np.mean(x)-2_950_000)/50_000,(np.mean(yy)-1_075_000)/50_000,
        (x[c]-2_950_000)/50_000,(yy[c]-1_075_000)/50_000,
        (x[-1]-2_950_000)/50_000,(yy[-1]-1_075_000)/50_000,
        np.sin(np.arctan2(yy[-1]-yy[c],x[-1]-x[c])),
        np.cos(np.arctan2(yy[-1]-yy[c],x[-1]-x[c])),
        len(tw)/1000,np.ptp(ttv)/500,np.mean(geo),np.sum(np.diff(geo)!=0)/20])
    X=np.r_[*[rs(a) for a in seq],*prefix,*twseq,stats].astype(np.float32)
    marker=d.EGFDU.to_numpy(float)
    target=rs(marker-marker[c],NY).astype(np.float32)
    return dict(X=X,Y=target,d=d,c=c,u0=float(d.TVT.iloc[c]+z[c]),
                true_marker=marker-marker[c])

items=[a for a in (encode(p) for p in FILES) if a]
X=np.stack([a["X"] for a in items]); Y=np.stack([a["Y"] for a in items])
kf=KFold(5,shuffle=True,random_state=SEED)
models=("ridge","extra","forest")
P={m:np.zeros_like(Y) for m in models}
for k,(tr,va) in enumerate(kf.split(X)):
    pca=PCA(n_components=24,random_state=SEED).fit(Y[tr])
    C=pca.transform(Y[tr])
    sc=StandardScaler().fit(X[tr])
    P["ridge"][va]=pca.inverse_transform(Ridge(alpha=120).fit(
        sc.transform(X[tr]),C).predict(sc.transform(X[va])))
    P["extra"][va]=pca.inverse_transform(ExtraTreesRegressor(
        n_estimators=700,max_features=.7,min_samples_leaf=3,
        n_jobs=-1,random_state=SEED+k).fit(X[tr],C).predict(X[va]))
    P["forest"][va]=pca.inverse_transform(RandomForestRegressor(
        n_estimators=450,max_features=.7,min_samples_leaf=4,
        n_jobs=-1,random_state=SEED+k).fit(X[tr],C).predict(X[va]))
    print("fold",k,flush=True)

def scores(pred):
    te=[]; me=[]
    for i,a in enumerate(items):
        d=a["d"]; c=a["c"]; n=len(d)-c-1
        rel=np.interp(np.linspace(0,1,n+1)[1:],np.linspace(0,1,NY),pred[i])
        true=d.TVT.to_numpy(float)[c+1:]
        tvp=a["u0"]+rel-d.Z.to_numpy(float)[c+1:]
        te.append(tvp-true)
        mt=a["true_marker"][c+1:]
        me.append(rel-mt)
    te=np.concatenate(te); me=np.concatenate(me)
    return {"tvt_rmse":float(np.sqrt(np.mean(te*te))),
            "marker_rmse":float(np.sqrt(np.nanmean(me*me)))}

res={m:scores(P[m]) for m in models}
# Prediction diversity/oracle blend grid.
grid=[]
for a in np.linspace(0,1,21):
    grid.append({"extra_weight":float(a),**scores(
        (1-a)*P["ridge"]+a*P["extra"])})
best=min(grid,key=lambda q:q["tvt_rmse"])
out={"wells":len(items),"input_features":X.shape[1],"results":res,
     "best_ridge_extra":best,
     "schema_audit":"No formation marker or TVT target column appears in X; EGFDU only constructs Y.",
     "protocol":"5-fold complete-well CV; TVT identity U_PS + predicted dEGFDU - Z"}
np.savez_compressed(OUT/"oof.npz", **P, Y=Y)
pd.DataFrame(grid).to_csv(OUT/"blend_grid.csv",index=False)
(OUT/"summary.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
