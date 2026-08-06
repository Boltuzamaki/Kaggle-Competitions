"""Leakage-safe reconstruction of formation-marker *shape* for whole-well PCA CV.

The six supplied train marker columns are forbidden validation inputs because
they are absent from the test schema.  This experiment reconstructs their
common normalized shape using only trajectory and the visible TVT prefix.
Every target-well fit stops at its organizer TVT_input boundary.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "data/train").glob("*__horizontal_well.csv"))
FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
N_IN, N_OUT, SEED = 48, 96, 20260730

def interp(v, q):
    v=np.asarray(v,float); x=np.linspace(0,1,len(v)); ok=np.isfinite(v)
    return np.interp(q,x[ok],v[ok]) if ok.sum()>1 else np.full(len(q),np.nanmedian(v) if ok.any() else 0.)

def proxy_curve(d, cut, kind):
    """Predict relative structural U shape, with value exactly zero at cut."""
    u=(d.TVT_input+d.Z).to_numpy(float)
    md=d.MD.to_numpy(float); x=d.X.to_numpy(float); y=d.Y.to_numpy(float)
    ii=np.arange(cut+1); # only visible prefix
    # Weight recent prefix more because it is spatially closest to suffix.
    if kind.startswith("md"):
        deg=int(kind[-1]); t=(md-md[cut])/1000
        use=ii[max(0,cut-1200):]
        co=np.polyfit(t[use],u[use]-u[cut],deg,w=np.linspace(.35,1,len(use)))
        p=np.polyval(co,t)
    elif kind=="xy_ridge2":
        use=ii[max(0,cut-1600):]
        xy=np.c_[(x-x[cut])/1000,(y-y[cut])/1000]
        pf=PolynomialFeatures(2,include_bias=True)
        A=pf.fit_transform(xy)
        m=Ridge(alpha=2.).fit(A[use],u[use]-u[cut])
        p=m.predict(A)
    elif kind=="tangent":
        # Robust local structural slope, extrapolated along displacement.
        use=ii[max(0,cut-900):]
        A=np.c_[(x-x[cut])/1000,(y-y[cut])/1000]
        m=HuberRegressor(epsilon=1.5,alpha=2.).fit(A[use],u[use]-u[cut])
        p=m.predict(A)
    else: raise ValueError(kind)
    return p-p[cut]

def encode(path):
    d=pd.read_csv(path); nvis=int(d.TVT_input.notna().sum())
    if nvis<20 or nvis>=len(d)-3:return None
    cut=nvis-1; md=d.MD.to_numpy(float); z=d.Z.to_numpy(float)
    x=d.X.to_numpy(float); y=d.Y.to_numpy(float); tvt=d.TVT.to_numpy(float)
    u=tvt+z; q=np.linspace(0,1,N_IN); qp=np.linspace(0,1,20)
    base=[(md-md[cut])/1000,(z-z[cut])/50,(x-x[cut])/3000,(y-y[cut])/3000,np.gradient(z)*20]
    gr=d.GR.interpolate(limit_direction="both").fillna(d.GR.median()).to_numpy(float)
    gr=(gr-np.nanmedian(gr))/(np.nanstd(gr)+1e-6)
    tail=np.r_[interp(u[:nvis]-u[cut],qp),len(d)/6000,nvis/len(d),
      (md[-1]-md[cut])/4000,np.sin(np.arctan2(y[-1]-y[cut],x[-1]-x[cut])),
      np.cos(np.arctan2(y[-1]-y[cut],x[-1]-x[cut]))]
    common=base+[gr,gaussian_filter1d(gr,10),gaussian_filter1d(gr,40)]
    feats={}
    true=[(d[c].to_numpy(float)-d[c].iloc[cut])/50 for c in FORMS]
    feats["true_marker_leak"]=np.r_[*[interp(a,q) for a in base+true+common[5:]],tail]
    feats["no_marker"]=np.r_[*[interp(a,q) for a in common],tail]
    proxies={}
    for kind in ["md1","md2","md3","xy_ridge2","tangent"]:
        p=proxy_curve(d,cut,kind)/50
        proxies[kind]=p
        feats[kind]=np.r_[*[interp(a,q) for a in base+[p]*6+common[5:]],tail]
    target=interp(u[cut:]-u[cut],np.linspace(0,1,N_OUT))
    return dict(d=d,cut=cut,u0=u[cut],target=target,feats=feats,proxies=proxies,
                true_shape=d.ANCC.to_numpy(float)-d.ANCC.iloc[cut])

items=[z for z in (encode(p) for p in FILES) if z]
Y=np.stack([z["target"] for z in items]); kinds=list(items[0]["feats"])
X={k:np.stack([z["feats"][k] for z in items]) for k in kinds}
P={k:np.zeros_like(Y) for k in kinds}; kf=KFold(5,shuffle=True,random_state=SEED)
# Fold-trained spatial surface proxies. Unlike target-well polynomial
# extrapolation, these borrow formation geometry only from outer-training wells.
P.update({f"spatial_poly{d}":np.zeros_like(Y) for d in (1,2,3)})
spatial_proxy_err={f"spatial_poly{d}":[] for d in (1,2,3)}
for fold,(tr,va) in enumerate(kf.split(Y)):
    pca=PCA(n_components=16,random_state=SEED).fit(Y[tr]); sc=pca.transform(Y[tr])
    for k in kinds:
        ss=StandardScaler().fit(X[k][tr])
        m=Ridge(alpha=120.).fit(ss.transform(X[k][tr]),sc)
        P[k][va]=pca.inverse_transform(m.predict(ss.transform(X[k][va])))
    # Sample complete marker surfaces only from outer-training wells.
    sx=[]; sy=[]
    for i in tr:
        d=items[i]["d"]; jj=np.linspace(0,len(d)-1,min(32,len(d))).astype(int)
        sx.append(d[["X","Y"]].to_numpy(float)[jj]); sy.append(d.ANCC.to_numpy(float)[jj])
    sx=np.vstack(sx); sy=np.concatenate(sy)
    ok=np.isfinite(sy)&np.isfinite(sx).all(1); sx=sx[ok]; sy=sy[ok]
    xy_scaler=StandardScaler().fit(sx); sxs=xy_scaler.transform(sx)
    for deg in (1,2,3):
        pf=PolynomialFeatures(deg,include_bias=True); A=pf.fit_transform(sxs)
        surf=Ridge(alpha=1.0).fit(A,sy)
        xva=[]
        for i in va:
            d=items[i]["d"]; c=items[i]["cut"]
            raw=surf.predict(pf.transform(xy_scaler.transform(d[["X","Y"]].to_numpy(float))))
            rel=raw-raw[c]
            true=items[i]["true_shape"]
            ok=np.isfinite(true[c+1:])
            spatial_proxy_err[f"spatial_poly{deg}"].append((rel[c+1:][ok]-true[c+1:][ok]))
            no=X["no_marker"][i]
            geom=no[:5*N_IN]; grtail=no[5*N_IN:]
            xva.append(np.r_[geom,*[interp(rel/50,np.linspace(0,1,N_IN)) for _ in FORMS],grtail])
        # Train-side spatial features from the same outer-training surface are
        # legitimate fitted transformations, analogous to model predictions.
        xtr=[]
        for i in tr:
            d=items[i]["d"]; c=items[i]["cut"]
            raw=surf.predict(pf.transform(xy_scaler.transform(d[["X","Y"]].to_numpy(float))))
            rel=raw-raw[c]; no=X["no_marker"][i]
            xtr.append(np.r_[no[:5*N_IN],*[interp(rel/50,np.linspace(0,1,N_IN)) for _ in FORMS],no[5*N_IN:]])
        ss=StandardScaler().fit(np.stack(xtr))
        m=Ridge(alpha=120.).fit(ss.transform(np.stack(xtr)),sc)
        P[f"spatial_poly{deg}"][va]=pca.inverse_transform(m.predict(ss.transform(np.stack(xva))))
    print("fold",fold,flush=True)

def score(p):
    es=[]
    for i,it in enumerate(items):
        d=it["d"]; c=it["cut"]; n=len(d)-c-1
        rel=np.interp(np.linspace(0,1,n+1)[1:],np.linspace(0,1,N_OUT),p[i])
        pred=it["u0"]+rel-d.Z.to_numpy(float)[c+1:]
        es.append(d.TVT.to_numpy(float)[c+1:]-pred)
    return float(np.sqrt(np.mean(np.concatenate(es)**2)))

scores={k:score(v) for k,v in P.items()}
proxy_err={}
for k in ["md1","md2","md3","xy_ridge2","tangent"]:
    ee=[]
    for it in items:
        c=it["cut"]; ee.append(it["proxies"][k][c+1:]-it["true_shape"][c+1:])
    e=np.concatenate(ee); e=e[np.isfinite(e)]
    proxy_err[k]={"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(abs(e)))}
for k,ee in spatial_proxy_err.items():
    e=np.concatenate(ee); e=e[np.isfinite(e)]
    proxy_err[k]={"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(abs(e)))}
out={"n_wells":len(items),"scores":scores,"marker_shape_proxy_errors_ft":proxy_err,
     "protocol":"5-fold whole-well CV; proxy fit uses target well visible prefix only; suffix marker labels used for audit only"}
print(json.dumps(out,indent=2))
(ROOT/"exp/results").mkdir(exist_ok=True)
(ROOT/"exp/results/marker_proxy_pca_cv.json").write_text(json.dumps(out,indent=2))
