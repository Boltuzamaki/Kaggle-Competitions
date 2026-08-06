"""Strict cross-well functional residual transfer around the locked Student-t stack.

No formation columns are read. Validation-well targets are used only for scoring.
"""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/spatial_residual_graph"
OUT.mkdir(parents=True, exist_ok=True)

z = np.load(ROOT / "exp/results/pf_student10_full_oof/meta_oof.npz", allow_pickle=True)
y, base, groups = z["y"], z["replacement"], z["groups"].astype(str)
assert len(y) == len(base) == len(groups)
ERR=y-base

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

records=[]; curves=[]; lengths=[]; row_slices={}
grid=np.linspace(0,1,48)
print("loading 773 wells", flush=True)
for wid in np.unique(groups):
    ix=np.flatnonzero(groups==wid)
    d=pd.read_csv(ROOT/f"data/train/{wid}__horizontal_well.csv",
                  usecols=["MD","X","Y","Z","TVT_input","GR"])
    hid=np.flatnonzero(d.TVT_input.isna().to_numpy())
    if len(hid)!=len(ix):
        raise RuntimeError(f"alignment {wid}: csv={len(hid)} meta={len(ix)}")
    q=d.iloc[hid]; p=(q.MD.to_numpy()-q.MD.iloc[0])/max(q.MD.iloc[-1]-q.MD.iloc[0],1)
    r=y[ix]-base[ix]
    c=np.interp(grid,p,r); c=gaussian_filter1d(c,1.25)
    # Legal well-level geometry: heel/start, toe, tangent and prefix summaries.
    dx=float(q.X.iloc[-1]-q.X.iloc[0]); dy=float(q.Y.iloc[-1]-q.Y.iloc[0]); norm=max(np.hypot(dx,dy),1)
    vis=d[d.TVT_input.notna()]
    records.append(dict(wid=wid,x=float(q.X.iloc[0]),y=float(q.Y.iloc[0]),z=float(q.Z.iloc[0]),
       ux=dx/norm,uy=dy/norm,length=norm, dz=float(q.Z.iloc[-1]-q.Z.iloc[0]),
       gr=float(q.GR.mean()),grsd=float(q.GR.std()),prefix_tvt=float(vis.TVT_input.iloc[-1]),
       prefix_slope=float((vis.TVT_input.iloc[-1]-vis.TVT_input.iloc[max(0,len(vis)-128)])/max(1,min(127,len(vis)-1)))))
    curves.append(c); lengths.append(len(ix)); row_slices[wid]=ix
W=pd.DataFrame(records); C=np.asarray(curves); lengths=np.asarray(lengths)
print("loaded", len(W), flush=True)

# Spatial blocks: contiguous x/y tiles; sorted round-robin assignment balances fold mass.
xy=W[["x","y"]].to_numpy(); xy=(xy-xy.mean(0))/(xy.std(0)+1e-9)
ang=np.arctan2(W.uy,W.ux).to_numpy()
block_key=np.floor(xy[:,0]*1.4).astype(int)*100 + np.floor(xy[:,1]*1.4).astype(int)
blocks=np.unique(block_key)
block_mass={b:int(lengths[block_key==b].sum()) for b in blocks}
loads=np.zeros(5,int); bfold={}
for b in sorted(blocks,key=lambda q:block_mass[q],reverse=True):
    f=int(np.argmin(loads)); bfold[b]=f; loads[f]+=block_mass[b]
spfold=np.array([bfold[b] for b in block_key])
randfold=np.empty(len(W),int)
for f,(_,va) in enumerate(KFold(5,shuffle=True,random_state=20260803).split(W)): randfold[va]=f

def graph_predict(tr,va,ncomp,hxy,hang,ridge):
    # Fit basis only on outer training wells: avoids validation residual leakage.
    pca=PCA(n_components=min(ncomp,len(tr)-1), random_state=1).fit(C[tr])
    A=pca.transform(C[tr])
    pred=[]; support=[]
    for i in va:
        delta=xy[tr]-xy[i]
        # Rotate offsets into the query well's along/cross-track frame.
        along=delta[:,0]*W.ux.iloc[i]+delta[:,1]*W.uy.iloc[i]
        cross=-delta[:,0]*W.uy.iloc[i]+delta[:,1]*W.ux.iloc[i]
        da=np.arccos(np.clip(np.cos(ang[tr]-ang[i]),-1,1))
        w=np.exp(-.5*((along/hxy)**2+(cross/(hxy*.55))**2+(da/hang)**2))
        # Global shrinkage makes unsupported graph nodes safely approach mean residual.
        coef=(w[:,None]*A).sum(0)/(w.sum()+ridge)
        pred.append(pca.inverse_transform(coef[None,:])[0]); support.append(float(w.sum()))
    return np.asarray(pred),np.asarray(support)

def evaluate(folds,name):
    rows=[]; best=None
    # Compact pilot grid; expand only if the spatial-block check is positive.
    for hxy,hang,ridge in [(.6,.5,4.),(1.2,1.2,4.),(2.4,3.2,4.),
                            (.6,1.2,32.),(1.2,3.2,32.)]:
        print(name,hxy,hang,ridge,flush=True)
        pc=np.zeros_like(C); sup=np.zeros(len(W))
        for f in range(5):
            va=np.flatnonzero(folds==f); tr=np.flatnonzero(folds!=f)
            pc[va],sup[va]=graph_predict(tr,va,8,hxy,hang,ridge)
        corr=np.corrcoef(C.ravel(),pc.ravel())[0,1]
        # Predeclared conservative correction amplitudes; score pooled hidden rows.
        correction=np.empty_like(base)
        for i,wid in enumerate(W.wid):
            ix=row_slices[wid]; p=np.linspace(0,1,len(ix))
            correction[ix]=np.interp(p,grid,pc[i])
        # Algebraic RMSE avoids repeated 30 MB temporaries under parallel-run pressure.
        e2=float(np.dot(ERR,ERR)); ec=float(np.dot(ERR,correction)); c2=float(np.dot(correction,correction))
        scores={str(alpha):float(np.sqrt((e2-2*alpha*ec+alpha*alpha*c2)/len(y)))
                for alpha in [0,.05,.1,.2,.35,.5,1.0]}
        rec=dict(split=name,hxy=hxy,hang=hang,ridge=ridge,corr=float(corr),support=float(np.median(sup)),scores=scores)
        rows.append(rec)
        val=min(scores.values())
        if best is None or val<best[0]: best=(val,rec,pc)
    return rows,best

rr,br=evaluate(randfold,"group_random")
ss,bs=evaluate(spfold,"leave_spatial_block_out")
pd.DataFrame([{**{k:v for k,v in r.items() if k!='scores'},**{f"a_{k}":v for k,v in r['scores'].items()}} for r in rr+ss]).to_csv(OUT/"grid.csv",index=False)
def row_correction(pc):
    out=np.empty_like(base)
    for i,wid in enumerate(W.wid):
        ix=row_slices[wid]; out[ix]=np.interp(np.linspace(0,1,len(ix)),grid,pc[i])
    return out
locked=row_correction(bs[2]); locked_alpha=.5
leg_scores={}
for leg in ["accepted","add","replacement"]:
    b=z[leg]
    leg_scores[leg]={"base":rmse(y,b),"corrected":rmse(y,b+locked_alpha*locked)}
fold_rows=[]
for f in range(5):
    wg=set(W.wid[spfold==f]); ix=np.array([g in wg for g in groups])
    fold_rows.append(dict(fold=f,rows=int(ix.sum()),base=rmse(y[ix],base[ix]),corrected=rmse(y[ix],base[ix]+locked_alpha*locked[ix])))
pd.DataFrame(fold_rows).to_csv(OUT/"locked_spatial_fold_metrics.csv",index=False)
np.savez_compressed(OUT/"locked_correction.npz",groups=groups,correction=locked)
summary={"baseline":rmse(y,base),"n_wells":len(W),"n_rows":len(y),"spatial_fold_row_loads":loads.tolist(),
 "random_best":br[1],"spatial_best":bs[1],
 "locked_alpha":locked_alpha,"leg_scores":leg_scores,"locked_spatial_folds":fold_rows,
 "decision":"reject unless spatial-block correction improves baseline and is directionally consistent"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
W.assign(random_fold=randfold,spatial_fold=spfold).to_csv(OUT/"wells.csv",index=False)
print(json.dumps(summary,indent=2))
