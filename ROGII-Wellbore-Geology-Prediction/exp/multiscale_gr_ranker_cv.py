"""Grouped complete-well multiscale GR candidate ranker with DP decoding."""
from pathlib import Path
import argparse, glob, json, joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/train"
OUT=ROOT/"exp/results/multiscale_gr_ranker"; OFF=np.arange(-48,48.1,2,dtype=np.float32)
STRIDE=5

def robust(x):
    x=np.asarray(x,float); med=np.nanmedian(x)
    sc=np.nanpercentile(x,75)-np.nanpercentile(x,25)
    return ((x-med)/max(float(sc),5.)).astype(np.float32)

def smooth(x,n):
    return pd.Series(x).rolling(n,center=True,min_periods=1).mean().to_numpy()

def load_well(path,v4_delta,global_idx):
    wid=Path(path).name.split("__")[0]; h=pd.read_csv(path)
    t=pd.read_csv(DATA/f"{wid}__typewell.csv").sort_values("TVT")
    ps=int(h.TVT_input.notna().sum())
    if ps<8 or ps>=len(h): return None
    anchor=float(h.TVT_input.iloc[ps-1])
    rows=np.unique(np.r_[np.arange(ps-1,len(h),STRIDE),len(h)-1])
    hg=h.GR.interpolate(limit_direction="both").fillna(h.GR.median()).to_numpy(float)
    tg=t.GR.interpolate(limit_direction="both").fillna(t.GR.median()).to_numpy(float)
    tt=t.TVT.to_numpy(float)
    # Per-well calibration uses visible inputs only.
    ref=np.interp(h.TVT_input.iloc[:ps].to_numpy(float),tt,tg)
    A=np.c_[ref,np.ones(ps)]
    slope,intercept=np.linalg.lstsq(A,hg[:ps],rcond=None)[0]
    slope=float(np.clip(slope,.25,2.5)); intercept=float(np.clip(intercept,-100,100))
    hs=[robust(smooth(hg,n)) for n in (5,21,61)]
    ts=[robust(smooth(tg,n)) for n in (5,21,61)]
    base=np.zeros(len(h),np.float32); base[ps:]=v4_delta
    b=base[rows]; cand=anchor+b[:,None]+OFF[None,:]
    shp=cand.shape; feats=[]
    costs=[]
    for hx,tx in zip(hs,ts):
        tv=np.interp(cand,tt,tx)
        hv=np.broadcast_to(hx[rows,None],shp)
        d=np.abs(hv-tv)
        feats.extend([hv,tv,hv-tv,d]); costs.append(d)
    # Texture/gradient compatibility and local shape of the cost curve.
    hd=np.gradient(hs[1])[rows,None]
    td=np.interp(cand,tt,np.gradient(ts[1]))
    feats.extend([np.broadcast_to(hd,shp)-td,
                  np.abs(np.broadcast_to(hd,shp)-td)])
    cost=sum(costs)/3
    cgrad=np.gradient(cost,axis=1); ccurv=np.gradient(cgrad,axis=1)
    feats.extend([cgrad,ccurv])
    z=h.Z.to_numpy(float); md=h.MD.to_numpy(float)
    prog=(md[rows]-md[ps-1])/max(md[-1]-md[ps-1],1.)
    feats.extend([
      np.broadcast_to((OFF/48)[None,:],shp),
      np.broadcast_to(prog[:,None],shp),
      np.broadcast_to(((z[rows]-z[ps-1])/50)[:,None],shp),
      np.broadcast_to((b/30)[:,None],shp)])
    X=np.stack(feats,-1).astype(np.float32)
    y=h.TVT.to_numpy(float)[rows]-anchor-b
    cls=np.argmin(np.abs(OFF[None,:]-y[:,None]),axis=1)
    return dict(well=wid,X=X,y=y.astype(np.float32),cls=cls,rows=rows,
      ps=ps,n=len(h),base_suffix=np.asarray(v4_delta,np.float32),
      global_idx=np.asarray(global_idx,np.int64))

def train_ranker(items,max_stations,seed):
    rng=np.random.default_rng(seed); xx=[]; yy=[]; groups=[]
    for w in items:
        eligible=np.flatnonzero(w["rows"]>=w["ps"])
        if len(eligible)>max_stations:
            eligible=np.sort(rng.choice(eligible,max_stations,replace=False))
        x=w["X"][eligible].reshape(-1,w["X"].shape[-1])
        dist=np.abs(OFF[None,:]-w["y"][eligible,None])
        rel=np.clip(12-np.rint(dist/2),0,12).astype(np.int16).reshape(-1)
        xx.append(x); yy.append(rel); groups.extend([len(OFF)]*len(eligible))
    X=np.concatenate(xx); y=np.concatenate(yy)
    model=LGBMRanker(objective="lambdarank",metric="ndcg",n_estimators=220,
      learning_rate=.045,num_leaves=31,min_child_samples=200,
      subsample=.8,colsample_bytree=.85,reg_lambda=2,n_jobs=-1,verbosity=-1,
      random_state=seed)
    model.fit(X,y,group=np.asarray(groups),eval_at=[1,3])
    return model

def decode(scores):
    # Convert ranking scores into station posterior confidence. Ambiguous rows
    # receive stronger continuity regularization.
    s=scores-scores.max(1,keepdims=True)
    p=np.exp(np.clip(s,-30,0)); p/=p.sum(1,keepdims=True)
    entropy=-(p*np.log(p+1e-12)).sum(1)/np.log(p.shape[1])
    margin=np.partition(scores,-2,axis=1)[:,-1]-np.partition(scores,-2,axis=1)[:,-2]
    unary=-np.log(p+1e-12)
    n,k=scores.shape; zero=np.argmin(abs(OFF))
    prev=unary[0]+.08*(np.arange(k)-zero)**2; back=np.zeros((n,k),np.int16)
    jj=np.arange(k)
    for i in range(1,n):
        weight=.025+.15*entropy[i]/(1+max(float(margin[i]),0))
        cur=np.full(k,np.inf)
        for d in range(-4,5):
            src=jj-d; ok=(src>=0)&(src<k)
            val=prev[src[ok]]+weight*d*d
            imp=val<cur[ok]; dst=jj[ok][imp]
            cur[dst]=val[imp]; back[i,dst]=src[ok][imp]
        prev=cur+unary[i]
    path=np.empty(n,np.int16); path[-1]=np.argmin(prev)
    for i in range(n-1,0,-1): path[i-1]=back[i,path[i]]
    return OFF[path],entropy.astype(np.float32),margin.astype(np.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--max-wells",type=int); ap.add_argument("--max-stations",type=int,default=80)
    a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    feat=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    vo=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"])
    v4={w:(vo[q.index].astype(np.float32),q.index.to_numpy())
        for w,q in feat.groupby("well",sort=False)}
    paths=sorted(glob.glob(str(DATA/"*__horizontal_well.csv")))
    if a.max_wells: paths=paths[:a.max_wells]
    wells=[x for x in (load_well(p,*v4[Path(p).name.split("__")[0]]) for p in paths) if x]
    splits=list(GroupKFold(min(5,len(wells))).split(wells,groups=[w["well"] for w in wells]))
    oof=np.full(len(feat),np.nan,np.float32); foldid=np.full(len(feat),-1,np.int8)
    fold_rows=[]; well_rows=[]
    for f,(ti,vi) in enumerate(splits[:a.folds]):
        model=train_ranker([wells[i] for i in ti],a.max_stations,2026+f)
        se=base_se=nrow=0
        for j in vi:
            w=wells[j]; scores=model.predict(w["X"].reshape(-1,w["X"].shape[-1])).reshape(w["X"].shape[:2])
            pred,ent,margin=decode(scores); m=w["rows"]>=w["ps"]
            err=pred[m]-w["y"][m]; se+=np.square(err).sum()
            base_se+=np.square(w["y"][m]).sum(); nrow+=m.sum()
            suffix=np.arange(w["ps"],w["n"])
            correction=np.interp(suffix,w["rows"],pred).astype(np.float32)
            delta=w["base_suffix"]+correction
            oof[w["global_idx"]]=delta; foldid[w["global_idx"]]=f
            well_rows.append(dict(well=w["well"],fold=f,rows=int(m.sum()),
              rmse=float(np.sqrt(np.mean(err**2))),entropy=float(ent[m].mean()),
              margin=float(margin[m].mean())))
        fold_rows.append(dict(fold=f,rows=int(nrow),rmse=float(np.sqrt(se/nrow)),
                              base_rmse=float(np.sqrt(base_se/nrow))))
        joblib.dump(model,OUT/f"fold{f}.joblib")
        print(json.dumps(fold_rows[-1]),flush=True)
    used=np.concatenate([w["global_idx"] for w in wells])
    if a.folds==len(splits) and (not np.isfinite(oof[used]).all() or (foldid[used]<0).any()):
        raise RuntimeError("incomplete OOF")
    total=float(np.sqrt(np.average([x["rmse"]**2 for x in fold_rows],
                                   weights=[x["rows"] for x in fold_rows])))
    baseline=float(np.sqrt(np.average([x["base_rmse"]**2 for x in fold_rows],
                                      weights=[x["rows"] for x in fold_rows])))
    summary=dict(pooled_rmse=total,base_rmse=baseline,gain=baseline-total,
                 folds=fold_rows,wells=len(wells),offset_step=2,
                 max_train_stations_per_well=a.max_stations,
                 grouped_complete_well_cv=True,visible_only_calibration=True)
    np.savez_compressed(OUT/"oof_delta.npz",prediction=oof,fold=foldid,
      target=feat.target.to_numpy(np.float32),ids=feat.id.to_numpy(str))
    pd.DataFrame(well_rows).to_csv(OUT/"well_metrics.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
