"""Complete-well structured prediction of a smooth residual warp."""
from pathlib import Path
import argparse, glob, json, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/train"
OUT=ROOT/"exp/results/fullwell_warp_knots"
OFF=np.arange(-48,48.1,2); BINS=np.linspace(0,1,25); KNOTS=np.linspace(0,1,13)

def robust(x):
    x=np.asarray(x,float); med=np.nanmedian(x)
    return (x-med)/max(float(np.nanpercentile(x,75)-np.nanpercentile(x,25)),5.)
def sm(x,n): return pd.Series(x).rolling(n,center=True,min_periods=1).mean().to_numpy()

def load(path,v4_delta,global_idx):
    wid=Path(path).name.split("__")[0]; h=pd.read_csv(path)
    t=pd.read_csv(DATA/f"{wid}__typewell.csv").sort_values("TVT")
    ps=int(h.TVT_input.notna().sum())
    if ps<8 or ps>=len(h): return None
    anchor=float(h.TVT_input.iloc[ps-1]); rows=np.arange(ps,len(h),5)
    if rows[-1]!=len(h)-1: rows=np.r_[rows,len(h)-1]
    hg=h.GR.interpolate(limit_direction="both").fillna(h.GR.median()).to_numpy(float)
    tg=t.GR.interpolate(limit_direction="both").fillna(t.GR.median()).to_numpy(float)
    tt=t.TVT.to_numpy(float)
    ref=np.interp(h.TVT_input.iloc[:ps],tt,tg)
    slope,intercept=np.linalg.lstsq(np.c_[ref,np.ones(ps)],hg[:ps],rcond=None)[0]
    slope=float(np.clip(slope,.25,2.5));intercept=float(np.clip(intercept,-100,100))
    cand=anchor+v4_delta[rows-ps,None]+OFF[None,:]
    costs=[]
    for n in (5,21,61,121):
        hn=robust(sm(hg,n))[rows,None]
        tn=robust(sm(tg,n))
        costs.append(np.abs(hn-np.interp(cand,tt,tn)))
    cost=np.mean(costs,axis=0)
    prog=(h.MD.to_numpy(float)[rows]-h.MD.iloc[ps-1])/max(h.MD.iloc[-1]-h.MD.iloc[ps-1],1)
    # Multiple posterior temperatures preserve both sharp matches and broad
    # ambiguity without exposing a raw station-by-candidate tensor.
    station=[]
    for temp in (.12,.3,.7,1.5):
        p=np.exp(np.clip(-(cost-cost.min(1,keepdims=True))/temp,-30,0))
        p/=p.sum(1,keepdims=True)
        station.extend([p@OFF,np.sqrt(p@(OFF**2)-(p@OFF)**2),
                        -(p*np.log(p+1e-12)).sum(1)])
    order=np.partition(cost,1,axis=1)
    station.extend([OFF[np.argmin(cost,axis=1)],cost.min(1),
                    order[:,1]-order[:,0],
                    np.gradient(OFF[np.argmin(cost,axis=1)])])
    station=np.stack(station,1)
    z=h.Z.to_numpy(float); base=v4_delta[rows-ps]
    traj=np.c_[prog,(z[rows]-z[ps-1])/50,base/30,
               np.gradient(base)/2,np.gradient(np.gradient(base))/2]
    station=np.c_[station,traj]
    # Fixed-size full-well representation: median and spread per progress bin.
    F=[]
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        m=(prog>=lo)&(prog<hi+(hi==1))
        if not m.any(): m=np.abs(prog-(lo+hi)/2)==np.min(np.abs(prog-(lo+hi)/2))
        F.extend(np.nanmedian(station[m],axis=0))
        F.extend(np.nanpercentile(station[m],75,axis=0)-np.nanpercentile(station[m],25,axis=0))
    # Smooth target warp coefficients at fixed progress knots.
    truth=h.TVT.to_numpy(float)[ps:]-anchor
    residual=truth-v4_delta
    full_prog=(h.MD.to_numpy(float)[ps:]-h.MD.iloc[ps-1])/max(h.MD.iloc[-1]-h.MD.iloc[ps-1],1)
    Y=np.interp(KNOTS,full_prog,residual)
    return dict(well=wid,F=np.asarray(F,np.float32),Y=Y.astype(np.float32),
      prog=full_prog.astype(np.float32),target=residual.astype(np.float32),
      base=np.asarray(v4_delta,np.float32),global_idx=np.asarray(global_idx,np.int64))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--max-wells",type=int);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    feat=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    vo=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"])
    v4={w:(vo[q.index].astype(np.float32),q.index.to_numpy()) for w,q in feat.groupby("well",sort=False)}
    paths=sorted(glob.glob(str(DATA/"*__horizontal_well.csv")))
    if a.max_wells:paths=paths[:a.max_wells]
    wells=[x for x in (load(p,*v4[Path(p).name.split("__")[0]]) for p in paths) if x]
    X=np.stack([w["F"] for w in wells]);Y=np.stack([w["Y"] for w in wells])
    splits=list(KFold(min(5,len(wells)),shuffle=True,random_state=2026).split(X))
    oof=np.full(len(feat),np.nan,np.float32);fid=np.full(len(feat),-1,np.int8);fr=[]
    for f,(tr,va) in enumerate(splits[:a.folds]):
        models=[
          ExtraTreesRegressor(n_estimators=400,min_samples_leaf=5,max_features=.7,n_jobs=-1,random_state=10+f),
          RandomForestRegressor(n_estimators=300,min_samples_leaf=6,max_features=.7,n_jobs=-1,random_state=20+f),
          make_pipeline(StandardScaler(),Ridge(alpha=300))]
        for m in models:m.fit(X[tr],Y[tr])
        coef=sum(m.predict(X[va]) for m in models)/len(models)
        se=bs=nr=0
        for q,j in enumerate(va):
            w=wells[j];corr=np.interp(w["prog"],KNOTS,coef[q])
            se+=np.square(corr-w["target"]).sum();bs+=np.square(w["target"]).sum();nr+=len(corr)
            oof[w["global_idx"]]=w["base"]+corr;fid[w["global_idx"]]=f
        fr.append(dict(fold=f,rows=nr,rmse=float(np.sqrt(se/nr)),base_rmse=float(np.sqrt(bs/nr))))
        joblib.dump(models,OUT/f"fold{f}.joblib");print(json.dumps(fr[-1]),flush=True)
    total=float(np.sqrt(np.average([x["rmse"]**2 for x in fr],weights=[x["rows"] for x in fr])))
    base=float(np.sqrt(np.average([x["base_rmse"]**2 for x in fr],weights=[x["rows"] for x in fr])))
    summary=dict(pooled_rmse=total,base_rmse=base,gain=base-total,folds=fr,wells=len(wells),
      complete_well_samples=True,no_neighbor_data=True,warp_knots=len(KNOTS))
    np.savez_compressed(OUT/"oof_delta.npz",prediction=oof,fold=fid,
      target=feat.target.to_numpy(np.float32),ids=feat.id.to_numpy(str))
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
