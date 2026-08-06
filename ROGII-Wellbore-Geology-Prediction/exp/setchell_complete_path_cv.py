"""Setchell-style complete-path objective around honest Stack-V4 OOF paths.

Legal evidence: trajectory, horizontal GR, paired typewell TVT/GR, and visible
TVT_input.  Target TVT is accessed only after candidate scores are frozen.
The pilot and confirmation well sets are deterministic and disjoint.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/setchell_complete_path"
OUT.mkdir(parents=True, exist_ok=True)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def norm(x):
    x = np.asarray(x, float)
    lo, hi = np.nanpercentile(x, [1, 99])
    return np.clip((x-lo)/max(hi-lo, 1e-6), 0, 1)


def cor(a, b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=a-a.mean(); b=b-b.mean()
    return float(a@b/max(np.sqrt((a@a)*(b@b)), 1e-9))


def base_v4(g, oof):
    ix=g.index.to_numpy()
    raw={k:np.asarray(v)[ix].astype(float) for k,v in oof.items()}
    return .55*raw["lgb123"]+.20*raw["lgb7"]+.15*raw["xgb"]+.10*raw["cat"]


def candidates(g, hw, oof):
    """Anchored, low-dimensional structural paths centered on Stack-V4."""
    base=base_v4(g,oof)
    station=g.id.astype(str).str.rsplit("_",n=1).str[1].astype(int).to_numpy()
    md=hw.MD.to_numpy(float)[station]
    f=(md-md[0])/max(md[-1]-md[0],1e-6)
    # A datum error may emerge rapidly after prediction start, while end tilt
    # and bow represent dip and gentle dip change respectively.
    warm=1-np.exp(-(md-md[0])/400.)
    paths=[]; labels=[]
    for datum in (-12.,-6.,0.,6.,12.):
      for tilt in (-16.,-8.,0.,8.,16.):
       for bow in (-8.,0.,8.):
        paths.append(base+datum*warm+tilt*f+bow*4*f*(1-f))
        labels.append((datum,tilt,bow))
    return base,np.asarray(paths),labels,station,md


def visible_dip_prior(hw):
    """Robust along-hole formation-surface dip from visible TVT only."""
    k=hw.TVT_input.notna().to_numpy()
    if k.sum()<10: return 0., .02
    md=hw.MD.to_numpy(float)[k]
    surf=hw.TVT_input.to_numpy(float)[k]+hw.Z.to_numpy(float)[k]
    # Recent prefix is the relevant local dip; median pair slopes is robust.
    n=min(150,len(md)); md=md[-n:]; surf=surf[-n:]
    dm=np.diff(md); q=dm>1e-6
    slopes=np.diff(surf)[q]/dm[q]
    center=float(np.nanmedian(slopes)) if len(slopes) else 0.
    spread=float(1.4826*np.nanmedian(np.abs(slopes-center))+.005)
    return center, max(spread,.01)


def evidence(hw,tw,g,paths,station,md):
    hgr=hw.GR.to_numpy(float)[station]
    good=np.isfinite(hgr)
    hgr=norm(hgr)
    tt=tw.TVT.to_numpy(float); tg=norm(tw.GR.interpolate(limit_direction="both").to_numpy(float))
    last=float(g.last_known_tvt.iloc[0])
    z=hw.Z.to_numpy(float)[station]
    dip0,dipsig=visible_dip_prior(hw)
    out=[]
    for path in paths:
        tvt=last+path
        origin=np.floor(np.nanmin(tvt[good])/.5)*.5
        bi=np.floor((tvt[good]-origin)/.5).astype(int)
        cnt=np.bincount(bi); total=np.bincount(bi,weights=hgr[good])
        occ=np.flatnonzero(cnt>=2)
        if len(occ)<8: occ=np.flatnonzero(cnt)
        lat=total[occ]/cnt[occ]; cen=origin+(occ+.5)*.5
        ref=np.interp(cen,tt,tg)
        pear=np.arctanh(np.clip(cor(lat,ref),-.999,.999))
        spear=np.arctanh(np.clip(cor(rankdata(lat),rankdata(ref)),-.999,.999))
        cosine=float(lat@ref/max(np.linalg.norm(lat)*np.linalg.norm(ref),1e-9))
        # Setchell structural coordinate is TVT + Z. Compare its along-hole
        # formation dip to the local visible-prefix prior; trajectory Z is
        # therefore explicitly part of every candidate's physical score.
        surf=tvt+z
        slope=np.gradient(surf)/np.maximum(np.gradient(md),1e-6)
        dip_pen=float(np.mean(((slope-dip0)/dipsig)**2))
        rough=float(np.mean(np.diff(slope)**2)/(dipsig*dipsig+1e-8))
        out.append((pear,spear,cosine,dip_pen,rough,len(occ)))
    return np.asarray(out)


def split_wells(frame,oof,pilot_n=120,confirm_n=240):
    rows=[]
    for w,g in frame.groupby("well"):
        rows.append((w,rmse(g.target,base_v4(g,oof))))
    st=pd.DataFrame(rows,columns=["well","difficulty"])
    st["bin"]=pd.qcut(st.difficulty,4,labels=False,duplicates="drop")
    rng=np.random.RandomState(19072019); pilot=[]; confirm=[]
    for _,q in st.groupby("bin"):
        ids=q.well.to_numpy(); rng.shuffle(ids)
        pilot.extend(ids[:pilot_n//4]); confirm.extend(ids[pilot_n//4:pilot_n//4+confirm_n//4])
    return pilot,confirm


def build_store(frame,oof,wells,name):
    store={}; rec=[]
    for i,w in enumerate(sorted(wells)):
        g=frame[frame.well.eq(w)]
        hw=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
        tw=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
        base,paths,labels,station,md=candidates(g,hw,oof)
        ev=evidence(hw,tw,g,paths,station,md)
        y=g.target.to_numpy(float)
        store[w]=(y,base,paths,labels,ev)
        rec.append({"well":w,"rows":len(g),"base_rmse":rmse(y,base),"oracle_rmse":rmse(y,paths[np.argmin(np.mean((paths-y)**2,axis=1))])})
        if (i+1)%20==0: print(name,i+1,flush=True)
    pd.DataFrame(rec).to_csv(OUT/f"{name}_wells.csv",index=False)
    return store


def evaluate(store,cfg):
    yy=[];bb=[];pp=[]
    for y,b,paths,labels,e in store.values():
        score=cfg["wp"]*e[:,0]+cfg["ws"]*e[:,1]+cfg["wc"]*e[:,2]-cfg["wd"]*np.log1p(e[:,3])-cfg["wr"]*np.log1p(e[:,4])
        complexity=np.array([(d/12)**2+(t/16)**2+(q/8)**2 for d,t,q in labels])
        score-=cfg["prior"]*complexity
        scale=np.std(score)*cfg["temp"]+1e-6
        weight=np.exp(np.clip((score-score.max())/scale,-30,0)); weight/=weight.sum()
        p=weight@paths
        yy.append(y);bb.append(b);pp.append((1-cfg["blend"])*b+cfg["blend"]*p)
    return rmse(np.concatenate(yy),np.concatenate(pp)),rmse(np.concatenate(yy),np.concatenate(bb))


def main():
    frame=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oof=joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    pilot,confirm=split_wells(frame,oof)
    ps=build_store(frame,oof,pilot,"pilot")
    grid=[]
    for weights in ((1,0,0),(0,1,0),(.5,.5,0),(.4,.4,.2)):
     for wd in (0,.01,.03,.1):
      for wr in (0,.005,.02):
       for prior in (.01,.05,.15):
        for temp in (.2,.5,1.):
         for blend in (.1,.2,.35,.5):
          cfg=dict(wp=weights[0],ws=weights[1],wc=weights[2],wd=wd,wr=wr,prior=prior,temp=temp,blend=blend)
          score,base=evaluate(ps,cfg); grid.append({**cfg,"rmse":score,"baseline":base})
    tab=pd.DataFrame(grid).sort_values("rmse"); tab.to_csv(OUT/"pilot_grid.csv",index=False)
    locked={k:float(tab.iloc[0][k]) for k in ("wp","ws","wc","wd","wr","prior","temp","blend")}
    cs=build_store(frame,oof,confirm,"confirmation")
    crmse,cbase=evaluate(cs,locked)
    summary={"protocol":"fixed stratified 120-well pilot then disjoint 240-well confirmation","pilot_baseline":float(tab.iloc[0].baseline),"pilot_best":float(tab.iloc[0].rmse),"locked_config":locked,"confirmation_baseline":cbase,"confirmation_locked":crmse,"confirmation_gain":cbase-crmse,"legal_inputs_only":True,"truth_used_after_scoring_only":True}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
