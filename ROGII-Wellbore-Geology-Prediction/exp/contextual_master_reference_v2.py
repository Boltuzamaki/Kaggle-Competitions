"""Leave-supertype-out master stratigraphic GR reference pilot.

Each unique training typewell contributes robust-normalized GR templates in
formation-relative coordinates.  For a validation well, all copies of its
paired typewell are excluded from the master.  The master is warped onto the
validation typewell's legal Geology intervals, calibrated on the visible
horizontal prefix, and matched sequence-wise around an honest OOF base path.
"""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/contextual_master_reference_v2"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/"exp"))
from contextual_reference_matcher_v1 import OFF, robust_affine, viterbi

U = np.linspace(0, 1, 201)


def fingerprint(t):
    q=t.dropna(subset=["TVT","GR"]).sort_values("TVT")
    x=q.TVT.to_numpy(); g=q.GR.to_numpy()
    return (round(float(x[0]),1), round(float(x[-1]),1),
            round(float(np.nansum(g[:200])),0), len(x))


def robust_z(g):
    m=np.nanmedian(g); s=1.4826*np.nanmedian(np.abs(g-m))
    return (g-m)/max(s, 5.), m, max(s, 5.)


def geology_runs(t):
    lab=t.Geology.astype("string").fillna("UNKNOWN").to_numpy()
    starts=np.r_[0, np.where(lab[1:] != lab[:-1])[0]+1]
    ends=np.r_[starts[1:], len(t)]
    return [(lab[a], a, b) for a,b in zip(starts,ends) if lab[a]!="UNKNOWN" and b-a>=8]


def templates(t):
    z,_,_=robust_z(t.GR.interpolate(limit_direction="both").to_numpy(float))
    out={}
    for lab,a,b in geology_runs(t):
        x=np.linspace(0,1,b-a)
        out.setdefault(str(lab),[]).append(np.interp(U,x,z[a:b]))
    return {k:np.mean(v,axis=0) for k,v in out.items()}


def build_bank():
    reps={}
    for p in sorted((ROOT/"data/train").glob("*__typewell.csv")):
        t=pd.read_csv(p)
        fp=fingerprint(t)
        if fp not in reps:
            reps[fp]=templates(t)
    return reps


def master_for(t, bank, exclude):
    # Aggregate unique typewells, excluding the validation supertype.
    labs={k for fp,d in bank.items() if fp!=exclude for k in d}
    master={}
    for k in labs:
        a=[d[k] for fp,d in bank.items() if fp!=exclude and k in d]
        if len(a)>=3:
            master[k]=np.nanmedian(np.stack(a),axis=0)
    out=np.full(len(t),np.nan)
    for lab,a,b in geology_runs(t):
        if str(lab) in master:
            out[a:b]=np.interp(np.linspace(0,1,b-a),U,master[str(lab)])
    return pd.Series(out).interpolate(limit_direction="both").fillna(0).to_numpy()


def decode(ref, tt, hg, ht, p, base, smooth_ref=True):
    # Calibrate a standardized/master or paired reference to horizontal units.
    a,b=robust_affine(np.interp(ht,tt,ref),hg[:p])
    rr=a*ref+b
    cand=base[:,None]+OFF[None,:]
    ev=hg[p:]
    cost=np.zeros(cand.shape,np.float32)
    sc=max(8.,1.4826*np.median(np.abs(hg[:p]-np.median(hg[:p]))))
    for sig,w in ((1.2,.1),(4,.25),(10,.35),(22,.3)):
        e=gaussian_filter1d(ev,sig,mode="nearest")
        r=gaussian_filter1d(rr,sig,mode="nearest")
        cost += w*np.minimum(np.abs(e[:,None]-np.interp(cand,tt,r))/sc,4)
    cost=gaussian_filter1d(cost,4,axis=0,mode="nearest")
    logits=-(cost-cost.min(1,keepdims=True))/.20
    pr=np.exp(np.clip(logits,-30,0)); pr/=pr.sum(1,keepdims=True)
    soft=base+pr@OFF
    idx=viterbi(cost,.10,.01)
    return base+OFF[idx],soft


def one(w,g,base,bank):
    h=pd.read_csv(ROOT/"data/train"/f"{w}__horizontal_well.csv")
    t=pd.read_csv(ROOT/"data/train"/f"{w}__typewell.csv").sort_values("TVT").reset_index(drop=True)
    p=h.TVT_input.notna().sum()
    if len(h)-p != len(g): return None
    tt=t.TVT.to_numpy(float); tg=t.GR.interpolate(limit_direction="both").to_numpy(float)
    hg=h.GR.interpolate(limit_direction="both").to_numpy(float)
    ht=h.TVT_input.iloc[:p].to_numpy(float)
    absbase=float(g.last_known_tvt.iloc[0])+base
    fp=fingerprint(t)
    mz=master_for(t,bank,fp)
    pz,_,_=robust_z(tg)
    pp,ps=decode(pz,tt,hg,ht,p,absbase)
    mp,ms=decode(mz,tt,hg,ht,p,absbase)
    # Blend references before decoding, not prediction CSVs.
    blendz=.5*pz+.5*mz
    bp,bs=decode(blendz,tt,hg,ht,p,absbase)
    anchor=float(g.last_known_tvt.iloc[0])
    return {"well":w,"y":g.target.to_numpy(float),"base":base,
            "paired_path":pp-anchor,"paired_soft":ps-anchor,
            "master_path":mp-anchor,"master_soft":ms-anchor,
            "blend_path":bp-anchor,"blend_soft":bs-anchor}


def score(rows,key,a=1):
    y=np.concatenate([r["y"] for r in rows])
    p=np.concatenate([(1-a)*r["base"]+a*r[key] for r in rows])
    return float(np.sqrt(np.mean((y-p)**2)))


def main(limit=120):
    bank=build_bank()
    print("unique supertypes",len(bank),flush=True)
    d=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oo=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb123"])
    wells=np.array(sorted(d.well.unique())); rng=np.random.RandomState(823)
    if limit and limit<len(wells): wells=np.sort(rng.choice(wells,limit,False))
    rows=[]
    for j,w in enumerate(wells):
        g=d[d.well==w]; r=one(w,g,oo[g.index],bank)
        if r: rows.append(r)
        if (j+1)%20==0: print("processed",j+1,flush=True)
    base=score(rows,"base")
    grid=[]
    keys=["paired_path","paired_soft","master_path","master_soft","blend_path","blend_soft"]
    for k in keys:
        for a in (.025,.05,.1,.15,.2,.3,.5,1):
            s=score(rows,k,a); grid.append({"kind":k,"blend":a,"rmse":s,"gain":base-s})
    grid=pd.DataFrame(grid).sort_values("rmse")
    summary={"wells":len(rows),"unique_supertypes":len(bank),"base_rmse":base,
             "raw":{k:score(rows,k) for k in keys},"best":grid.iloc[0].to_dict()}
    print(json.dumps(summary,indent=2),flush=True)
    grid.to_csv(OUT/f"grid_{len(rows)}w.csv",index=False)
    (OUT/f"summary_{len(rows)}w.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 120)
