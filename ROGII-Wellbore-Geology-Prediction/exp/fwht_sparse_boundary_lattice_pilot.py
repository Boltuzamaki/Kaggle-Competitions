"""Fixed-120 legal pilot: sparse Walsh-boundary alignment around Stack-V4.

Truth is used only after all candidate paths have been decoded.  The matcher
uses horizontal GR, typewell TVT/GR, and the honest grouped Stack-V4 OOF path.
"""
from pathlib import Path
import json, time
import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/fwht_sparse_boundary_lattice"
OUT.mkdir(parents=True, exist_ok=True)
SEED, N_WELLS = 912, 120
SHIFTS = np.arange(-24, 25, dtype=float)
ORDER_CACHE = {}


def robust_norm(x):
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy(float)
    lo, hi = np.nanpercentile(x, [1, 99])
    return np.clip((x-lo) / max(hi-lo, 1e-6), 0, 1)


def fwht(a):
    a = np.asarray(a, float).copy(); n = len(a); h = 1
    while h < n:
        block=a.reshape(-1,2*h); x=block[:,:h].copy(); y=block[:,h:].copy()
        block[:,:h]=x+y; block[:,h:]=x-y
        h *= 2
    return a


def bit_reverse(x, bits):
    y = 0
    for _ in range(bits): y=(y << 1) | (x & 1); x >>= 1
    return y


def walsh_lowpass_segment(x, keep_ratio):
    n=len(x); bits=int(np.log2(n)); coef=fwht(x)
    # FWHT is Hadamard ordered; sequency index i maps through bitreverse(gray(i)).
    if n not in ORDER_CACHE:
        ORDER_CACHE[n]=np.array([bit_reverse(i ^ (i >> 1), bits) for i in range(n)])
    order=ORDER_CACHE[n]
    keep=max(2, int(np.ceil(n*keep_ratio)))
    mask=np.zeros(n, bool); mask[order[:keep]]=True; coef[~mask]=0
    return fwht(coef)/n


def walsh_filter(x, keep_ratio, minimum=32):
    """Greedy dyadic segmentation; mask seams later through zero derivatives."""
    out=np.asarray(x,float).copy(); seams=[]; start=0; remaining=len(x)
    while remaining >= minimum:
        n=2**int(np.floor(np.log2(remaining)))
        out[start:start+n]=walsh_lowpass_segment(x[start:start+n],keep_ratio)
        start += n; remaining -= n
        if remaining: seams.append(start)
    if remaining and start:
        out[start:]=out[start-1]
    return out, seams


def boundary_signal(x, keep_ratio):
    z=robust_norm(x)
    if len(z)>=7: z=savgol_filter(z, 7, 2, mode="interp")
    f,seams=walsh_filter(z,keep_ratio)
    d=np.r_[0.,np.diff(f)]
    for s in seams:
        d[max(0,s-1):min(len(d),s+2)]=0
    return d


def sparse_indices(d, q, spacing):
    threshold=np.quantile(np.abs(d),q); candidates=np.flatnonzero(np.abs(d)>=threshold)
    order=candidates[np.argsort(-np.abs(d[candidates]))]; chosen=[]
    for j in order:
        if all(abs(j-k)>=spacing for k in chosen): chosen.append(int(j))
    return np.array(sorted(chosen),int)


def build_emission(path, hgr, tw_tvt, tw_gr, cfg, boundary_cache):
    key=cfg["keep"]
    if key not in boundary_cache:
        boundary_cache[key]=(boundary_signal(hgr,key),boundary_signal(tw_gr,key))
    hd,td=boundary_cache[key]
    anchors=sparse_indices(hd,cfg["quantile"],cfg["spacing"])
    if len(anchors)<5: return None, len(anchors)
    # Aggregate sparse evidence in fixed 32-station blocks.
    nb=int(np.ceil(len(path)/32)); emission=np.zeros((nb,len(SHIFTS)))
    scale=max(np.median(np.abs(hd[anchors]))+np.median(np.abs(td))+1e-4,.02)
    for j in anchors:
        b=min(j//32,nb-1); pos=path[j]+SHIFTS
        ref=np.interp(pos,tw_tvt,td,left=0,right=0)
        mag=np.minimum(np.abs(hd[j]-ref)/scale,4.0)
        sign=(np.sign(hd[j])*np.sign(ref)<0).astype(float)
        emission[b] += mag + cfg["sign_penalty"]*sign
    return emission, len(anchors)


def decode(path, emission, cfg):
    if emission is None: return path.copy()
    nb=len(emission)
    # Viterbi over correction states, allowing at most one foot/block.
    cost=np.full_like(emission,np.inf); parent=np.zeros_like(emission,dtype=np.int16)
    cost[0]=emission[0]+cfg["center_penalty"]*(SHIFTS/24.)**2
    for b in range(1,nb):
        prev=cost[b-1]; cand=np.vstack([
            np.r_[np.inf,prev[:-1]+cfg["transition"]], prev,
            np.r_[prev[1:]+cfg["transition"],np.inf]])
        choice=np.argmin(cand,axis=0); parent[b]=np.arange(len(SHIFTS))+choice-1
        cost[b]=emission[b]+cand[choice,np.arange(len(SHIFTS))]
    states=np.empty(nb,int); states[-1]=int(np.argmin(cost[-1]))
    for b in range(nb-1,0,-1): states[b-1]=parent[b,states[b]]
    centers=np.minimum(np.arange(len(path))//32,nb-1)
    correction=SHIFTS[states[centers]]
    return path+cfg["shrink"]*correction


def main():
    t0=time.time(); frame=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    base=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"],float)
    wells=pd.Series(frame.well).drop_duplicates().to_numpy()
    sample=set(np.random.RandomState(SEED).choice(wells,N_WELLS,replace=False))
    configs=[]
    for keep in (1/32,1/16):
      for quantile in (.85,.92):
       for spacing in (16,32):
        for transition in (1.,4.,16.):
         for shrink in (.15,.3,.5):
          configs.append(dict(keep=keep,quantile=quantile,spacing=spacing,
             transition=transition,shrink=shrink,sign_penalty=.5,center_penalty=2.))
    agg=[dict(sse=0.,wins=0,anchors=[]) for _ in configs]; base_sse=0.; rows=0; detail=[]
    for wi,(well,g) in enumerate(frame[frame.well.isin(sample)].groupby("well",sort=True)):
        ix=g.index.to_numpy(); station=g.id.astype(str).str.rsplit("_",n=1).str[1].astype(int).to_numpy()
        hw=pd.read_csv(ROOT/f"data/train/{well}__horizontal_well.csv")
        tw=pd.read_csv(ROOT/f"data/train/{well}__typewell.csv").sort_values("TVT")
        truth=g.last_known_tvt.to_numpy(float)+g.target.to_numpy(float)
        path=g.last_known_tvt.to_numpy(float)+base[ix]
        hgr=hw.GR.interpolate(limit_direction="both").to_numpy(float)[station]
        tw_tvt=tw.TVT.to_numpy(float); tw_gr=tw.GR.interpolate(limit_direction="both").to_numpy(float)
        bs=float(np.square(path-truth).sum()); base_sse+=bs; rows+=len(g)
        wellrow={"well":str(well),"rows":len(g),"base_rmse":float(np.sqrt(bs/len(g)))}
        boundary_cache={}; emission_cache={}
        for ci,cfg in enumerate(configs):
            ekey=(cfg["keep"],cfg["quantile"],cfg["spacing"])
            if ekey not in emission_cache:
                emission_cache[ekey]=build_emission(path,hgr,tw_tvt,tw_gr,cfg,boundary_cache)
            emission,na=emission_cache[ekey]
            pred=decode(path,emission,cfg); ss=float(np.square(pred-truth).sum())
            agg[ci]["sse"]+=ss; agg[ci]["wins"]+=int(ss<bs); agg[ci]["anchors"].append(na)
        detail.append(wellrow)
        if (wi+1)%20==0: print("wells",wi+1,"seconds",round(time.time()-t0,1),flush=True)
    grid=[]
    for cfg,a in zip(configs,agg):
        grid.append({**cfg,"rmse":float(np.sqrt(a["sse"]/rows)),"well_wins":a["wins"],
                     "median_anchors":float(np.median(a["anchors"]))})
    grid=pd.DataFrame(grid).sort_values("rmse"); best=grid.iloc[0].to_dict()
    summary={"protocol":"fixed seed-912 120-well target-independent sample; legal horizontal GR, typewell TVT/GR, honest grouped V4 OOF center; truth scoring only",
      "wells":N_WELLS,"rows":rows,"baseline":float(np.sqrt(base_sse/rows)),"best":best,
      "gate":bool(best["rmse"] < np.sqrt(base_sse/rows)-.15 and best["well_wins"]>=72),
      "seconds":time.time()-t0,"paper":"https://doi.org/10.3389/feart.2026.1736164"}
    grid.to_csv(OUT/"grid.csv",index=False); pd.DataFrame(detail).to_csv(OUT/"wells.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
