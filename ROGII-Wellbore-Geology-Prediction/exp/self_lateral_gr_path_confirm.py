"""Locked, disjoint confirmation of self-lateral complete-path matching."""
from pathlib import Path
import json, sys
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"exp"))
from whole_well_gr_warp_selector_cv import candidate_paths, rmse
from self_lateral_gr_path_cv import self_scores, weighted_path_fast


def stratified(stats, seed, n, exclude=()):
    s=stats[~stats.well.isin(exclude)].copy()
    s["bin"]=pd.qcut(s.base_rmse,4,labels=False)
    rng=np.random.RandomState(seed); out=[]
    for _,g in s.groupby("bin"):
        out.extend(rng.choice(g.well,min(len(g),n//4),False))
    return out


def main():
    frame=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oof=joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    stats=[]
    for well,g in frame.groupby("well"):
        base,_,_=candidate_paths(g,oof);stats.append((well,rmse(g.target,base)))
    stats=pd.DataFrame(stats,columns=["well","base_rmse"])
    tuned=stratified(stats,913,200)
    selected=stratified(stats,1776,200,tuned)
    yy=[];bb=[];pp=[]; per=[];used=0
    for i,(well,g) in enumerate(frame[frame.well.isin(selected)].groupby("well",sort=True)):
        hw=pd.read_csv(ROOT/f"data/train/{well}__horizontal_well.csv")
        base,paths,labels=candidate_paths(g,oof)
        evidence,support=self_scores(hw,g,paths,4.0)
        eligible=support>=.03
        if eligible.any():
            lab=np.asarray(labels,float)
            complexity=(lab[:,1]/20)**2+(lab[:,2]/12)**2+(lab[:,3]/6)**2
            cost=-evidence+.6*complexity;cost[~eligible]=np.inf
            finite=np.isfinite(cost);scale=np.std(cost[finite])*.1+1e-6
            weight=np.zeros(len(cost));weight[finite]=np.exp(np.clip(-(cost[finite]-cost[finite].min())/scale,-30,0));weight/=weight.sum()
            post=weighted_path_fast(weight,paths,labels);pred=.5*base+.5*post;used+=1
        else: pred=base
        y=g.target.to_numpy(float);yy.append(y);bb.append(base);pp.append(pred)
        per.append({"well":well,"baseline":rmse(y,base),"self_match":rmse(y,pred),"support":float(support.max())})
        if (i+1)%20==0:print(f"wells={i+1}",flush=True)
    ya,ba,pa=map(np.concatenate,(yy,bb,pp)); table=pd.DataFrame(per)
    summary={"wells":len(per),"rows":len(ya),"baseline":rmse(ya,ba),"self_match":rmse(ya,pa),
             "gain":rmse(ya,ba)-rmse(ya,pa),"well_wins":int((table.self_match<table.baseline).sum()),
             "used_wells":used,"disjoint_from_tuning":True,"locked_parameters":True}
    out=ROOT/"exp/results/self_lateral_gr_path_confirm";out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/"wells.csv",index=False);(out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
