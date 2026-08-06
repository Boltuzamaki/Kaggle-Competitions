"""Same-well visible-prefix GR reference shift posterior around V4 OOF."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/self_prefix_gr_shift";OUT.mkdir(parents=True,exist_ok=True)
f=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
base=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"],float)
y=f.target.to_numpy(float); shifts=np.arange(-60,61,2,dtype=float)

def rmse(p): return float(np.sqrt(np.mean((p-y)**2)))
def ref_curve(tvt,gr):
    # Robust 0.5-ft bins collapse nonmonotonic traversal/duplicate TVT samples.
    z=pd.DataFrame({"b":np.round(np.asarray(tvt,float)*2)/2,"g":gr})
    z=z[np.isfinite(z.b)&np.isfinite(z.g)].groupby("b",as_index=False).g.median()
    x=z.b.to_numpy(float); raw=z.g.to_numpy(float)
    if len(x)<12:return None
    return x,[raw,gaussian_filter1d(raw,2),gaussian_filter1d(raw,6)]

records=[]; well_ix=[]
for wi,(w,ix0) in enumerate(f.groupby("well",sort=False).indices.items()):
    ix=np.asarray(ix0);well_ix.append(ix)
    h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
    hs=pd.to_numeric(h.GR,errors="coerce").interpolate(limit_direction="both")
    hg=hs.fillna(hs.median() if hs.notna().any() else 0).to_numpy(float)
    vis=h.TVT_input.notna().to_numpy()
    ref=ref_curve(h.loc[vis,"TVT_input"].to_numpy(float),hg[vis])
    rec={"well":w,"prefix_tvt_min":float(np.nanmin(h.loc[vis,"TVT_input"])),
         "prefix_tvt_max":float(np.nanmax(h.loc[vis,"TVT_input"]))}
    if ref is None:
        rec["valid"]=False;rec["costs"]=[0.]*len(shifts);rec["support"]=[0.]*len(shifts)
        records.append(rec);continue
    rx,rgs=ref
    rowmap={f"{w}_{j}":j for j in h.index}
    hr=np.array([rowmap[x] for x in f.id.iloc[ix]],int)
    path=f.last_known_tvt.to_numpy(float)[ix]+base[ix]
    truth=f.last_known_tvt.to_numpy(float)[ix]+y[ix]
    take=np.linspace(0,len(ix)-1,min(900,len(ix))).astype(int)
    q=path[take]; obs0=hg[hr]
    obs=[obs0[take],gaussian_filter1d(obs0,5)[take],
         gaussian_filter1d(obs0,15)[take]]
    scale=max(1.4826*np.median(np.abs(hg[vis]-np.median(hg[vis]))),8.)
    costs=[];support=[]
    for s in shifts:
        qq=q+s;m=(qq>=rx[0])&(qq<=rx[-1])
        support.append(float(m.mean()))
        if m.sum()<30:
            costs.append(np.nan);continue
        vals=[]
        for o,g in zip(obs,rgs):
            z=(o[m]-np.interp(qq[m],rx,g))/scale
            vals.append(np.mean(np.log1p((z/2)**2)))
        # Mild deterministic support penalty prevents tiny-overlap winners.
        costs.append(float(np.mean(vals)+.03*(1-m.mean())))
    truth_support=((truth>=rx[0])&(truth<=rx[-1])).mean()
    rec.update(valid=bool(np.isfinite(costs).sum()>=5),costs=costs,support=support,
               truth_support=float(truth_support),
               prefix_span=float(rx[-1]-rx[0]),hidden_truth_span=float(np.ptp(truth)))
    records.append(rec)
    if wi%100==0:print("processed",wi,rec["truth_support"],flush=True)

C=np.asarray([r["costs"] for r in records],float)
valid=np.isfinite(C).sum(1)>=5
# Missing shifts are excluded from posterior, invalid wells get correction zero.
grid=[];cache={}
for temp in (.01,.02,.04,.08,.16,.32):
    A=np.where(np.isfinite(C),C,np.inf); mn=np.min(A,axis=1,keepdims=True)
    logits=-(A-mn)/temp; P=np.where(np.isfinite(logits),np.exp(np.clip(logits,-80,0)),0)
    P/=np.maximum(P.sum(1,keepdims=True),1e-12)
    means=P@shifts;means[~valid]=0
    std=np.sqrt(np.maximum(P@(shifts**2)-means**2,0))
    for hedge in (.05,.1,.15,.2,.25,.35,.5):
        corr=np.zeros(len(f),np.float32)
        for j,ix in enumerate(well_ix):corr[ix]=hedge*means[j]
        score=rmse(base+corr)
        grid.append({"temperature":temp,"hedge":hedge,"rmse":score,
                     "posterior_std":float(np.mean(std[valid]))})
        cache[(temp,hedge)]=corr
g=pd.DataFrame(grid).sort_values("rmse");best=g.iloc[0].to_dict()
corr=cache[(best["temperature"],best["hedge"])]
heel=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")["correction"]
blend=[]
for a in (0,.25,.5,.75,1):
 for b in (0,.25,.5,.75,1):
  blend.append({"self_weight":a,"heel_weight":b,
                "rmse":rmse(base+a*corr+b*heel)})
support=np.array([r.get("truth_support",0) for r in records])
summary={"rows":len(f),"wells":len(records),"base":rmse(base),
 "best_self":best,"best_self_plus_heel":min(blend,key=lambda d:d["rmse"]),
 "typewell_heel":rmse(base+heel),"valid_wells":int(valid.sum()),
 "truth_support_mean":float(support.mean()),"truth_support_median":float(np.median(support)),
 "truth_support_lt25":int((support<.25).sum()),"truth_support_gt75":int((support>.75).sum()),
 "protocol":"same-well visible prefix GR reference only; hidden GR inference-available"}
g.to_csv(OUT/"grid.csv",index=False);pd.DataFrame(blend).to_csv(OUT/"heel_blend.csv",index=False)
pd.DataFrame([{k:v for k,v in r.items() if k not in ("costs","support")} for r in records]).to_csv(OUT/"well_support.csv",index=False)
np.savez_compressed(OUT/"oof.npz",correction=corr,base=base,y=y,groups=f.well.to_numpy())
(OUT/"costs.json").write_text(json.dumps(records))
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
