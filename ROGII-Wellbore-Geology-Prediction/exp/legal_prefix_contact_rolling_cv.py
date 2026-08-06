"""Strict rolling-suffix CV for legal prefix-derived structural contacts.

Only official common-schema columns MD/X/Y/Z/GR and the visible TVT_input prefix
are read.  In particular this experiment never opens typewells, formation
surfaces, same-ID twins, public predictions, or hidden TVT before predictions
are frozen.  A "contact" is the structural datum S = TVT + Z.  Candidate
continuations extrapolate S from the visible prefix and recover TVT=S-Z.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/legal_prefix_contact_rolling"
OUT.mkdir(parents=True, exist_ok=True)


def line(xfit, yfit, xq):
    x0 = xfit[-1]
    xx = xfit - x0
    den = np.sum((xx-xx.mean())**2)
    slope = np.sum((xx-xx.mean())*(yfit-yfit.mean())) / max(den, 1e-12)
    return yfit.mean() + slope*(xq-x0-xx.mean())


def candidates(d, cut):
    """Freeze legal candidates using rows [0, cut), return complete suffix."""
    z=d.Z.to_numpy(float); md=d.MD.to_numpy(float)
    x=d.X.to_numpy(float); y=d.Y.to_numpy(float)
    vis=d.TVT_input.to_numpy(float)[:cut]
    s=vis+z[:cut]
    q=np.arange(cut,len(d)); out={}
    # Several estimates of a locally planar structural contact.
    for w in (100,250,500,1000,2000):
        a=max(0,cut-w); out[f"mdlin{w}"]=line(md[a:cut],s[a:cut],md[q])-z[q]
        out[f"const{w}"]=np.full(len(q),np.median(s[a:cut]))-z[q]
        A=np.c_[x[a:cut]-x[cut-1],y[a:cut]-y[cut-1],np.ones(cut-a)]
        coef=np.linalg.lstsq(A,s[a:cut],rcond=None)[0]
        out[f"plane{w}"]=(np.c_[x[q]-x[cut-1],y[q]-y[cut-1],np.ones(len(q))]@coef)-z[q]
    # Consensus contacts reduce window instability.
    out["cons_lin"] = np.median(np.stack([out[f"mdlin{w}"] for w in (250,500,1000,2000)]),axis=0)
    out["cons_plane"] = np.median(np.stack([out[f"plane{w}"] for w in (250,500,1000,2000)]),axis=0)
    out["last_tvt"] = np.full(len(q),vis[-1])
    return out


def select_from_prefix(d, ps, names):
    """Rolling origins wholly inside prefix; evaluate proportional long tails."""
    # Multiple origins with meaningful horizons, emphasizing the latest origin.
    origins=sorted(set([int(ps*f) for f in (.45,.55,.65,.75)]))
    score={k:[] for k in names}
    truth=d.TVT_input.to_numpy(float)
    for cut in origins:
        if cut<100 or ps-cut<50: continue
        cc=candidates(d.iloc[:ps].copy(),cut)
        for k in names:
            e=truth[cut:ps]-cc[k]
            score[k].append(float(np.sqrt(np.mean(e*e))))
    med={k:np.median(v) if v else np.inf for k,v in score.items()}
    return min(med,key=med.get),med


rows=[]; truth=[]; preds={}; choices=[]
files=sorted((ROOT/"data/train").glob("*__horizontal_well.csv"))
for fi,f in enumerate(files):
    # Explicit common-schema projection prevents accidental formation access.
    raw=pd.read_csv(f,usecols=["MD","X","Y","Z","TVT","GR","TVT_input"])
    # Match the official/reference harness exactly: PS is the count of visible
    # TVT_input rows (the files can contain isolated missing values).
    ps=int(raw.TVT_input.notna().sum())
    if ps<100 or ps>=len(raw)-5: continue
    legal=raw[["MD","X","Y","Z","GR","TVT_input"]].copy()
    cc=candidates(legal,ps)
    if not preds: preds={k:[] for k in cc}
    ytrue=raw.TVT.to_numpy(float)[ps:].copy() # accessed only after cc frozen
    truth.append(ytrue)
    for k,v in cc.items(): preds[k].append(v)
    selectable=[k for k in cc if k not in {"last_tvt"}]
    winner,score=select_from_prefix(legal,ps,selectable)
    preds.setdefault("rolling_selected",[]).append(cc[winner])
    choices.append(dict(well=f.name.split("__")[0],rows=len(ytrue),ps=ps,winner=winner,
                        prefix_cv=float(score[winner])))
    if (fi+1)%100==0: print("processed",fi+1,flush=True)

yy=np.concatenate(truth)
metrics={k:float(np.sqrt(np.mean((yy-np.concatenate(v))**2))) for k,v in preds.items()}
ranked=sorted(metrics.items(),key=lambda q:q[1])
pd.DataFrame(ranked,columns=["candidate","pooled_rmse"]).to_csv(OUT/"scores.csv",index=False)
pd.DataFrame(choices).to_csv(OUT/"rolling_choices.csv",index=False)
np.savez_compressed(OUT/"oof.npz",y=yy,
                    **{k:np.concatenate(v) for k,v in preds.items()})
summary={"protocol":"strict organizer TVT_input masks; common schema only",
         "wells":len(truth),"rows":len(yy),"best":ranked[0],"scores":dict(ranked),
         "forbidden_inputs_read":False,
         "columns_read":["MD","X","Y","Z","TVT","GR","TVT_input"],
         "target_access":"TVT suffix only after per-well predictions frozen"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
