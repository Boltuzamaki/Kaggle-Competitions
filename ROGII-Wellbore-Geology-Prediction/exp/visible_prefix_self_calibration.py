"""Legal rolling-origin selection of deterministic per-well forecasts."""
from pathlib import Path
import glob,json,joblib
import numpy as np,pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data/train"
OUT=ROOT/"exp/results/visible_prefix_self_calibration";OUT.mkdir(parents=True,exist_ok=True)
CANDS=["flat","lin100_h50","lin100","lin300_h50","lin300","lin800_h50","lin800",
       "quad500","quad1200"]

def fit_predict(h,cut,end,name):
 md=h.MD.to_numpy(float);z=h.Z.to_numpy(float)
 tv=h.TVT_input.to_numpy(float);S=tv[:cut]+z[:cut]
 q=np.arange(cut,end);x=md[:cut]-md[cut-1];xf=md[q]-md[cut-1]
 if name=="flat": sp=np.full(len(q),S[-1])
 elif name.startswith("lin"):
  win=int(name.split("_")[0][3:]);ii=np.arange(max(0,cut-win),cut)
  slope=np.polyfit(x[ii],S[ii],1)[0];slope=float(np.clip(slope,-.10,.10))
  if name.endswith("h50"):slope*=.5
  sp=S[-1]+slope*xf
 else:
  win=int(name[4:]);ii=np.arange(max(0,cut-win),cut)
  # Fit in local coordinates and constrain curvature to a geological range.
  co=np.polyfit(x[ii],S[ii],2);co[0]=np.clip(co[0],-2e-6,2e-6)
  slope=np.clip(co[1],-.10,.10)
  sp=S[-1]+slope*xf+co[0]*xf*xf
 return sp-z[q]

def gr_shift(h,t,cut,pred,rows):
 """Bounded correction using only always-visible GR and typewell TVT/GR."""
 tt=t.TVT.to_numpy(float);tg=t.GR.interpolate(limit_direction="both").fillna(t.GR.median()).to_numpy(float)
 hg=h.GR.interpolate(limit_direction="both").fillna(h.GR.median()).to_numpy(float)
 ref=np.interp(h.TVT_input.iloc[:cut],tt,tg)
 aa,bb=np.linalg.lstsq(np.c_[ref,np.ones(cut)],hg[:cut],rcond=None)[0]
 aa=float(np.clip(aa,.25,2.5));idx=np.linspace(0,len(rows)-1,min(500,len(rows))).astype(int)
 scale=max(np.median(abs(hg[:cut]-(aa*ref+bb)))*1.4826,5)
 shifts=np.arange(-8,8.1,1);cost=[]
 for s in shifts:
  z=(hg[rows[idx]]-(aa*np.interp(pred[idx]+s,tt,tg)+bb))/scale
  cost.append(np.mean(np.log1p((z/2)**2)))
 p=np.exp(-(np.array(cost)-min(cost))/.2);p/=p.sum()
 return float(np.clip(.2*(p@shifts),-3,3))

def candidates(h,t,cut,end):
 rows=np.arange(cut,end);out={}
 for name in CANDS:
  p=fit_predict(h,cut,end,name);out[name]=p
  out[name+"_gr"]=p+gr_shift(h,t,cut,p,rows)
 return out

feat=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
v4=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"])
bywell={w:q.index.to_numpy() for w,q in feat.groupby("well",sort=False)}
oof=np.full(len(feat),np.nan,np.float32);rows=[]
all_suffix={k:[] for k in CANDS+[x+"_gr" for x in CANDS]};all_y=[]
for path in sorted(glob.glob(str(DATA/"*__horizontal_well.csv"))):
 wid=Path(path).name.split("__")[0];h=pd.read_csv(path)
 t=pd.read_csv(DATA/f"{wid}__typewell.csv").sort_values("TVT")
 ps=int(h.TVT_input.notna().sum())
 if ps<500 or ps>=len(h):continue
 # Multiple rolling origins, always strictly inside visible TVT_input.
 origins=[]
 for back in (900,600,350):
  cut=ps-back
  if cut>=300:
   end=min(ps,cut+min(500,back))
   origins.append((cut,end))
 scores={k:[] for k in all_suffix}
 for cut,end in origins:
  cp=candidates(h,t,cut,end);truth=h.TVT.iloc[cut:end].to_numpy(float)
  for k,p in cp.items():scores[k].append(float(np.sqrt(np.mean((p-truth)**2))))
 mean={k:float(np.mean(v)) for k,v in scores.items()}
 chosen=min(mean,key=mean.get)
 cp=candidates(h,t,ps,len(h));truth=h.TVT.iloc[ps:].to_numpy(float)
 for k,p in cp.items():all_suffix[k].append(p)
 all_y.append(truth);pred=cp[chosen];gi=bywell[wid];oof[gi]=pred-h.TVT_input.iloc[ps-1]
 suffix_rmse={k:float(np.sqrt(np.mean((p-truth)**2))) for k,p in cp.items()}
 oracle=min(suffix_rmse,key=suffix_rmse.get)
 rows.append(dict(well=wid,rows=len(truth),chosen=chosen,oracle=oracle,
  pseudo_rmse=mean[chosen],suffix_rmse=suffix_rmse[chosen],
  oracle_rmse=suffix_rmse[oracle],choice_correct=chosen==oracle))

wr=pd.DataFrame(rows);y=np.concatenate(all_y)
fixed={}
for k,parts in all_suffix.items():fixed[k]=float(np.sqrt(np.mean((np.concatenate(parts)-y)**2)))
used=np.concatenate([bywell[w] for w in wr.well])
route=float(np.sqrt(np.mean((oof[used]-feat.target.to_numpy(float)[used])**2)))
v4rm=float(np.sqrt(np.mean((v4[used]-feat.target.to_numpy(float)[used])**2)))
rho,pval=spearmanr(wr.pseudo_rmse,wr.suffix_rmse)
weighted_oracle=float(np.sqrt(np.average(wr.oracle_rmse**2,weights=wr.rows)))
summary={"wells":len(wr),"rows":int(wr.rows.sum()),"origins_back":[900,600,350],
 "candidates":list(fixed),"routed_rmse":route,"v4_rmse":v4rm,
 "best_fixed":min(fixed,key=fixed.get),"best_fixed_rmse":min(fixed.values()),
 "per_well_oracle_rmse":weighted_oracle,
 "pseudo_vs_suffix_spearman":float(rho),"spearman_p":float(pval),
 "exact_choice_rate":float(wr.choice_correct.mean()),
 "selection_uses_suffix_truth":False,"no_neighbors":True}
np.savez_compressed(OUT/"oof_delta.npz",prediction=oof,target=feat.target.to_numpy(np.float32),
                    ids=feat.id.to_numpy(str),global_indices=used)
wr.to_csv(OUT/"well_metrics.csv",index=False)
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
