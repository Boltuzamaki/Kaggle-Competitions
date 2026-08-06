"""Complete-well candidate ranker using raw fixed-bin legal vectors."""
from pathlib import Path
import json,sys,joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/raw_vector_complete_well_ranker";OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/"exp"))
from whole_well_gr_warp_selector_cv import robust_calibrate,rmse

def paths_for(g,oof):
    ix=g.index.to_numpy();base=np.asarray(oof["lgb7"])[ix].astype(float)
    u=np.linspace(-1,1,len(g));B=np.c_[np.ones(len(g)),u,u*u-1/3,u**3-.6*u]
    coef=[];paths=[]
    for a in (-20,-12,-7,-3,0,3,7,12,20):
      for b in (-12,-6,0,6,12):
       for c in (-6,0,6):
        for d in (-4,0,4):
         z=np.array([a,b,c,d]);coef.append(z);paths.append(base+B@z)
    return base,np.asarray(paths),np.asarray(coef)

def fixed(v,n=64):
    return np.interp(np.linspace(0,len(v)-1,n),np.arange(len(v)),v)

def main(limit=200):
 d=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
 oof=joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
 stats=[]
 for w,g in d.groupby("well"):
  b,_,_=paths_for(g,oof);stats.append((w,rmse(g.target,b)))
 st=pd.DataFrame(stats,columns=["well","err"]);st["bin"]=pd.qcut(st.err,4,labels=False)
 rng=np.random.default_rng(932);sel=[]
 for _,q in st.groupby("bin"):sel.extend(rng.choice(q.well,min(len(q),limit//4),False))
 d=d[d.well.isin(sel)]
 feats=[];meta=[];paths_by={};truth_by={};base_by={}
 for wi,(w,g) in enumerate(d.groupby("well",sort=True)):
  hw=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
  tw=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
  hs=pd.to_numeric(hw.GR,errors="coerce").interpolate(limit_direction="both")
  hall=hs.fillna(hs.median()).to_numpy(float)
  hr=g.id.astype(str).str.rsplit("_",n=1).str[1].astype(int).to_numpy()
  hgr=hall[hr];a,b,pfx=robust_calibrate(hw,tw)
  tt=tw.TVT.to_numpy(float);ts=pd.to_numeric(tw.GR,errors="coerce").interpolate(limit_direction="both")
  tg=ts.fillna(ts.median()).to_numpy(float)
  base,paths,coef=paths_for(g,oof);last=float(g.last_known_tvt.iloc[0])
  cg=a*np.interp((last+paths).ravel(),tt,tg).reshape(paths.shape)+b
  scale=max(1.4826*np.median(np.abs(hgr-np.median(hgr))),8.)
  # Candidate-independent raw context is still preserved as vector columns.
  ctx=np.r_[fixed((hgr-np.median(hgr))/scale),
            fixed(g.d_z.to_numpy(float)/max(np.std(g.d_z),1)),
            fixed(g.dz_dmd.to_numpy(float)),
            fixed(g.gr_s21.to_numpy(float)/max(np.median(g.gr_s21),1))]
  y=g.target.to_numpy(float);loss=np.sqrt(np.mean((paths-y[None])**2,axis=1))
  u64=np.linspace(0,len(g)-1,64)
  for j in range(len(paths)):
   r=(cg[j]-hgr)/scale
   rs=gaussian_filter1d(r,8)
   raw=np.r_[fixed(r),fixed(np.abs(r)),fixed(np.log1p((r/2)**2)),
             fixed(rs),fixed(paths[j]-base),ctx,coef[j]/[20,12,6,4],
             [pfx/30,scale/30,float(np.std(hgr))/30]]
   feats.append(raw.astype(np.float32))
   meta.append((w,j,loss[j]))
  paths_by[w]=paths;truth_by[w]=y;base_by[w]=base
  if (wi+1)%20==0:print("features",wi+1,flush=True)
 X=np.asarray(feats,np.float32);md=pd.DataFrame(meta,columns=["well","candidate","loss"])
 # Higher label is better, normalized within each candidate group.
 md["label"]=md.groupby("well").loss.transform("max")-md.loss
 groups=md.well.to_numpy();pred=np.zeros(len(md));folds=[]
 for k,(tr,va) in enumerate(GroupKFold(5).split(X,groups=groups)):
  model=CatBoostRanker(iterations=450,depth=7,learning_rate=.045,
    loss_function="YetiRank",l2_leaf_reg=15,random_seed=950+k,
    verbose=False,allow_writing_files=False,thread_count=8,random_strength=.5)
  model.fit(X[tr],md.label.to_numpy()[tr],group_id=groups[tr])
  pred[va]=model.predict(X[va])
  folds.append({"fold":k,"wells":int(md.iloc[va].well.nunique())})
  print("fold",k,flush=True)
 yy=[];bb=[];hard=[];soft=[];oracle=[]
 for w,q in md.assign(score=pred).groupby("well",sort=True):
  s=q.score.to_numpy();pa=paths_by[w]; y0=truth_by[w]
  hard.append(pa[np.argmax(s)])
  temp=.35*np.std(s)+1e-5;wt=np.exp(np.clip((s-s.max())/temp,-30,0));wt/=wt.sum()
  soft.append(wt@pa);yy.append(y0);bb.append(base_by[w])
  oracle.append(pa[np.argmin(q.loss.to_numpy())])
 y=np.concatenate(yy);base=np.concatenate(bb);hard=np.concatenate(hard)
 soft=np.concatenate(soft);oracle=np.concatenate(oracle)
 grid=[]
 for name,p in (("hard",hard),("soft",soft)):
  for a0 in (0,.05,.1,.15,.2,.3,.4,.55,.7,1):
   grid.append({"method":name,"blend":a0,"rmse":rmse(y,(1-a0)*base+a0*p)})
 grid=pd.DataFrame(grid).sort_values("rmse")
 summary={"wells":len(paths_by),"candidate_rows":len(md),"feature_columns":X.shape[1],
  "rows":len(y),"baseline":rmse(y,base),"candidate_oracle":rmse(y,oracle),
  "hard":rmse(y,hard),"soft":rmse(y,soft),"best":grid.iloc[0].to_dict(),"folds":folds}
 grid.to_csv(OUT/"grid.csv",index=False);md.assign(score=pred).to_parquet(OUT/"candidate_oof.parquet",index=False)
 (OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=="__main__":main(int(sys.argv[1]) if len(sys.argv)>1 else 200)
