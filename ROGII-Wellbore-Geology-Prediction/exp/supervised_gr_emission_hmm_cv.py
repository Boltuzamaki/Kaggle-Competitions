"""Supervised contrastive GR emission model plus smooth HMM decoding."""
from pathlib import Path
import argparse,glob,json,joblib
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data/train"
OUT=ROOT/"exp/results/supervised_gr_emission_hmm";OFF=np.arange(-60,60.1,2)
WIN=np.array([-20,-12,-8,-4,0,4,8,12,20],float);STRIDE=5
def robust(x):
 x=np.asarray(x,float); med=np.nanmedian(x);sc=np.nanpercentile(x,75)-np.nanpercentile(x,25)
 return (x-med)/max(float(sc),5.)
def sm(x,n):return pd.Series(x).rolling(n,center=True,min_periods=1).mean().to_numpy()

def prep(path,v4_delta,global_idx):
 wid=Path(path).name.split("__")[0];h=pd.read_csv(path);t=pd.read_csv(DATA/f"{wid}__typewell.csv").sort_values("TVT")
 ps=int(h.TVT_input.notna().sum())
 if ps<8 or ps>=len(h):return None
 anchor=float(h.TVT_input.iloc[ps-1]);rows=np.unique(np.r_[np.arange(ps-1,len(h),STRIDE),len(h)-1])
 hg=h.GR.interpolate(limit_direction="both").fillna(h.GR.median()).to_numpy(float)
 tg=t.GR.interpolate(limit_direction="both").fillna(t.GR.median()).to_numpy(float);tt=t.TVT.to_numpy(float)
 # visible-only affine GR calibration
 ref=np.interp(h.TVT_input.iloc[:ps],tt,tg)
 slope,intercept=np.linalg.lstsq(np.c_[ref,np.ones(ps)],hg[:ps],rcond=None)[0]
 slope=float(np.clip(slope,.25,2.5));intercept=float(np.clip(intercept,-100,100))
 hs=[robust(sm(hg,n)) for n in (5,21,61)];ts=[robust(sm(tg,n)) for n in (5,21,61)]
 base=np.zeros(len(h),np.float32);base[ps:]=v4_delta;b=base[rows]
 truth=h.TVT.to_numpy(float)[rows];md=h.MD.to_numpy(float);z=h.Z.to_numpy(float)
 prog=(md[rows]-md[ps-1])/max(md[-1]-md[ps-1],1)
 return dict(well=wid,h=h,t=t,ps=ps,anchor=anchor,rows=rows,hg=hg,tg=tg,tt=tt,
  hs=hs,ts=ts,slope=slope,intercept=intercept,base=b,base_suffix=np.asarray(v4_delta,np.float32),
  truth=truth,prog=prog,z=z,md=md,global_idx=np.asarray(global_idx,np.int64),n=len(h))

def emission_features(w,row_indices,candidate_tvt):
 """Vectorized candidates: one row per (station,candidate TVT)."""
 ri=np.asarray(row_indices,int); c=np.asarray(candidate_tvt,float)
 # Horizontal local samples use trajectory-index offsets scaled to approximate
 # the same nine-point window; typewell samples use true TVT coordinates.
 feats=[]
 for hx,tx in zip(w["hs"],w["ts"]):
  hp=np.stack([np.interp(ri+d,np.arange(len(hx)),hx) for d in (-20,-12,-8,-4,0,4,8,12,20)],1)
  tp=np.stack([np.interp(c+d,w["tt"],tx) for d in WIN],1)
  diff=hp-tp
  feats.extend([diff,abs(diff),diff**2])
  feats.extend([diff.mean(1)[:,None],diff.std(1)[:,None],
                (np.gradient(hp,axis=1)*np.gradient(tp,axis=1)).mean(1)[:,None]])
 # Candidate-local typewell curvature and trajectory context.
 tc=np.interp(c,w["tt"],w["ts"][1]);tl=np.interp(c-4,w["tt"],w["ts"][1]);tr=np.interp(c+4,w["tt"],w["ts"][1])
 pos=np.searchsorted(w["rows"],ri).clip(0,len(w["rows"])-1)
 feats.extend([(tr-tl)[:,None],(tr-2*tc+tl)[:,None],
   w["prog"][pos,None],((w["z"][ri]-w["z"][w["ps"]-1])/50)[:,None],
   (w["base"][pos]/30)[:,None]])
 return np.concatenate(feats,1).astype(np.float32)

def train(items,max_stations,seed):
 rng=np.random.default_rng(seed);XX=[];yy=[]
 neg=np.array([-60,-44,-32,-24,-16,-10,-6,-2,2,6,10,16,24,32,44,60],float)
 for w in items:
  pos=np.flatnonzero(w["rows"]>=w["ps"])
  if len(pos)>max_stations:pos=np.sort(rng.choice(pos,max_stations,False))
  ri=w["rows"][pos];true=w["truth"][pos]
  XX.append(emission_features(w,ri,true));yy.append(np.ones(len(ri),np.uint8))
  # Hard shifts cover near-confusable and distant candidates symmetrically.
  shifts=rng.choice(neg,(len(ri),8),replace=True)
  XX.append(emission_features(w,np.repeat(ri,8),(true[:,None]+shifts).ravel()))
  yy.append(np.zeros(len(ri)*8,np.uint8))
 X=np.concatenate(XX);y=np.concatenate(yy)
 model=LGBMClassifier(n_estimators=300,learning_rate=.04,num_leaves=31,
   min_child_samples=120,subsample=.85,colsample_bytree=.8,reg_lambda=3,
   class_weight={0:1,1:8},n_jobs=-1,verbosity=-1,random_state=seed)
 model.fit(X,y);return model

def decode(score,base):
 n,k=score.shape;unary=-np.log(np.clip(score,1e-7,1));zero=np.argmin(abs(OFF))
 prev=unary[0]+.12*(np.arange(k)-zero)**2;back=np.zeros((n,k),np.int16);jj=np.arange(k)
 # Residual state is allowed to move both directions; transition mean follows
 # the local V4 residual-frame change (zero), with robust heavy-tail allowance.
 for i in range(1,n):
  cur=np.full(k,np.inf)
  for d in range(-6,7):
   src=jj-d;ok=(src>=0)&(src<k)
   penalty=.055*d*d+.025*abs(d)
   val=prev[src[ok]]+penalty;imp=val<cur[ok];dst=jj[ok][imp]
   cur[dst]=val[imp];back[i,dst]=src[ok][imp]
  prev=cur+unary[i]
 p=np.empty(n,np.int16);p[-1]=np.argmin(prev)
 for i in range(n-1,0,-1):p[i-1]=back[i,p[i]]
 return OFF[p]

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--folds",type=int,default=1);ap.add_argument("--max-wells",type=int,default=200)
 ap.add_argument("--max-stations",type=int,default=100);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 feat=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl");vo=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"])
 v4={w:(vo[q.index].astype(np.float32),q.index.to_numpy()) for w,q in feat.groupby("well",sort=False)}
 paths=sorted(glob.glob(str(DATA/"*__horizontal_well.csv")))
 if a.max_wells:paths=paths[:a.max_wells]
 wells=[x for x in (prep(p,*v4[Path(p).name.split("__")[0]]) for p in paths) if x]
 splits=list(GroupKFold(min(5,len(wells))).split(wells,groups=[w["well"] for w in wells]))
 oof=np.full(len(feat),np.nan,np.float32);fid=np.full(len(feat),-1,np.int8);fr=[]
 for f,(ti,vi) in enumerate(splits[:a.folds]):
  model=train([wells[i] for i in ti],a.max_stations,2026+f);se=bs=nr=0
  for j in vi:
   w=wells[j];ri=w["rows"];center=w["anchor"]+w["base"]
   R=np.repeat(ri,len(OFF));C=(center[:,None]+OFF).ravel()
   score=model.predict_proba(emission_features(w,R,C))[:,1].reshape(len(ri),len(OFF))
   pred=decode(score,w["base"]);m=ri>=w["ps"];res=w["truth"]-center
   se+=np.square(pred[m]-res[m]).sum();bs+=np.square(res[m]).sum();nr+=m.sum()
   suffix=np.arange(w["ps"],w["n"]);corr=np.interp(suffix,ri,pred).astype(np.float32)
   oof[w["global_idx"]]=w["base_suffix"]+corr;fid[w["global_idx"]]=f
  fr.append(dict(fold=f,rows=int(nr),rmse=float(np.sqrt(se/nr)),base_rmse=float(np.sqrt(bs/nr))))
  joblib.dump(model,OUT/f"fold{f}.joblib");print(json.dumps(fr[-1]),flush=True)
 total=float(np.sqrt(np.average([x["rmse"]**2 for x in fr],weights=[x["rows"] for x in fr])))
 base=float(np.sqrt(np.average([x["base_rmse"]**2 for x in fr],weights=[x["rows"] for x in fr])))
 summary=dict(pooled_rmse=total,base_rmse=base,gain=base-total,folds=fr,wells=len(wells),
  supervised_true_tvt_training_only=True,grouped_complete_well_cv=True,no_neighbors=True)
 np.savez_compressed(OUT/"oof_delta.npz",prediction=oof,fold=fid,target=feat.target.to_numpy(np.float32),ids=feat.id.to_numpy(str))
 (OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
