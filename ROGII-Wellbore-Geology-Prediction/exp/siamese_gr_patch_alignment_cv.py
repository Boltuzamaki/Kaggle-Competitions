"""GPU Siamese GR patch embeddings and bounded emission-path decoding."""
from pathlib import Path
import argparse,glob,json,joblib
import numpy as np,pandas as pd,torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data/train"
OUT=ROOT/"exp/results/siamese_gr_patch_alignment"
PATCH=np.linspace(-20,20,21);NEG=np.arange(-60,60.1,4);OFF=np.arange(-60,60.1,2);STRIDE=5

def norm_patch(a):
 a=np.asarray(a,np.float32);return (a-a.mean(-1,keepdims=True))/np.maximum(a.std(-1,keepdims=True),3.)
def load(path,v4delta,gidx):
 wid=Path(path).name.split("__")[0];h=pd.read_csv(path);t=pd.read_csv(DATA/f"{wid}__typewell.csv").sort_values("TVT")
 ps=int(h.TVT_input.notna().sum())
 if ps<20 or ps>=len(h):return None
 hg=h.GR.interpolate(limit_direction="both").fillna(h.GR.median()).to_numpy(float)
 tg=t.GR.interpolate(limit_direction="both").fillna(t.GR.median()).to_numpy(float);tt=t.TVT.to_numpy(float)
 rows=np.unique(np.r_[np.arange(ps-1,len(h),STRIDE),len(h)-1])
 base=np.zeros(len(h),np.float32);base[ps:]=v4delta
 md=h.MD.to_numpy(float);x=h.X.to_numpy(float);y=h.Y.to_numpy(float);z=h.Z.to_numpy(float)
 prog=(md-md[ps-1])/max(md[-1]-md[ps-1],1)
 dx=np.gradient(x);dy=np.gradient(y);dz=np.gradient(z)
 ctx=np.c_[prog,dz,np.hypot(dx,dy),np.sin(np.arctan2(dy,dx)),np.cos(np.arctan2(dy,dx))]
 return dict(well=wid,h=h,ps=ps,hg=hg,tg=tg,tt=tt,rows=rows,base=base,
  truth=h.TVT.to_numpy(float),anchor=float(h.TVT_input.iloc[ps-1]),ctx=ctx.astype(np.float32),
  base_suffix=np.asarray(v4delta,np.float32),global_idx=np.asarray(gidx,np.int64),n=len(h))

def hpatch(w,rows):
 rows=np.asarray(rows,float)
 raw=np.stack([np.interp(rows+d,np.arange(len(w["hg"])),w["hg"]) for d in PATCH],-1)
 raw=norm_patch(raw);smooth=pd.Series(w["hg"]).rolling(21,center=True,min_periods=1).mean().to_numpy()
 sm=np.stack([np.interp(rows+d,np.arange(len(smooth)),smooth) for d in PATCH],-1)
 sm=norm_patch(sm);grad=np.gradient(raw,axis=-1)
 return np.stack([raw,sm,grad],1).astype(np.float32)
def tpatch(w,centers):
 c=np.asarray(centers,float).reshape(-1)
 raw=np.stack([np.interp(c+d,w["tt"],w["tg"]) for d in PATCH],-1)
 raw=norm_patch(raw);smooth=pd.Series(w["tg"]).rolling(21,center=True,min_periods=1).mean().to_numpy()
 sm=np.stack([np.interp(c+d,w["tt"],smooth) for d in PATCH],-1)
 sm=norm_patch(sm);grad=np.gradient(raw,axis=-1)
 return np.stack([raw,sm,grad],1).astype(np.float32)

class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(3,32,5,padding=2),nn.GELU(),
   nn.Conv1d(32,48,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1))
 def forward(self,x):return F.normalize(self.net(x).squeeze(-1),dim=-1)
class Siamese(nn.Module):
 def __init__(self):
  super().__init__();self.he=Encoder();self.te=Encoder();self.ctx=nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,48))
  self.logtemp=nn.Parameter(torch.tensor(np.log(12.),dtype=torch.float32))
 def forward(self,h,t,c):
  q=F.normalize(self.he(h)+self.ctx(c),dim=-1);b,k=t.shape[:2]
  z=self.te(t.reshape(b*k,*t.shape[2:])).reshape(b,k,-1)
  return torch.einsum("bd,bkd->bk",q,z)*self.logtemp.exp().clamp(2,30)

def batches(items,max_station,rng):
 samples=[]
 for wi,w in enumerate(items):
  ix=np.arange(w["ps"],w["n"])
  if len(ix)>max_station:ix=rng.choice(ix,max_station,False)
  samples.extend((wi,int(r)) for r in ix)
 rng.shuffle(samples);return samples
def make_batch(items,samples,candidates):
 hp=[];tp=[];cc=[]
 for wi,r in samples:
  w=items[wi];cent=w["truth"][r]+candidates
  hp.append(hpatch(w,[r])[0]);tp.append(tpatch(w,cent));cc.append(w["ctx"][r])
 return map(torch.from_numpy,(np.stack(hp),np.stack(tp),np.stack(cc)))

def train(items,dev,epochs,max_station):
 model=Siamese().to(dev);opt=torch.optim.AdamW(model.parameters(),8e-4,weight_decay=2e-4)
 rng=np.random.default_rng(2026)
 for ep in range(epochs):
  model.train();samples=batches(items,max_station,rng);losses=[]
  for st in range(0,len(samples),96):
   h,t,c=make_batch(items,samples[st:st+96],NEG);h,t,c=h.to(dev),t.to(dev),c.to(dev)
   opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=dev.type,enabled=dev.type=="cuda"):
    score=model(h,t,c);target=torch.full((len(h),),len(NEG)//2,device=dev,dtype=torch.long)
    loss=F.cross_entropy(score,target,label_smoothing=.02)
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss))
  print(f"epoch={ep} loss={np.mean(losses):.5f}",flush=True)
 return model

@torch.no_grad()
def emissions(model,w,dev,batch=128):
 rows=w["rows"];out=[]
 for st in range(0,len(rows),batch):
  r=rows[st:st+batch];center=w["anchor"]+w["base"][r]
  hp=torch.from_numpy(hpatch(w,r)).to(dev)
  tp=np.stack([tpatch(w,c+OFF) for c in center])
  cc=torch.from_numpy(w["ctx"][r]).to(dev);tt=torch.from_numpy(tp).to(dev)
  out.append(model(hp,tt,cc).float().cpu().numpy())
 return np.concatenate(out)
def decode(score):
 u=-F.log_softmax(torch.from_numpy(score),-1).numpy();n,k=u.shape;zero=np.argmin(abs(OFF))
 # Exact visible-prefix anchor.
 u[0]=50;u[0,zero]=0;prev=u[0];back=np.zeros((n,k),np.int16);jj=np.arange(k)
 for i in range(1,n):
  cur=np.full(k,np.inf)
  for d in range(-4,5):
   src=jj-d;ok=(src>=0)&(src<k);v=prev[src[ok]]+.07*d*d
   imp=v<cur[ok];dst=jj[ok][imp];cur[dst]=v[imp];back[i,dst]=src[ok][imp]
  prev=cur+u[i]
 p=np.empty(n,np.int16);p[-1]=np.argmin(prev)
 for i in range(n-1,0,-1):p[i-1]=back[i,p[i]]
 return OFF[p]

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--max-wells",type=int,default=200);ap.add_argument("--epochs",type=int,default=8)
 ap.add_argument("--max-stations",type=int,default=140);a=ap.parse_args()
 torch.manual_seed(2026);dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");OUT.mkdir(parents=True,exist_ok=True)
 feat=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl");vo=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"])
 v4={w:(vo[q.index].astype(np.float32),q.index.to_numpy()) for w,q in feat.groupby("well",sort=False)}
 paths=sorted(glob.glob(str(DATA/"*__horizontal_well.csv")))[:a.max_wells]
 wells=[q for q in (load(p,*v4[Path(p).name.split("__")[0]]) for p in paths) if q]
 ti,vi=list(GroupKFold(5).split(wells,groups=[w["well"] for w in wells]))[0]
 model=train([wells[i] for i in ti],dev,a.epochs,a.max_stations);model.eval()
 se=bs=nr=top1=top3=oracle_se=0;oof=np.full(len(feat),np.nan,np.float32)
 for j in vi:
  w=wells[j];score=emissions(model,w,dev);pred=decode(score);m=w["rows"]>=w["ps"]
  res=w["truth"][w["rows"]]-(w["anchor"]+w["base"][w["rows"]])
  se+=np.square(pred[m]-res[m]).sum();bs+=np.square(res[m]).sum();nr+=m.sum()
  truecls=np.argmin(abs(OFF[None,:]-res[m,None]),axis=1);rank=np.argsort(score[m],axis=1)[:,::-1]
  top1+=(rank[:,0]==truecls).sum();top3+=np.any(rank[:,:3]==truecls[:,None],axis=1).sum()
  oracle=np.clip(np.rint(res[m]/2)*2,OFF[0],OFF[-1]);oracle_se+=np.square(oracle-res[m]).sum()
  suffix=np.arange(w["ps"],w["n"]);corr=np.interp(suffix,w["rows"],pred).astype(np.float32)
  oof[w["global_idx"]]=w["base_suffix"]+corr
 summary={"wells":len(wells),"train_wells":len(ti),"valid_wells":len(vi),"rows":int(nr),
  "pooled_rmse":float(np.sqrt(se/nr)),"v4_rmse":float(np.sqrt(bs/nr)),
  "oracle_lattice_rmse":float(np.sqrt(oracle_se/nr)),"emission_top1":float(top1/nr),
  "emission_top3":float(top3/nr),"epochs":a.epochs,"no_neighbors":True,
  "outer_fold_supervised_only":True,"per_station_learned_embeddings":True}
 torch.save(model.state_dict(),OUT/"fold0.pt")
 np.savez_compressed(OUT/"oof_delta.npz",prediction=oof,target=feat.target.to_numpy(np.float32),ids=feat.id.to_numpy(str))
 (OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
