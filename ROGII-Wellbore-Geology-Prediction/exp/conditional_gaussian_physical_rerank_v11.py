"""Conditional residual samples around v6, reranked by legal GR likelihood."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/conditional_physical_rerank_v11';OUT.mkdir(parents=True,exist_ok=True)
z=np.load(ROOT/'exp/results/generative_curve_prior_expert_errors_v6/oof.npz');W=z['wells'].astype(str);C=z['true'];M=z['pred'];lens=z['length'];folds=list(GroupKFold(5).split(W,groups=W));sel=set(np.random.RandomState(1110).choice(W,100,False));student=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);groups=student['groups'].astype(str);base=.1*student['accepted']+.9*student['replacement'];y=student['y'];pred=M.copy();oracle=M.copy();rank=[]
for fo,(tr,va0) in enumerate(folds):
 va=np.array([i for i in va0 if W[i] in sel]);
 if not len(va):continue
 cp=PCA(20,random_state=1).fit(C[tr]-M[tr]);coef=cp.transform(C[tr]-M[tr]);sd=np.std(coef,0)
 for i in va:
  w=W[i];mask=groups==w;n=mask.sum();h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(ROOT/'data/train'/f'{w}__typewell.csv').dropna().sort_values('TVT');ps=int(h.TVT_input.notna().sum());anchor=float(h.TVT_input.iloc[ps-1]);hg=pd.to_numeric(h.GR.iloc[ps:],errors='coerce').interpolate(limit_direction='both').to_numpy();tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();rng=np.random.RandomState(12000+i);cur=[];cost=[];err=[]
  for k in range(64):
   d=cp.inverse_transform((rng.randn(20)*sd*.45)[None])[0];d=gaussian_filter1d(d,2);d-=d[0];q=M[i]+d;rr=np.interp(np.linspace(0,1,n),np.linspace(0,1,len(q)),q);path=anchor+base[mask]+rr;mis=hg-np.interp(path,tt,tg,left=np.nan,right=np.nan);valid=np.isfinite(mis);lik=np.mean(np.log1p((mis[valid]/45)**2)) if valid.any() else 99;prior=2e3*np.mean(np.diff(rr,2)**2);cost.append(lik+prior);cur.append(q);err.append(np.mean((C[i]-q)**2))
  cost=np.asarray(cost);ww=np.exp(-(cost-cost.min())/.03);ww/=ww.sum();pred[i]=ww@np.asarray(cur);oi=np.argmin(err);oracle[i]=cur[oi];rank.append(np.argsort(cost).tolist().index(oi)+1)
def exact(P):
 se=nn=0
 for i,w in enumerate(W):
  if w not in sel:continue
  m=groups==w;n=m.sum();r=np.interp(np.linspace(0,1,n),np.linspace(0,1,len(P[i])),P[i]);se+=np.sum((y[m]-base[m]-r)**2);nn+=n
 return float(np.sqrt(se/nn))
out={'wells':100,'student':exact(0*M),'v6':exact(M),'posterior':exact(pred),'sample_oracle':exact(oracle),'median_oracle_likelihood_rank':float(np.median(rank))};np.savez_compressed(OUT/'pilot_100.npz',wells=W,pred=pred,oracle=oracle,selected=np.array(sorted(sel)));out['sha256']=hashlib.sha256((OUT/'pilot_100.npz').read_bytes()).hexdigest();(OUT/'pilot_100.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
