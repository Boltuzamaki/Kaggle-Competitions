"""Outer-trained patch likelihood ratio for candidate residual paths."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/learned_path_likelihood_ratio_v1';OUT.mkdir(parents=True,exist_ok=True)
z=np.load(ROOT/'exp/results/generative_curve_prior_expert_errors_v6/oof.npz');W=z['wells'].astype(str);C=z['true'];M=z['pred'];lens=z['length'];student=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);groups=student['groups'].astype(str);base=.1*student['accepted']+.9*student['replacement'];y=student['y'];folds=list(GroupKFold(5).split(W,groups=W));order=np.random.RandomState(1801).permutation(W);pilot=set(order[:100]);confirm=set(order[100:300]);evalset=set(W);raw={}
def load(w):
 if w in raw:return raw[w]
 h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(ROOT/'data/train'/f'{w}__typewell.csv').dropna().sort_values('TVT');ps=int(h.TVT_input.notna().sum());hg=pd.to_numeric(h.GR,errors='coerce').interpolate(limit_direction='both').to_numpy();tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();ref=np.interp(h.TVT_input.iloc[:ps],tt,tg);A=np.c_[ref,np.ones(ps)];co=np.linalg.lstsq(A,hg[:ps],rcond=None)[0];tg=tg*co[0]+co[1];raw[w]=(h,ps,hg,tt,tg);return raw[w]
def features(hg,tt,tg,path,idx):
 q=np.interp(path[idx],tt,tg);qm=np.interp(path[np.maximum(0,idx-4)],tt,tg);qp=np.interp(path[np.minimum(len(path)-1,idx+4)],tt,tg);x=hg[idx];xm=hg[np.maximum(0,idx-4)];xp=hg[np.minimum(len(hg)-1,idx+4)];return np.c_[x,q,x-q,(xp-xm),(qp-qm),(xp-xm)-(qp-qm),np.abs(x-q),np.abs((xp-xm)-(qp-qm))]
pred=M.copy();oracle=M.copy();ranks=[]
for fo,(tr,va0) in enumerate(folds):
 XX=[];YY=[]
 for i in tr:
  w=W[i];h,ps,hg,tt,tg=load(w);n=len(h)-ps;idx=np.linspace(ps,min(len(h)-1,ps+n-1),48).astype(int);truth=h.TVT.to_numpy();
  for d in [0,-20,-10,-5,-2,-1,1,2,5,10,20]:XX.append(features(hg,tt,tg,truth+d,idx));YY.extend([1 if d==0 else 0]*len(idx))
 X=np.vstack(XX);lab=np.asarray(YY);clf=LGBMClassifier(n_estimators=350,num_leaves=15,min_child_samples=100,learning_rate=.04,reg_lambda=10,verbosity=-1,n_jobs=8,class_weight='balanced',random_state=1900+fo).fit(X,lab);cp=PCA(20,random_state=1).fit(C[tr]-M[tr]);sd=np.std(cp.transform(C[tr]-M[tr]),0)
 for i in va0:
  if W[i] not in evalset:continue
  w=W[i];h,ps,hg,tt,tg=load(w);m=groups==w;n=m.sum();anchor=h.TVT_input.iloc[ps-1];rng=np.random.RandomState(20000+i);cur=[];score=[];err=[];idx=np.arange(ps,len(h),16)
  for k in range(64):
   d=cp.inverse_transform((rng.randn(20)*sd*.45)[None])[0];d-=d[0];q=M[i]+d;rr=np.interp(np.linspace(0,1,n),np.linspace(0,1,len(q)),q);path=np.r_[h.TVT_input.iloc[:ps].to_numpy(),anchor+base[m]+rr];p=clf.predict_proba(features(hg,tt,tg,path,idx))[:,1];logit=np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)));score.append(np.mean(np.clip(logit,-5,5)));cur.append(q);err.append(np.mean((C[i]-q)**2))
  score=np.asarray(score);ww=np.exp((score-score.max())/.2);ww/=ww.sum();pred[i]=ww@np.asarray(cur);oi=np.argmin(err);oracle[i]=cur[oi];ranks.append((w,int(np.where(np.argsort(-score)==oi)[0][0])+1))
 print('fold',fo,flush=True)
def exact(P,S):
 se=nn=0
 for i,w in enumerate(W):
  if w not in S:continue
  m=groups==w;n=m.sum();r=np.interp(np.linspace(0,1,n),np.linspace(0,1,len(P[i])),P[i]);se+=np.sum((y[m]-base[m]-r)**2);nn+=n
 return float(np.sqrt(se/nn))
def rep(S):q=[r for w,r in ranks if w in S];return {'wells':len(S),'student':exact(0*M,S),'v6':exact(M,S),'posterior':exact(pred,S),'oracle':exact(oracle,S),'median_oracle_rank':float(np.median(q))}
out={'pilot':rep(pilot),'confirmation':rep(confirm),'full':rep(set(W)),'prior_rank_reference':19,'protocol':'frozen outer GKF patch classifier; full replay; 48 positions/train well; score stride16'};np.savez_compressed(OUT/'pred_full.npz',wells=W,pred=pred,oracle=oracle,ranks=np.array(ranks,dtype=object));out['sha256']=hashlib.sha256((OUT/'pred_full.npz').read_bytes()).hexdigest();(OUT/'summary_full.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
