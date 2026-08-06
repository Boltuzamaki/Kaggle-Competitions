"""Sylvester-style multiscale landmark DP around immutable level2 center."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/wavelet_landmark_dtw_v1';OUT.mkdir(parents=True,exist_ok=True)
l=np.load(ROOT/'exp/results/legal_level2_all_oof_v1/oof.npz');s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);g=s['groups'].astype(str);W=np.array(sorted(set(g)));student=.1*s['accepted']+.9*s['replacement'];center=student+l['anchored_difference'];y=s['y'];rng=np.random.RandomState(1601);order=rng.permutation(W);pilot=set(order[:120]);confirm=set(order[120:360]);sh=np.arange(-8,9,dtype=float)
def norm(x):
 x=pd.Series(x).interpolate(limit_direction='both').to_numpy(float);a,b=np.quantile(x,[.01,.99]);return np.clip((x-a)/(b-a+1e-6),0,1)
def desc(x,coord):
 z=[norm(x)]
 for sc in [2,4,8,16]:
  sm=gaussian_filter1d(z[0],sc);z.extend([np.abs(np.gradient(sm,coord)),np.abs(np.gradient(np.gradient(sm,coord),coord))])
 Z=np.stack(z,1);med=np.median(Z,0);sd=np.median(np.abs(Z-med),0)*1.4826+1e-4;return (Z-med)/sd
def one(w):
 m=g==w;n=m.sum();h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(ROOT/'data/train'/f'{w}__typewell.csv').dropna().sort_values('TVT');ps=int(h.TVT_input.notna().sum());anchor=h.TVT_input.iloc[ps-1];idx=np.arange(0,n,8);idx=np.r_[idx,n-1] if idx[-1]!=n-1 else idx;hg=h.GR.iloc[ps:].to_numpy();H=desc(hg,h.MD.iloc[ps:].to_numpy())[idx];tt=tw.TVT.to_numpy();T=desc(tw.GR.to_numpy(),tt);path=anchor+center[m][idx]
 E=np.empty((len(idx),len(sh)))
 for j,d in enumerate(sh):
  q=path+d;V=np.stack([np.interp(q,tt,T[:,k],left=np.nan,right=np.nan) for k in range(T.shape[1])],1);r=np.abs(H-V);E[:,j]=np.nanmean(np.minimum(r,2.)**2,1);E[~np.isfinite(E[:,j]),j]=20
 dp=np.full_like(E,np.inf);back=np.zeros(E.shape,int);dp[0]=E[0]+.08*sh**2
 for t in range(1,len(idx)):
  for j in range(len(sh)):
   lo=max(0,j-1);hi=min(len(sh),j+2);q=dp[t-1,lo:hi]+.12*(sh[lo:hi]-sh[j])**2;k=lo+np.argmin(q);dp[t,j]=E[t,j]+q[k-lo];back[t,j]=k
 st=np.zeros(len(idx),int);st[-1]=np.argmin(dp[-1])
 for t in range(len(idx)-1,0,-1):st[t-1]=back[t,st[t]]
 shift=np.interp(np.arange(n),idx,sh[st]);shift*=np.minimum(1,np.arange(n)/200);pred=center[m]+.2*shift
 return {'w':w,'se0':float(np.sum((y[m]-center[m])**2)),'se':float(np.sum((y[m]-pred)**2)),'n':n,'mean_shift':float(np.mean(np.abs(shift)))}
rows=[one(w) for w in order[:360]]
def report(S):
 q=[r for r in rows if r['w'] in S];return {'wells':len(q),'base':float(np.sqrt(sum(r['se0'] for r in q)/sum(r['n'] for r in q))),'wavelet':float(np.sqrt(sum(r['se'] for r in q)/sum(r['n'] for r in q))),'well_wins':sum(r['se']<r['se0'] for r in q),'mean_abs_shift':float(np.mean([r['mean_shift'] for r in q]))}
out={'pilot':report(pilot),'confirmation':report(confirm),'locked':{'scales':[2,4,8,16],'offset':[-8,8,1],'stride':8,'cap':2,'transition':.12,'anchor_prior':.08,'blend':.2},'protocol':'fixed seed disjoint 120/240; target read only after path frozen'};pd.DataFrame(rows).to_csv(OUT/'well_rows.csv',index=False);(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['rows_sha256']=hashlib.sha256((OUT/'well_rows.csv').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
