"""Graph-Laplacian/Sylvester reconciliation of multiscale GR strat shifts."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

R=Path(__file__).resolve().parents[1];SRC=R/'exp/results/legal_level2_all_oof_v1/oof.npz';OUT=R/'exp/results/sylvester_gr_landmark_graph_v1';OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];g=S['groups'].astype(str);center=.1*S['accepted']+.9*S['replacement']+np.load(SRC,allow_pickle=True)['anchored_difference'];err=y-center;W=np.sort(np.unique(g));cuts=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1,len(g)];ixs={g[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])};offs=np.linspace(-12,12,49);profiles=[];pose=[];T=[];lens=[]
def affine(x,y):
 q=np.isfinite(x)&np.isfinite(y);return np.linalg.lstsq(np.c_[x[q],np.ones(q.sum())],y[q],rcond=None)[0]
for j,w in enumerate(W):
 ix=ixs[w];h=pd.read_csv(R/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(R/'data/train'/f'{w}__typewell.csv').sort_values('TVT').dropna(subset=['TVT','GR']);k=h.TVT_input.notna().sum();q=h.iloc[k:];tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();pt=np.interp(h.TVT_input.iloc[:k],tt,tg);co=affine(pt,h.GR.iloc[:k]);last=float(h.TVT_input.iloc[k-1]);cost=np.zeros(len(offs));land=[]
 for sig in [0,4,16]:
  hg=q.GR.to_numpy();tgs=gaussian_filter1d(tg,sig) if sig else tg;hgs=gaussian_filter1d(np.nan_to_num(hg,nan=np.nanmedian(hg)),sig) if sig else hg
  ref=co[0]*np.interp(last+center[ix][None,:]+offs[:,None],tt,tgs)+co[1];res=hgs[None]-ref;scale=max(20,1.4826*np.nanmedian(abs((h.GR.iloc[:k].to_numpy()-(co[0]*pt+co[1]))-np.nanmedian(h.GR.iloc[:k].to_numpy()-(co[0]*pt+co[1])))));cost+=np.nanmean(2*np.log1p((res/scale)**2/3),axis=1)
  grad=np.gradient(np.nan_to_num(hgs,nan=np.nanmedian(hgs)));land += [np.quantile(abs(grad),.5),np.quantile(abs(grad),.9),np.mean(grad>np.quantile(grad,.9))]
 prof=-(cost-cost.mean())/(cost.std()+1e-8);profiles.append(np.r_[prof,land]);dx=q.X.iloc[-1]-q.X.iloc[0];dy=q.Y.iloc[-1]-q.Y.iloc[0];pose.append([q.X.iloc[0],q.Y.iloc[0],q.Z.iloc[0],dx,dy]);
 t=np.linspace(0,1,len(ix));T.append(np.linalg.lstsq(np.c_[np.ones(len(ix)),t],err[ix],rcond=None)[0][0]);lens.append(len(ix))
 if (j+1)%100==0:print('features',j+1,flush=True)
P=StandardScaler().fit_transform(np.asarray(profiles));XY=StandardScaler().fit_transform(np.asarray(pose));T=np.asarray(T);lens=np.asarray(lens);gf=list(GroupKFold(5).split(W,groups=W));sf=pd.read_csv(R/'exp/results/spatial_residual_graph/wells.csv').set_index('wid').loc[W].spatial_fold.to_numpy()
def solve_split(tr,va,cfg):
 # Pairwise observed shift is posterior-profile centroid difference.
 mu=(np.exp(P[:,:49]-P[:,:49].max(1,keepdims=True))@offs)/(np.exp(P[:,:49]-P[:,:49].max(1,keepdims=True)).sum(1)+1e-9)
 dprof=((P[:,None]-P[None])**2).mean(2);dxy=((XY[:,None,:2]-XY[None,:,:2])**2).mean(2);D=dprof+cfg['spatial']*dxy;np.fill_diagonal(D,np.inf);A=np.zeros_like(D)
 for i in range(len(W)):
  nn=np.argsort(D[i])[:cfg['k']];A[i,nn]=np.exp(-D[i,nn]/(np.median(D[i,nn])+1e-8))
 A=(A+A.T)/2;c=T[tr]-mu[tr];L=np.diag(A.sum(1))-A;Lqq=L[np.ix_(va,va)]+1e-4*np.eye(len(va));rhs=-L[np.ix_(va,tr)]@c;return cfg['alpha']*(mu[va]+np.linalg.solve(Lqq,rhs))
def score_split(va,p):
 se0=se1=n=0;wins=0
 for i,a in zip(va,p):
  ix=ixs[W[i]];se0+=float(err[ix]@err[ix]);se1+=float((err[ix]-a)@(err[ix]-a));n+=len(ix);wins+=np.mean((err[ix]-a)**2)<np.mean(err[ix]**2)
 return np.sqrt(se1/n),np.sqrt(se0/n),wins/len(va)
# Lock on standard GKF fold 0 only.
tr0,va0=gf[0];grid=[]
for k in [8,20,40]:
 for sp in [.1,.5,2,8]:
  for a in [.1,.2,.35,.5,1]:
   cfg={'k':k,'spatial':sp,'alpha':a};p=solve_split(tr0,va0,cfg);sc,ba,wi=score_split(va0,p);grid.append({**cfg,'rmse':sc,'baseline':ba,'win_rate':wi})
d=pd.DataFrame(grid).sort_values('rmse');d.to_csv(OUT/'pilot_grid.csv',index=False);lock={k:(int(d.iloc[0][k]) if k=='k' else float(d.iloc[0][k])) for k in ['k','spatial','alpha']};confirm=[]
for f in range(1,5):tr,va=gf[f];p=solve_split(tr,va,lock);a,b,w=score_split(va,p);confirm.append({'fold':f,'rmse':a,'baseline':b,'gain':b-a,'win_rate':w})
spatial=[]
for f in range(5):va=np.flatnonzero(sf==f);tr=np.flatnonzero(sf!=f);p=solve_split(tr,va,lock);a,b,w=score_split(va,p);spatial.append({'fold':f,'rmse':a,'baseline':b,'gain':b-a,'win_rate':w})
gain=np.average([x['gain'] for x in confirm],weights=[lens[gf[x['fold']][1]].sum() for x in confirm]);summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'pilot_best':d.iloc[0].to_dict(),'locked':lock,'grouped_confirmation':confirm,'spatial_blocks':spatial,'passed':bool(gain>=.03 and sum(x['gain']>0 for x in confirm)>=3 and sum(x['gain']>0 for x in spatial)>=3),'gate':'>=.03 weighted grouped gain and >=3/4 grouped, >=3/5 spatial wins','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(OUT/'summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/'features.npz',wells=W,profile=P,pose=XY,target_datum=T);print(json.dumps(summary,indent=2))
