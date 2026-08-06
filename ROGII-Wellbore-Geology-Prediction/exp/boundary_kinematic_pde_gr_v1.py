"""Boundary plane/PDE propagation with robust GR update around L2 center."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd

R=Path(__file__).resolve().parents[1];SRC=R/'exp/results/legal_level2_all_oof_v1/oof.npz';OUT=R/'exp/results/boundary_kinematic_pde_gr_v1';OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];g=S['groups'].astype(str);center=.1*S['accepted']+.9*S['replacement']+np.load(SRC,allow_pickle=True)['anchored_difference'];cuts=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1,len(g)];ixs={g[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])};common=set(ixs)
pilot=[w for w in pd.read_csv(R/'exp/results/setchell_complete_path/pilot_wells.csv').well.astype(str) if w in common];confirm=[w for w in pd.read_csv(R/'exp/results/setchell_complete_path/confirmation_wells.csv').well.astype(str) if w in common];scales=np.array([0,.02,.05,.1,.2])
def robust_plane(X,z,ridge):
 X=X-X[-1];z=z-z[-1];A=np.c_[X,np.ones(len(X))];keep=np.ones(len(X),bool)
 for _ in range(4):
  co=np.linalg.solve(A[keep].T@A[keep]+np.diag([ridge,ridge,1e-6]),A[keep].T@z[keep]);rr=z-A@co;mad=1.4826*np.median(abs(rr-np.median(rr)))+1e-5;keep=abs(rr-np.median(rr))<3*mad
 return co
def affine(x,y):
 A=np.c_[x,np.ones(len(x))];q=np.isfinite(x)&np.isfinite(y);return np.linalg.lstsq(A[q],y[q],rcond=None)[0]
def build(wells,tag):
 st={};rows=[]
 for j,w in enumerate(wells):
  ix=ixs[w];h=pd.read_csv(R/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(R/'data/train'/f'{w}__typewell.csv').sort_values('TVT').dropna(subset=['TVT','GR']);k=h.TVT_input.notna().sum();q=h.iloc[k:];last=h.iloc[k-1];tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();pt=np.interp(h.TVT_input.iloc[:k],tt,tg);ac=affine(pt,h.GR.iloc[:k]);obs=q.GR.to_numpy();families={}
  for win in [200,500,1500]:
   p=h.iloc[max(0,k-win):k];XY=p[['X','Y']].to_numpy();ss=(p.TVT_input+p.Z).to_numpy()
   for ridge in [1e2,1e4]:
    co=robust_plane(XY,ss,ridge);dxy=q[['X','Y']].to_numpy()-XY[-1];sdelta=dxy@co[:2];phys=sdelta-(q.Z.to_numpy()-last.Z);corr=np.clip(phys-center[ix],-25,25);paths=center[ix][None]+scales[:,None]*corr
    predgr=ac[0]*np.interp(float(last.TVT_input)+paths,tt,tg)+ac[1];res=obs[None]-predgr;pr=h.GR.iloc[:k].to_numpy()-(ac[0]*pt+ac[1]);sig=max(20,1.4826*np.nanmedian(abs(pr-np.nanmedian(pr))));ev={nu:np.nanmean(.5*(nu+1)*np.log1p((res/sig)**2/nu),axis=1) for nu in [3.,10.]};families[(win,ridge)]=(paths,ev)
  st[w]=(y[ix],center[ix],families);rows.append({'well':w,'rows':len(ix)});
  if (j+1)%30==0:print(tag,j+1,flush=True)
 return st
def evaluate(st,cfg):
 yy=[];bb=[];pp=[];wins=0
 for yt,cen,fam in st.values():
  paths,ev=fam[(cfg['win'],cfg['ridge'])];score=-ev[cfg['nu']]-cfg['prior']*(scales/.1)**2;temp=np.std(score)*cfg['temp']+1e-8;ww=np.exp(np.clip((score-score.max())/temp,-30,0));ww/=ww.sum();p=ww@paths;wins+=np.mean((yt-p)**2)<np.mean((yt-cen)**2);yy.append(yt);bb.append(cen);pp.append(p)
 yy=np.concatenate(yy);bb=np.concatenate(bb);pp=np.concatenate(pp);rm=lambda a,b:float(np.sqrt(np.mean((a-b)**2)));return rm(yy,pp),rm(yy,bb),wins/len(st)
ps=build(pilot,'pilot');grid=[]
for win in [200,500,1500]:
 for ridge in [1e2,1e4]:
  for nu in [3.,10.]:
   for prior in [.02,.1,.5,2.]:
    for temp in [.35,.7,1.5]:
     c={'win':win,'ridge':ridge,'nu':nu,'prior':prior,'temp':temp};a,b,w=evaluate(ps,c);grid.append({**c,'rmse':a,'baseline':b,'well_win_rate':w})
d=pd.DataFrame(grid).sort_values('rmse');d.to_csv(OUT/'pilot_grid.csv',index=False);lock={k:(int(d.iloc[0][k]) if k=='win' else float(d.iloc[0][k])) for k in ['win','ridge','nu','prior','temp']};cs=build(confirm,'confirmation');a,b,w=evaluate(cs,lock);gain=b-a;summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'pilot_baseline':float(d.iloc[0].baseline),'pilot':float(d.iloc[0].rmse),'locked':lock,'confirmation_baseline':b,'confirmation':a,'gain':gain,'well_win_rate':w,'passed':bool(gain>=.03 and w>=.55),'gate':'>=.03 and >=55% wins','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
