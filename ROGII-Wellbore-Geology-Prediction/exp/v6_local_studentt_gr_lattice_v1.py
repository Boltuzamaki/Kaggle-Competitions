"""Narrow robust GR/typewell likelihood lattice centered on immutable v6."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd

R=Path(__file__).resolve().parents[1];V6P=R/'exp/results/generative_curve_prior_expert_errors_v6/oof.npz';OUT=R/'exp/results/v6_local_studentt_gr_lattice_v1';OUT.mkdir(parents=True,exist_ok=True)
V=np.load(V6P,allow_pickle=True);W=V['wells'].astype(str);vm={w:i for i,w in enumerate(W)};G=np.linspace(0,1,128)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];groups=S['groups'].astype(str);base=.1*S['accepted']+.9*S['replacement'];cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)];ixs={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}
pilot=[w for w in pd.read_csv(R/'exp/results/setchell_complete_path/pilot_wells.csv').well.astype(str) if w in vm];confirm=[w for w in pd.read_csv(R/'exp/results/setchell_complete_path/confirmation_wells.csv').well.astype(str) if w in vm]

def affine(x,y):
 q=np.isfinite(x)&np.isfinite(y);x=x[q];y=y[q];A=np.c_[x,np.ones(len(x))];keep=np.ones(len(x),bool)
 for _ in range(4):
  co=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0];r=y-A@co;med=np.median(r);mad=1.4826*np.median(abs(r-med))+1e-6;keep=abs(r-med)<3*mad
 return co
def build(wells,tag):
 store={}; rec=[]
 for j,w in enumerate(wells):
  ix=ixs[w];h=pd.read_csv(R/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(R/'data/train'/f'{w}__typewell.csv').sort_values('TVT').dropna(subset=['TVT','GR']);k=int(h.TVT_input.notna().sum());assert len(ix)==len(h)-k
  center=base[ix]+np.interp(np.linspace(0,1,len(ix)),G,V['pred'][vm[w]]);f=np.linspace(0,1,len(ix));warm=1-np.exp(-np.arange(len(ix))/300)
  labels=[];paths=[]
  for d in [-3,-1.5,0,1.5,3]:
   for t in [-4,-2,0,2,4]:
    for b in [-2,0,2]:labels.append((d,t,b));paths.append(center+d*warm+t*f+b*4*f*(1-f))
  paths=np.asarray(paths);tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();last=float(h.TVT_input.iloc[k-1]);
  # Prefix-only amplitude calibration of typewell GR into horizontal GR domain.
  pt=np.interp(h.TVT_input.iloc[:k],tt,tg);co=affine(pt,h.GR.iloc[:k].to_numpy());predgr=co[0]*np.interp(last+paths,tt,tg)+co[1];obs=h.GR.iloc[k:].to_numpy();res=obs[None]-predgr
  pr= h.GR.iloc[:k].to_numpy()-(co[0]*pt+co[1]);scale=max(15.,1.4826*np.nanmedian(abs(pr-np.nanmedian(pr))))
  # Cache sufficient robust losses for the small nu/floor grid.
  ev={}
  for nu in [3.,10.]:
   for floor in [20.,35.,45.]:ev[(nu,floor)]=np.nanmean(.5*(nu+1)*np.log1p((res/max(scale,floor))**2/nu),axis=1)
  store[w]=(y[ix],center,paths,np.asarray(labels),ev)
  rec.append({'well':w,'rows':len(ix),'center_rmse':float(np.sqrt(np.mean((y[ix]-center)**2))),'scale':scale})
  if (j+1)%30==0:print(tag,j+1,flush=True)
 pd.DataFrame(rec).to_csv(OUT/f'{tag}_wells.csv',index=False);return store
def evaluate(st,cfg,details=False):
 yy=[];bb=[];pp=[];wins=0
 for yt,cen,paths,lab,ev in st.values():
  complexity=(lab[:,0]/3)**2+(lab[:,1]/4)**2+(lab[:,2]/2)**2;score=-ev[(cfg['nu'],cfg['floor'])]-cfg['prior']*complexity
  temp=np.std(score)*cfg['temp']+1e-8;ww=np.exp(np.clip((score-score.max())/temp,-30,0));ww/=ww.sum();p=(1-cfg['blend'])*cen+cfg['blend']*(ww@paths);wins+=np.mean((yt-p)**2)<np.mean((yt-cen)**2);yy.append(yt);bb.append(cen);pp.append(p)
 yy=np.concatenate(yy);bb=np.concatenate(bb);pp=np.concatenate(pp);return float(np.sqrt(np.mean((yy-pp)**2))),float(np.sqrt(np.mean((yy-bb)**2))),wins/len(st)
ps=build(pilot,'pilot');grid=[]
for nu in [3.,10.]:
 for floor in [20.,35.,45.]:
  for prior in [.02,.1,.5,2.]:
   for temp in [.35,.7,1.5]:
    for blend in [.2,.5,1.]:
     cfg={'nu':nu,'floor':floor,'prior':prior,'temp':temp,'blend':blend};sc,ba,wi=evaluate(ps,cfg);grid.append({**cfg,'rmse':sc,'baseline':ba,'well_win_rate':wi})
tab=pd.DataFrame(grid).sort_values('rmse');tab.to_csv(OUT/'pilot_grid.csv',index=False);locked={k:float(tab.iloc[0][k]) for k in ['nu','floor','prior','temp','blend']}
cs=build(confirm,'confirmation');sc,ba,wi=evaluate(cs,locked);gain=ba-sc;gate=bool(gain>=.03 and wi>=.55)
summary={'protocol':'fixed legacy 120-well pilot then disjoint 240-well confirmation','v6_sha256':hashlib.sha256(V6P.read_bytes()).hexdigest(),'pilot_baseline':float(tab.iloc[0].baseline),'pilot_best':float(tab.iloc[0].rmse),'locked':locked,'confirmation_baseline':ba,'confirmation':sc,'confirmation_gain':gain,'well_win_rate':wi,'full_gate':'>=0.03 ft and >=55% well wins','passed':gate,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
