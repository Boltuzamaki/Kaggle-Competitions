"""Rolling-prefix episodic continuation prior; strict outer-well pilot."""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'exp/results/episodic_curve_prior';OUT.mkdir(parents=True,exist_ok=True)
L=96
def rs(x): return np.interp(np.linspace(0,1,L),np.linspace(0,1,len(x)),np.asarray(x,float))
def feats(h,cut,end):
    hist=h.iloc[:cut]; fut=h.iloc[cut:end]
    q=[]
    # Information available at a cut: full future acquisition/trajectory and
    # only past TVT. Use identical construction at the organizer PS.
    for c in ['MD','Z','X','Y','GR']:
        x=rs(fut[c]); x=x-x[0] if c!='GR' else x
        q.extend(x.reshape(12,8).mean(1));q.extend([x.mean(),x.std(),x[-1]-x[0],np.quantile(x,.1),np.quantile(x,.9)])
    x=hist.MD.to_numpy();y=hist.TVT.to_numpy()
    for n in [50,150,400]:
        n=min(n,len(x)); co=np.polyfit(x[-n:]-x[-1],y[-n:]-y[-1],min(2,n-1))
        q.extend([co[-2] if len(co)>1 else 0,co[-3] if len(co)>2 else 0])
    q.extend([len(hist),len(fut),np.ptp(hist.MD),np.ptp(fut.MD)])
    return np.nan_to_num(q,nan=0,posinf=0,neginf=0)
def target(h,cut,end):
    g=h.iloc[cut:end]; horizon=max(float(np.ptp(g.MD)),1.)
    return rs(((g.TVT-g.TVT.iloc[0])-(g.Z-g.Z.iloc[0]))/horizon)
def rm(y,p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

limit=int(sys.argv[1]) if len(sys.argv)>1 else 200
vf=pd.read_pickle(ROOT/'r_v4b/train_feats.pkl'); wells=np.array(sorted(vf.well.unique()))
if limit and limit<len(wells): wells=np.random.RandomState(1801).choice(wells,limit,False)
raw={}; episodes=[]; query=[]; truth=[]; dz=[]; lens=[]
for j,w in enumerate(sorted(wells)):
    h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv')
    ps=int(h.TVT_input.notna().sum()); raw[w]=h
    if ps<100: continue
    # Late rolling origins stay in the lateral regime; early build-section
    # episodes are a known, severe domain mismatch for the real suffix.
    for frac in [.82,.88,.93,.97]:
        cut=max(30,int(ps*frac)); episodes.append((w,feats(h,cut,ps),target(h,cut,ps)))
    query.append(feats(h,ps,len(h))); truth.append(target(h,ps,len(h)))
    g=h.iloc[ps:]; dz.append(rs((g.Z-g.Z.iloc[0]).to_numpy()));lens.append(len(g))
qw=np.array(sorted(raw)); Q=np.asarray(query); T=np.asarray(truth);DZ=np.asarray(dz); lens=np.asarray(lens)
pred=np.zeros_like(T); rank=[]; oracle=np.zeros_like(T); foldrows=[]
for fold,(tr,va) in enumerate(GroupKFold(5).split(Q,groups=qw)):
    trw=set(qw[tr]); ee=[e for e in episodes if e[0] in trw]
    EX=np.asarray([e[1] for e in ee]); EY=np.asarray([e[2] for e in ee])
    sc=StandardScaler().fit(EX); a=sc.transform(EX); b=sc.transform(Q[va])
    cp=PCA(20,random_state=1).fit(EY); z=cp.transform(EY)
    m=ExtraTreesRegressor(n_estimators=350,min_samples_leaf=10,max_features=.65,n_jobs=-1,random_state=fold).fit(a,z)
    pred[va]=cp.inverse_transform(m.predict(b))
    # Oracle episodic continuation and feature rank diagnose analogue selection.
    dd=((b[:,None]-a[None])**2).mean(2); order=np.argsort(dd,axis=1)
    for jj,v in enumerate(va):
        er=((EY-T[v])**2).mean(1); oi=np.argmin(er);oracle[v]=EY[oi];rank.append(np.where(order[jj]==oi)[0][0]+1)
    foldrows.append({'fold':fold,'episodic_oracle':rm(T[va],oracle[va]),'generated_residual':rm(T[va],pred[va]),'zero':rm(T[va],0*T[va])})
    print(foldrows[-1],flush=True)
# Exact-real-suffix score using raw dZ baseline and Student baseline where same sampled wells.
z=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True); student=.1*z['accepted']+.9*z['replacement']; groups=z['groups'].astype(str);y=z['y']
grid=[]
for blend in [0,.1,.25,.5,.75,1]:
    yy=[];pp=[]
    for i,w in enumerate(qw):
        mask=groups==w; n=mask.sum(); horizon=max(float(np.ptp(raw[w].iloc[int(raw[w].TVT_input.notna().sum()):].MD)),1.)
        corr=horizon*np.interp(np.linspace(0,1,n),np.linspace(0,1,L),pred[i]); geo=np.interp(np.linspace(0,1,n),np.linspace(0,1,L),DZ[i])+corr
        yy.append(y[mask]);pp.append((1-blend)*student[mask]+blend*geo)
    grid.append({'blend_geo':blend,'rmse':rm(np.concatenate(yy),np.concatenate(pp))})
summary={'wells':len(qw),'episodes':len(episodes),'median_oracle_rank':float(np.median(rank)),'folds':foldrows,'grid':grid,'best':min(grid,key=lambda x:x['rmse'])}
print(json.dumps(summary,indent=2));(OUT/f'pilot_{limit}.json').write_text(json.dumps(summary,indent=2))
