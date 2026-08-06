"""Strict outer-GKF level-2 residual ensemble over versioned legal OOF curves."""
from pathlib import Path
import hashlib,json,os
import numpy as np,pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold

R=Path(__file__).resolve().parents[1];OUT=R/'exp/results'/os.environ.get('L2_RUN','legal_level2_all_oof_v1');OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];groups=S['groups'].astype(str);base=.1*S['accepted']+.9*S['replacement'];err=y-base
V=np.load(R/'exp/results/generative_curve_prior_expert_errors_v6/oof.npz',allow_pickle=True);W=V['wells'].astype(str);grid=np.linspace(0,1,128)
cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)];ixs={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}
folds=list(GroupKFold(5).split(W,groups=W));wf={w:f for f,(_,va) in enumerate(folds) for w in W[va]};rf=np.array([wf[w] for w in groups])
paths={
 'v1':R/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz','v3':R/'exp/results/generative_curve_prior_weighted_v3/oof.npz',
 'v4':R/'exp/results/generative_curve_prior_risk_v4/oof.npz','v6':R/'exp/results/generative_curve_prior_expert_errors_v6/oof.npz',
 'v7':R/'exp/results/generative_curve_prior_nested_v7/oof.npz','blup':R/'exp/results/functional_gaussian_prefix_conditioning_v1/predictions.npz',
 'fkernel':R/'exp/results/functional_operator_kernel_causal_prefix_v2/oof_curves.npz','tcn':R/'exp/results/causal_prefix_tcn_curve_v1/predictions.npz',
 'boost':R/'exp/results/causal_prefix_boost_fpca_v1/predictions.npz','metric':R/'exp/results/v6_supervised_metric_analogue_v1/oof.npz',
 'spatial':R/'exp/results/spatial_residual_graph/locked_correction.npz','setchell':R/'exp/results/setchell_fullfield_addone/correction.npz'}
if os.environ.get('INCLUDE_DENSITY')=='1':paths['density']=R/'exp/results/supervised_density_ratio_gr_full_gkf_v1/oof.npz'
manifest={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
def torow(wells,A):
 o=np.empty_like(y);mp={w:i for i,w in enumerate(wells.astype(str))}
 for w in W:
  c=A[mp[w]];ix=ixs[w];o[ix]=np.interp(np.linspace(0,1,len(ix)),np.linspace(0,1,len(c)),c)
 return o
legs={}
z=np.load(paths['v1'],allow_pickle=True);legs['v1_et']=torow(z['wells'],z['et'])
for n,k in [('v3','pred'),('v4','pred'),('v6','pred'),('v7','pred')]:z=np.load(paths[n],allow_pickle=True);legs[n]=torow(z['wells'],z[k])
z=np.load(paths['blup'],allow_pickle=True);legs['blup']=torow(z['wells'],z['curve'])
z=np.load(paths['fkernel'],allow_pickle=True);legs['fkernel']=z['row_prediction']
z=np.load(paths['tcn'],allow_pickle=True);legs['tcn']=torow(z['wells'],z['curve'])
z=np.load(paths['boost'],allow_pickle=True);legs['boost_d3']=torow(z['wells'],z['huber_d3']);legs['boost_d5']=torow(z['wells'],z['huber_d5'])
z=np.load(paths['metric'],allow_pickle=True);m=torow(z['wells'],z['pred']);c=z['confidence'];c=np.clip((c-np.quantile(c,.1))/(np.quantile(c,.9)-np.quantile(c,.1)+1e-8),0,1);cg=np.empty_like(y)
for w,a in zip(z['wells'].astype(str),c):cg[ixs[w]]=a
legs['metric_gate']=m*cg
# Spatial artifact used different folds: regenerate the locked graph on the standard outer GKF.
sw=pd.read_csv(R/'exp/results/spatial_residual_graph/wells.csv').set_index('wid').loc[W];xy=sw[['x','y']].to_numpy();xy=(xy-xy.mean(0))/(xy.std(0)+1e-9);ang=np.arctan2(sw.uy,sw.ux).to_numpy();sp=np.zeros_like(V['true'])
from sklearn.decomposition import PCA
for tr,va in folds:
 pc=PCA(8,random_state=1).fit(V['true'][tr]);A=pc.transform(V['true'][tr])
 for i in va:
  d=xy[tr]-xy[i];al=d[:,0]*sw.ux.iloc[i]+d[:,1]*sw.uy.iloc[i];cr=-d[:,0]*sw.uy.iloc[i]+d[:,1]*sw.ux.iloc[i];da=np.arccos(np.clip(np.cos(ang[tr]-ang[i]),-1,1));ww=np.exp(-.5*((al/1.2)**2+(cr/.66)**2+(da/1.2)**2));co=(ww[:,None]*A).sum(0)/(ww.sum()+4);sp[i]=pc.inverse_transform(co[None])[0]
legs['spatial_gkf']=torow(W,sp)
# Setchell exact ID mapping onto immutable rows.
z=np.load(paths['setchell'],allow_pickle=True);sm=dict(zip(z['id'].astype(str),z['correction']));fm=np.load(R/'exp/results/heel_calibrated_gr_datum/full_meta/oof.npz');f=pd.read_pickle(R/'r_v4b/train_feats.pkl').iloc[fm['global_indices'].astype(int)].reset_index(drop=True);ids=f.id.astype(str).to_numpy();legs['setchell']=np.array([sm.get(q,0.) for q in ids]);assert np.mean([q in sm for q in ids])>.999
if 'density' in paths:
 z=np.load(paths['density'],allow_pickle=True);assert np.array_equal(z['groups'].astype(str),groups);legs['density']=z['correction']
names=list(legs);X=np.column_stack([legs[n] for n in names]);assert np.isfinite(X).all()
# Per-fold sufficient statistics make inner selection and ablation exact/cheap.
Gs=[];hs=[];r2=[];ns=[]
for f0 in range(5):
 m=rf==f0;Gs.append(X[m].T@X[m]);hs.append(X[m].T@err[m]);r2.append(float(err[m]@err[m]));ns.append(int(m.sum()))
def solve(G,h,lam,kind):
 scale=np.trace(G)/len(h);L=lam*scale
 if kind=='signed':return np.linalg.solve(G+L*np.eye(len(h)),h)
 chol=np.linalg.cholesky(G+1e-8*np.eye(len(h)))
 return nnls(np.vstack([chol.T,np.sqrt(L)*np.eye(len(h))]),np.r_[np.linalg.solve(chol,h),np.zeros(len(h))])[0]
def run(cols,kind):
 oo=np.zeros(len(y));ws=[]
 for fo in range(5):
  train=[q for q in range(5) if q!=fo];best=None
  for lam in [1e-6,1e-4,1e-2,.1,1]:
   se=nn=0
   for iv in train:
    tt=[q for q in train if q!=iv];G=sum(Gs[q][np.ix_(cols,cols)] for q in tt);h=sum(hs[q][cols] for q in tt);w=solve(G,h,lam,kind);gv=Gs[iv][np.ix_(cols,cols)];hv=hs[iv][cols];se+=r2[iv]-2*w@hv+w@gv@w;nn+=ns[iv]
   sc=np.sqrt(se/nn)
   if best is None or sc<best[0]:best=(sc,lam)
  G=sum(Gs[q][np.ix_(cols,cols)] for q in train);h=sum(hs[q][cols] for q in train);w=solve(G,h,best[1],kind);m=rf==fo;oo[m]=X[m][:,cols]@w;ws.append({'fold':fo,'lambda':best[1],'weights':dict(zip([names[q] for q in cols],w.tolist()))})
 return oo,ws
results={};preds={}
for kind in ['nonnegative','signed']:
 p,w=run(list(range(len(names))),kind);preds[kind]=p;results[kind]={'rmse':float(np.sqrt(np.mean((err-p)**2))),'folds':[float(np.sqrt(np.mean((err[rf==f]-p[rf==f])**2))) for f in range(5)],'weights':w}
# Preserve v6 exactly and fit only small signed residual adjustments; this
# conservative parameterization cannot discard the strongest leg by accident.
vi=names.index('v6'); acols=[q for q in range(len(names)) if q!=vi]
def anchored(mode):
 A=X[:,acols] if mode=='raw' else X[:,acols]-X[:,[vi]]; target=err-X[:,vi]
 ga=[];ha=[];sa=[]
 for f0 in range(5):m=rf==f0;ga.append(A[m].T@A[m]);ha.append(A[m].T@target[m]);sa.append(float(target[m]@target[m]))
 oo=np.zeros(len(y));ww=[]
 for fo in range(5):
  train=[q for q in range(5) if q!=fo];best=None
  for lam in [1e-6,1e-4,1e-2,.1,1,10]:
   se=nn=0
   for iv in train:
    tt=[q for q in train if q!=iv];G=sum(ga[q] for q in tt);h=sum(ha[q] for q in tt);w=solve(G,h,lam,'signed');se+=sa[iv]-2*w@ha[iv]+w@ga[iv]@w;nn+=ns[iv]
   sc=np.sqrt(se/nn)
   if best is None or sc<best[0]:best=(sc,lam)
  G=sum(ga[q] for q in train);h=sum(ha[q] for q in train);w=solve(G,h,best[1],'signed');m=rf==fo;oo[m]=X[m,vi]+A[m]@w;ww.append({'fold':fo,'lambda':best[1],'weights':dict(zip([names[q] for q in acols],w.tolist()))})
 return oo,ww
for mode in ['raw','difference']:
 p,w=anchored(mode);key='anchored_'+mode;preds[key]=p;results[key]={'rmse':float(np.sqrt(np.mean((err-p)**2))),'folds':[float(np.sqrt(np.mean((err[rf==f]-p[rf==f])**2))) for f in range(5)],'weights':w}
# Leave-one-leg-out signed ablation.
abl=[]
for j,n in enumerate(names):
 p,_=run([q for q in range(len(names)) if q!=j],'signed');abl.append({'removed':n,'rmse':float(np.sqrt(np.mean((err-p)**2)))})
pd.DataFrame(abl).sort_values('rmse').to_csv(OUT/'ablation.csv',index=False);np.savez_compressed(OUT/'oof.npz',groups=groups,**preds)
summary={'baseline':float(np.sqrt(np.mean(err**2))),'legs':names,'artifact_sha256':manifest,'results':results,'ablation':abl,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['artifact_sha256']},indent=2))
